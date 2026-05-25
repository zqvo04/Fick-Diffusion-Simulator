"""
sampler.py — 콜로케이션 포인트 샘플링 전략

담당 역할:
  - 내부 PDE 잔차 포인트 (collocation points)
  - 경계조건(BC) 포인트
  - 초기조건(IC) 포인트

샘플링 전략:
  1. Uniform Random   : 단순 균일 샘플링
  2. Latin Hypercube  : 저불일치 샘플링 (pyDOE2 또는 scipy)
     → 같은 포인트 수 대비 도메인 커버리지 우월, Raissi et al. 원논문 채택 방식
"""

import torch
import numpy as np
from typing import Dict, Tuple


# ── LHS 구현 (scipy 활용, pyDOE2 fallback) ────────────────────────────────
def _lhs(n_samples: int, n_dims: int) -> np.ndarray:
    """
    Latin Hypercube Sampling. 결과: (n_samples, n_dims) ∈ [0, 1]
    scipy.stats.qmc.LatinHypercube 사용 (scipy >= 1.7)
    """
    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=n_dims, seed=42)
        return sampler.random(n=n_samples)
    except ImportError:
        # fallback: 단순 균일 랜덤
        return np.random.rand(n_samples, n_dims)


class DiffusionSampler:
    """
    2D Fick 확산 방정식을 위한 콜로케이션 포인트 샘플러.

    도메인:
        x ∈ [x_min, x_max]
        y ∈ [y_min, y_max]
        t ∈ [0, T]

    경계조건 설정 (Dirichlet):
        x = x_min, x_max : c = 0
        y = y_min, y_max : c = 0

    초기조건:
        t = 0 : c = c0(x, y)  (Gaussian 점 소스 또는 사용자 지정)

    Args:
        x_bounds  : (x_min, x_max)
        y_bounds  : (y_min, y_max)
        T         : 최대 시간
        n_pde     : PDE 콜로케이션 포인트 수
        n_bc      : 경계 포인트 수 (4면 합계)
        n_ic      : 초기조건 포인트 수
        strategy  : 'lhs' | 'uniform'
        device    : torch device
    """

    def __init__(
        self,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
        T: float = 1.0,
        n_pde: int = 10000,
        n_bc: int = 1000,
        n_ic: int = 1000,
        strategy: str = "lhs",
        device: torch.device = torch.device("cpu"),
    ):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.T = T
        self.n_pde = n_pde
        self.n_bc = n_bc
        self.n_ic = n_ic
        self.strategy = strategy
        self.device = device

    # ── 내부 유틸: numpy → torch (requires_grad=True) ──────────────────────
    def _to_tensor(self, arr: np.ndarray, requires_grad: bool = True) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32,
                            device=self.device, requires_grad=requires_grad)

    def _scale(self, raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """[0,1] → [lo, hi]"""
        return raw * (hi - lo) + lo

    # ── 1. PDE 콜로케이션 포인트 ──────────────────────────────────────────
    def sample_pde(self) -> Dict[str, torch.Tensor]:
        """
        내부 도메인 + 시간 전체에서 샘플링.
        반환: {'x': (N,1), 'y': (N,1), 't': (N,1)}  requires_grad=True
        """
        if self.strategy == "lhs":
            raw = _lhs(self.n_pde, n_dims=3)  # (N, 3) ∈ [0,1]
        else:
            raw = np.random.rand(self.n_pde, 3)

        x = self._scale(raw[:, 0:1], self.x_min, self.x_max)
        y = self._scale(raw[:, 1:2], self.y_min, self.y_max)
        t = self._scale(raw[:, 2:3], 0.0, self.T)

        return {
            "x": self._to_tensor(x),
            "y": self._to_tensor(y),
            "t": self._to_tensor(t),
        }

    # ── 2. 경계조건 포인트 (Dirichlet: c = 0 on all walls) ───────────────
    def sample_bc(self) -> Dict[str, torch.Tensor]:
        """
        4면 경계 균등 분배: n_bc // 4 포인트씩.

        face 0: x = x_min, y ∈ [y_min, y_max], t ∈ [0, T]
        face 1: x = x_max, y ∈ [y_min, y_max], t ∈ [0, T]
        face 2: y = y_min, x ∈ [x_min, x_max], t ∈ [0, T]
        face 3: y = y_max, x ∈ [x_min, x_max], t ∈ [0, T]
        반환: {'x', 'y', 't', 'c_target'}  (c_target = 0 for Dirichlet)
        """
        n = self.n_bc // 4
        pts = []

        for face in range(4):
            t_rand = np.random.rand(n, 1) * self.T
            free = np.random.rand(n, 1)

            if face == 0:  # x = x_min
                x = np.full((n, 1), self.x_min)
                y = self._scale(free, self.y_min, self.y_max)
            elif face == 1:  # x = x_max
                x = np.full((n, 1), self.x_max)
                y = self._scale(free, self.y_min, self.y_max)
            elif face == 2:  # y = y_min
                y = np.full((n, 1), self.y_min)
                x = self._scale(free, self.x_min, self.x_max)
            else:             # y = y_max
                y = np.full((n, 1), self.y_max)
                x = self._scale(free, self.x_min, self.x_max)

            pts.append((x, y, t_rand))

        x_bc = np.vstack([p[0] for p in pts])
        y_bc = np.vstack([p[1] for p in pts])
        t_bc = np.vstack([p[2] for p in pts])
        c_bc = np.zeros_like(x_bc)  # Dirichlet c = 0

        return {
            "x": self._to_tensor(x_bc),
            "y": self._to_tensor(y_bc),
            "t": self._to_tensor(t_bc),
            "c_target": self._to_tensor(c_bc, requires_grad=False),
        }

    # ── 3. 초기조건 포인트 ────────────────────────────────────────────────
    def sample_ic(
        self,
        ic_type: str = "gaussian",
        source_x: float = 0.5,
        source_y: float = 0.5,
        sigma: float = 0.05,
        c_max: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        t = 0에서의 초기 농도 분포 샘플링.

        ic_type='gaussian':
            c(x, y, 0) = c_max * exp(-((x-sx)²+(y-sy)²) / (2σ²))
            → 점 소스 (point source) 확산 시뮬레이션에 적합

        ic_type='uniform':
            c(x, y, 0) = c_max (전체 도메인 균일)

        반환: {'x', 'y', 't', 'c_target'}
        """
        if self.strategy == "lhs":
            raw = _lhs(self.n_ic, n_dims=2)
        else:
            raw = np.random.rand(self.n_ic, 2)

        x_ic = self._scale(raw[:, 0:1], self.x_min, self.x_max)
        y_ic = self._scale(raw[:, 1:2], self.y_min, self.y_max)
        t_ic = np.zeros((self.n_ic, 1))  # t = 0

        if ic_type == "gaussian":
            c_ic = c_max * np.exp(
                -((x_ic - source_x) ** 2 + (y_ic - source_y) ** 2) / (2 * sigma ** 2)
            )
        elif ic_type == "uniform":
            c_ic = np.full_like(x_ic, c_max)
        else:
            raise ValueError(f"Unknown ic_type: {ic_type}. Use 'gaussian' or 'uniform'.")

        return {
            "x": self._to_tensor(x_ic),
            "y": self._to_tensor(y_ic),
            "t": self._to_tensor(t_ic),
            "c_target": self._to_tensor(c_ic, requires_grad=False),
        }

    # ── 4. 전체 배치 한 번에 반환 (trainer에서 사용) ─────────────────────
    def sample_all(self, ic_kwargs: dict = None) -> Dict[str, Dict]:
        """
        PDE / BC / IC 포인트 전부 반환.
        ic_kwargs: sample_ic()에 전달할 키워드 인자
        """
        ic_kwargs = ic_kwargs or {}
        return {
            "pde": self.sample_pde(),
            "bc": self.sample_bc(),
            "ic": self.sample_ic(**ic_kwargs),
        }

    # ── 5. 시각화용 균일 격자 ────────────────────────────────────────────
    def make_grid(
        self, nx: int = 100, ny: int = 100, t_val: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        특정 시각 t=t_val에서의 2D 균일 격자 생성 (예측/시각화용).
        반환: {'x': (nx*ny,1), 'y': (nx*ny,1), 't': (nx*ny,1), 'shape': (nx, ny)}
        """
        xs = np.linspace(self.x_min, self.x_max, nx)
        ys = np.linspace(self.y_min, self.y_max, ny)
        XX, YY = np.meshgrid(xs, ys)  # (ny, nx)

        x_flat = XX.reshape(-1, 1)
        y_flat = YY.reshape(-1, 1)
        t_flat = np.full_like(x_flat, t_val)

        return {
            "x": self._to_tensor(x_flat, requires_grad=False),
            "y": self._to_tensor(y_flat, requires_grad=False),
            "t": self._to_tensor(t_flat, requires_grad=False),
            "shape": (ny, nx),
            "xs": xs,
            "ys": ys,
        }


# ── 빠른 동작 확인 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sampler = DiffusionSampler(
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        T=1.0,
        n_pde=10000,
        n_bc=1000,
        n_ic=1000,
        strategy="lhs",
    )

    batch = sampler.sample_all(ic_kwargs={"ic_type": "gaussian", "sigma": 0.05})

    print("=== PDE 포인트 ===")
    print(f"  x: {batch['pde']['x'].shape}, requires_grad={batch['pde']['x'].requires_grad}")
    print(f"  t 범위: [{batch['pde']['t'].min():.3f}, {batch['pde']['t'].max():.3f}]")

    print("\n=== BC 포인트 ===")
    print(f"  x: {batch['bc']['x'].shape}")
    print(f"  c_target 범위: [{batch['bc']['c_target'].min():.3f}, {batch['bc']['c_target'].max():.3f}]")

    print("\n=== IC 포인트 ===")
    print(f"  x: {batch['ic']['x'].shape}")
    print(f"  c_target 범위: [{batch['ic']['c_target'].min():.4f}, {batch['ic']['c_target'].max():.4f}]")

    grid = sampler.make_grid(nx=50, ny=50, t_val=0.1)
    print(f"\n=== 시각화 격자 ===")
    print(f"  shape: {grid['shape']}, x: {grid['x'].shape}")

    # IC 농도 분포 빠른 확인 (matplotlib)
    import matplotlib.pyplot as plt
    x_np = batch["ic"]["x"].detach().numpy().flatten()
    y_np = batch["ic"]["y"].detach().numpy().flatten()
    c_np = batch["ic"]["c_target"].detach().numpy().flatten()

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(x_np, y_np, c=c_np, cmap="hot", s=2)
    plt.colorbar(sc, ax=ax, label="c(x,y,0)")
    ax.set_title("Initial Condition — Gaussian Point Source")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("/home/claude/Fick-Diffusion-Simulator/outputs/figures/ic_check.png", dpi=120)
    print("\nIC 분포 저장 → outputs/figures/ic_check.png")

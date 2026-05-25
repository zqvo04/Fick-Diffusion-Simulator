"""
model.py — PINN 네트워크 아키텍처

입력: (x, y, t) → 출력: c(x, y, t)  [농도 예측]

설계 포인트:
- Xavier 초기화 + tanh 활성화: PDE 미분 계산 시 고차 미분이 소멸하지 않음
- 입력 정규화 레이어: 도메인 [0,L] × [0,T] → [-1, 1] 스케일링
- requires_grad 관리: autograd로 ∂c/∂x, ∂²c/∂x² 계산 필요
"""

import torch
import torch.nn as nn
from typing import Tuple


class FourierEmbedding(nn.Module):
    """
    선택적 Fourier Feature Embedding.
    고주파 공간 패턴 학습 시 수렴 가속 (Tancik et al., 2020).
    use_fourier=False 시 일반 MLP와 동일.
    """

    def __init__(self, input_dim: int, embed_dim: int = 64, sigma: float = 1.0):
        super().__init__()
        # 학습되지 않는 랜덤 주파수 행렬 (고정)
        B = torch.randn(input_dim, embed_dim // 2) * sigma
        self.register_buffer("B", B)  # optimizer 업데이트 제외

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, input_dim)
        proj = x @ self.B  # (N, embed_dim // 2)
        return torch.cat([torch.sin(2 * torch.pi * proj),
                          torch.cos(2 * torch.pi * proj)], dim=-1)  # (N, embed_dim)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D Fick's Diffusion.

    입력 텐서: [x, y, t]  shape (N, 3)
    출력 텐서: c           shape (N, 1)

    Args:
        hidden_layers : 은닉층 뉴런 수 리스트, e.g. [64, 64, 64, 64]
        x_bounds      : (x_min, x_max) 도메인 경계
        y_bounds      : (y_min, y_max)
        t_bounds      : (t_min, t_max)
        use_fourier   : Fourier embedding 사용 여부
        fourier_dim   : Fourier embedding 차원 (use_fourier=True 시)
    """

    def __init__(
        self,
        hidden_layers: list = [64, 64, 64, 64],
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
        t_bounds: Tuple[float, float] = (0.0, 1.0),
        use_fourier: bool = False,
        fourier_dim: int = 64,
    ):
        super().__init__()

        # 정규화 파라미터 등록 (학습 파라미터 아님)
        self.register_buffer("x_min", torch.tensor(x_bounds[0], dtype=torch.float32))
        self.register_buffer("x_max", torch.tensor(x_bounds[1], dtype=torch.float32))
        self.register_buffer("y_min", torch.tensor(y_bounds[0], dtype=torch.float32))
        self.register_buffer("y_max", torch.tensor(y_bounds[1], dtype=torch.float32))
        self.register_buffer("t_min", torch.tensor(t_bounds[0], dtype=torch.float32))
        self.register_buffer("t_max", torch.tensor(t_bounds[1], dtype=torch.float32))

        self.use_fourier = use_fourier

        # 입력 차원 결정
        if use_fourier:
            self.fourier = FourierEmbedding(input_dim=3, embed_dim=fourier_dim)
            first_dim = fourier_dim
        else:
            first_dim = 3  # (x, y, t)

        # MLP 레이어 구성
        layers = []
        in_dim = first_dim
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))  # 출력: 농도 c

        self.net = nn.Sequential(*layers)

        # Xavier 초기화 적용
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _normalize(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        """
        도메인 좌표 → [-1, 1] 정규화
        정규화 수식: x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        """
        x_n = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_n = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_n = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return x_n, y_n, t_n

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y, t : shape (N, 1) 또는 (N,), requires_grad=True 필수 (PDE loss 계산 시)
        Returns:
            c       : shape (N, 1)
        """
        x_n, y_n, t_n = self._normalize(x, y, t)

        # 차원 통일: (N,) → (N, 1)
        if x_n.dim() == 1:
            x_n = x_n.unsqueeze(-1)
            y_n = y_n.unsqueeze(-1)
            t_n = t_n.unsqueeze(-1)

        inp = torch.cat([x_n, y_n, t_n], dim=-1)  # (N, 3)

        if self.use_fourier:
            inp = self.fourier(inp)  # (N, fourier_dim)

        return self.net(inp)  # (N, 1)

    def predict(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """추론 전용 (no_grad). 시각화/평가에 사용."""
        with torch.no_grad():
            return self.forward(x, y, t)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 빠른 동작 확인 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = PINN(hidden_layers=[64, 64, 64, 64])
    print(f"파라미터 수: {model.count_parameters():,}")

    # 더미 입력 (requires_grad=True → PDE loss 계산용)
    N = 100
    x = torch.rand(N, 1, requires_grad=True)
    y = torch.rand(N, 1, requires_grad=True)
    t = torch.rand(N, 1, requires_grad=True)

    c = model(x, y, t)
    print(f"출력 shape: {c.shape}")   # (100, 1)
    print(f"출력 범위: [{c.min().item():.4f}, {c.max().item():.4f}]")

    # Fourier embedding 버전
    model_f = PINN(hidden_layers=[64, 64, 64, 64], use_fourier=True, fourier_dim=64)
    print(f"\n[Fourier] 파라미터 수: {model_f.count_parameters():,}")
    c_f = model_f(x, y, t)
    print(f"[Fourier] 출력 shape: {c_f.shape}")

"""
loss.py — PINN 복합 손실 함수 (핵심 구현)

★ 이 파일이 포트폴리오의 핵심 차별점 ★
  autograd로 ∂c/∂t, ∂²c/∂x², ∂²c/∂y² 을 직접 계산.
  DeepXDE, NeuralPDE 같은 프레임워크 미사용.

물리 방정식:
  ∂c/∂t = D(∂²c/∂x² + ∂²c/∂y²)     ← Fick's 2nd Law (2D)

잔차(residual) 정의:
  r(x,y,t) = ∂c/∂t - D(∂²c/∂x² + ∂²c/∂y²)
  완벽한 해 → r = 0

복합 손실:
  L_total = L_pde + λ_bc · L_bc + λ_ic · L_ic
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


# ────────────────────────────────────────────────────────────────────────────
# 핵심 유틸: autograd 1차/2차 편미분
# ────────────────────────────────────────────────────────────────────────────

def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    outputs를 inputs에 대해 미분.
    
    핵심 파라미터 설명:
      grad_outputs=torch.ones_like(outputs)
        → outputs가 스칼라가 아닌 벡터(N,1)이므로 필요.
          수학적으로는 ∂(Σ outputs_i) / ∂inputs 와 동일.
          각 포인트 i에서의 ∂c_i/∂x_i 를 독립적으로 계산.
      
      create_graph=True
        → 이 미분 연산 자체를 computational graph에 포함.
          2차 미분(∂²c/∂x²) 계산 시 필수.
          False면 1차 미분 후 graph가 끊겨서 다시 미분 불가.
      
      retain_graph=True  
        → 같은 forward graph를 여러 번 사용 (∂c/∂t, ∂c/∂x, ∂c/∂y 모두 필요).
          False면 첫 미분 후 graph 해제 → 두 번째 grad() 호출 시 에러.
    
    Returns:
        outputs와 동일한 shape의 미분값 텐서
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),  # 벡터-Jacobian product
        create_graph=True,   # 2차 미분 가능하도록 graph 유지
        retain_graph=True,   # 여러 입력 변수에 대해 반복 미분 허용
    )[0]                     # grad()는 tuple 반환 → [0]으로 텐서 추출


# ────────────────────────────────────────────────────────────────────────────
# PDE Residual 계산
# ────────────────────────────────────────────────────────────────────────────

def compute_pde_residual(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    D: float,
) -> torch.Tensor:
    """
    Fick's 2nd Law 잔차 계산:
        r = ∂c/∂t - D(∂²c/∂x² + ∂²c/∂y²)

    Args:
        model : PINN 네트워크 (입력: x,y,t → 출력: c)
        x,y,t : 콜로케이션 포인트, shape (N,1), requires_grad=True 필수
        D     : 확산계수 (float)

    Returns:
        residual : shape (N,1), 값이 0에 가까울수록 좋은 해

    미분 계산 흐름:
        c = model(x,y,t)          # forward: 농도 예측
        ∂c/∂t   = grad(c, t)      # 시간 미분 (1차)
        ∂c/∂x   = grad(c, x)      # 공간 미분 (1차, create_graph=True 필수)
        ∂²c/∂x² = grad(∂c/∂x, x) # 공간 미분 (2차)
        ∂c/∂y   = grad(c, y)      # (1차)
        ∂²c/∂y² = grad(∂c/∂y, y) # (2차)
    """

    # ── Step 1: Forward Pass ──────────────────────────────────────────────
    c = model(x, y, t)   # shape: (N, 1)
    
    # ── Step 2: 시간 미분 ∂c/∂t ──────────────────────────────────────────
    # create_graph=True: 이 결과를 다시 미분할 필요 없음
    # 하지만 grad() 함수 내부에서 create_graph=True로 통일해도 무방
    c_t = grad(c, t)     # shape: (N, 1)

    # ── Step 3: x 방향 2차 공간 미분 ∂²c/∂x² ────────────────────────────
    # 1차 먼저: create_graph=True 필수 (이걸 다시 미분할 것이므로)
    c_x  = grad(c, x)    # ∂c/∂x,   shape: (N, 1)
    c_xx = grad(c_x, x)  # ∂²c/∂x², shape: (N, 1)
    # ※ c_x를 계산할 때 create_graph=True가 있어야 c_x에 대한 grad()가 가능

    # ── Step 4: y 방향 2차 공간 미분 ∂²c/∂y² ────────────────────────────
    c_y  = grad(c, y)    # ∂c/∂y,   shape: (N, 1)
    c_yy = grad(c_y, y)  # ∂²c/∂y², shape: (N, 1)

    # ── Step 5: Residual 조립 ─────────────────────────────────────────────
    # Fick's 2nd Law: ∂c/∂t = D(∂²c/∂x² + ∂²c/∂y²)
    # 잔차: r = ∂c/∂t - D(∂²c/∂x² + ∂²c/∂y²) → 이 값이 0이 되도록 학습
    residual = c_t - D * (c_xx + c_yy)  # shape: (N, 1)

    return residual


# ────────────────────────────────────────────────────────────────────────────
# 복합 손실 함수 클래스
# ────────────────────────────────────────────────────────────────────────────

class FickLoss(nn.Module):
    """
    L_total = L_pde + λ_bc · L_bc + λ_ic · L_ic

    각 항의 역할:
      L_pde : 내부 도메인에서 PDE 잔차의 MSE → 물리 법칙 강제
      L_bc  : 경계에서 예측값과 BC target의 MSE → 경계조건 강제
      L_ic  : t=0에서 예측값과 IC target의 MSE → 초기조건 강제

    Args:
        D      : 확산계수
        lambda_bc : BC 손실 가중치 (기본 10.0)
        lambda_ic : IC 손실 가중치 (기본 10.0)
    
    λ 설정 가이드:
        - λ = 1   : PDE와 동등한 가중치
        - λ = 10  : BC/IC를 PDE보다 10배 강조 (권장 시작점)
        - λ = 100 : BC/IC를 거의 hard constraint처럼 처리
        훈련 초기에 λ 크게 → BC/IC 먼저 학습 → λ 줄이는 curriculum도 가능
    """

    def __init__(self, D: float, lambda_bc: float = 10.0, lambda_ic: float = 10.0):
        super().__init__()
        self.D = D
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic

    def pde_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_pde = (1/N) Σ r(x_i, y_i, t_i)²

        MSE를 쓰는 이유: 잔차가 양수/음수 모두 가능하므로 절댓값 대신 제곱.
        """
        residual = compute_pde_residual(model, x, y, t, self.D)
        return torch.mean(residual ** 2)

    def bc_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        c_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_bc = (1/N) Σ [c_pred(x_b, y_b, t_b) - c_target]²

        경계에서 예측값이 c_target(여기서는 0)에 가까워지도록.
        BC 포인트는 requires_grad 없이도 됨 (PDE 미분 불필요).
        """
        c_pred = model(x, y, t)
        return torch.mean((c_pred - c_target) ** 2)

    def ic_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        c_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_ic = (1/N) Σ [c_pred(x_i, y_i, 0) - c_ic_target(x_i, y_i)]²

        t=0에서의 예측이 초기 Gaussian 분포를 따르도록.
        """
        c_pred = model(x, y, t)
        return torch.mean((c_pred - c_target) ** 2)

    def forward(
        self,
        model: nn.Module,
        pde_batch: Dict[str, torch.Tensor],
        bc_batch:  Dict[str, torch.Tensor],
        ic_batch:  Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        전체 복합 손실 계산.

        Args:
            model     : PINN 네트워크
            pde_batch : {'x','y','t'}           ← sampler.sample_pde()
            bc_batch  : {'x','y','t','c_target'} ← sampler.sample_bc()
            ic_batch  : {'x','y','t','c_target'} ← sampler.sample_ic()

        Returns:
            loss_total : 역전파에 사용할 총 손실 (scalar tensor)
            loss_dict  : 각 항 로그용 {'pde','bc','ic','total'} float 딕셔너리
        """
        # 각 손실 항 계산
        l_pde = self.pde_loss(model, pde_batch["x"], pde_batch["y"], pde_batch["t"])
        l_bc  = self.bc_loss(model,  bc_batch["x"],  bc_batch["y"],  bc_batch["t"],  bc_batch["c_target"])
        l_ic  = self.ic_loss(model,  ic_batch["x"],  ic_batch["y"],  ic_batch["t"],  ic_batch["c_target"])

        # 가중합 조립
        loss_total = l_pde + self.lambda_bc * l_bc + self.lambda_ic * l_ic

        # 로그용 딕셔너리 (detach: 이 값들은 역전파에 쓰지 않음)
        loss_dict = {
            "pde":   l_pde.item(),
            "bc":    l_bc.item(),
            "ic":    l_ic.item(),
            "total": loss_total.item(),
        }

        return loss_total, loss_dict


# ────────────────────────────────────────────────────────────────────────────
# 동작 검증 (수식 레벨 단위 테스트)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.model import PINN
    from src.sampler import DiffusionSampler

    torch.manual_seed(42)

    # ── 1. autograd 미분 정확도 검증 ──────────────────────────────────────
    # 검증 방법: 알려진 함수 f(x,y,t) = sin(πx)sin(πy)exp(-t) 에 대해
    # ∂f/∂t, ∂²f/∂x² 를 autograd와 해석적 결과 비교
    print("=" * 55)
    print("[ 검증 1 ] autograd 미분 정확도")
    print("  f(x,y,t) = sin(πx)·sin(πy)·exp(-t)")
    print("  ∂f/∂t    = -sin(πx)·sin(πy)·exp(-t)")
    print("  ∂²f/∂x²  = -π²·sin(πx)·sin(πy)·exp(-t)")
    print("=" * 55)

    N = 50
    x_v = torch.rand(N, 1, requires_grad=True)
    y_v = torch.rand(N, 1, requires_grad=True)
    t_v = torch.rand(N, 1, requires_grad=True)

    import math
    f = torch.sin(math.pi * x_v) * torch.sin(math.pi * y_v) * torch.exp(-t_v)

    # autograd 계산
    df_dt  = grad(f, t_v)
    df_dx  = grad(f, x_v)
    d2f_dx2 = grad(df_dx, x_v)

    # 해석적 결과
    df_dt_exact   = -torch.sin(math.pi * x_v) * torch.sin(math.pi * y_v) * torch.exp(-t_v)
    d2f_dx2_exact = -(math.pi**2) * f

    err_t  = torch.abs(df_dt   - df_dt_exact).mean().item()
    err_xx = torch.abs(d2f_dx2 - d2f_dx2_exact).mean().item()

    print(f"  ∂f/∂t  평균 오차: {err_t:.2e}  {'✅ OK' if err_t < 1e-5 else '❌ 확인 필요'}")
    print(f"  ∂²f/∂x² 평균 오차: {err_xx:.2e}  {'✅ OK' if err_xx < 1e-5 else '❌ 확인 필요'}")

    # ── 2. FickLoss 전체 파이프라인 테스트 ───────────────────────────────
    print("\n" + "=" * 55)
    print("[ 검증 2 ] FickLoss 전체 파이프라인")
    print("=" * 55)

    model = PINN(hidden_layers=[32, 32, 32])
    sampler = DiffusionSampler(n_pde=500, n_bc=200, n_ic=200)
    criterion = FickLoss(D=1e-3, lambda_bc=10.0, lambda_ic=10.0)

    batch = sampler.sample_all(ic_kwargs={"ic_type": "gaussian", "sigma": 0.05})

    loss, log = criterion(model, batch["pde"], batch["bc"], batch["ic"])

    print(f"  L_pde  : {log['pde']:.6f}")
    print(f"  L_bc   : {log['bc']:.6f}")
    print(f"  L_ic   : {log['ic']:.6f}")
    print(f"  L_total: {log['total']:.6f}")
    print(f"  역전파 가능 여부: ", end="")
    loss.backward()
    print("✅ OK")

    # 파라미터 gradient 존재 확인
    has_grad = all(p.grad is not None for p in model.parameters())
    print(f"  파라미터 gradient 존재: {'✅ OK' if has_grad else '❌ 없음'}")

    # ── 3. λ 가중치 효과 확인 ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[ 검증 3 ] λ 변화에 따른 L_total 비교")
    print("=" * 55)

    model2 = PINN(hidden_layers=[32, 32, 32])
    for lam in [1.0, 10.0, 100.0]:
        crit = FickLoss(D=1e-3, lambda_bc=lam, lambda_ic=lam)
        _, log2 = crit(model2, batch["pde"], batch["bc"], batch["ic"])
        print(f"  λ={lam:5.0f} → L_total={log2['total']:.4f}  "
              f"(PDE:{log2['pde']:.4f}  "
              f"BC:{lam*log2['bc']:.4f}  "
              f"IC:{lam*log2['ic']:.4f})")

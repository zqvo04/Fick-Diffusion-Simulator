"""
test_loss.py — 손실 함수 단위 테스트

실행:
    pytest tests/test_loss.py -v

테스트 철학:
  1. autograd 수치 정확도 — 알려진 함수로 미분값 검증
  2. 물리적 특성 검증    — 완벽한 해를 넣었을 때 L_pde ≈ 0
  3. 손실 함수 거동      — 각 항이 독립적으로 올바르게 작동
  4. gradient 흐름 검증  — backward() 후 파라미터 gradient 존재 확인
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch
import numpy as np

from src.model   import PINN
from src.sampler import DiffusionSampler
from src.loss    import FickLoss, compute_pde_residual, grad


# ────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    torch.manual_seed(0)
    return PINN(hidden_layers=[32, 32])

@pytest.fixture
def sampler():
    return DiffusionSampler(n_pde=200, n_bc=100, n_ic=100, strategy="uniform")

@pytest.fixture
def criterion():
    return FickLoss(D=1e-2, lambda_bc=10.0, lambda_ic=10.0)


# ────────────────────────────────────────────────────────────────────────────
# 1. autograd 미분 정확도
# ────────────────────────────────────────────────────────────────────────────

class TestAutograd:
    """
    알려진 함수에 대해 autograd와 해석 미분을 비교.
    허용 오차: float32 정밀도 수준 (1e-5)
    """

    def test_first_order_derivative(self):
        """
        f(x) = sin(πx) → ∂f/∂x = πcos(πx)
        1차 미분 정확도 검증.
        """
        x = torch.rand(50, 1, requires_grad=True)
        f = torch.sin(math.pi * x)

        df_dx       = grad(f, x)
        df_dx_exact = math.pi * torch.cos(math.pi * x)

        err = torch.abs(df_dx - df_dx_exact).mean().item()
        assert err < 1e-5, f"1차 미분 오차 {err:.2e} > 1e-5"

    def test_second_order_derivative(self):
        """
        f(x) = sin(πx) → ∂²f/∂x² = -π²sin(πx)
        2차 미분 정확도 검증. create_graph=True 동작 확인.
        """
        x = torch.rand(50, 1, requires_grad=True)
        f = torch.sin(math.pi * x)

        df_dx    = grad(f, x)
        d2f_dx2  = grad(df_dx, x)
        exact    = -(math.pi ** 2) * torch.sin(math.pi * x)

        err = torch.abs(d2f_dx2 - exact).mean().item()
        assert err < 1e-4, f"2차 미분 오차 {err:.2e} > 1e-4"

    def test_partial_derivative_independence(self):
        """
        f(x,y) = x² + y³ 에서
        ∂f/∂x = 2x  (y 무관)
        ∂f/∂y = 3y² (x 무관)
        편미분이 서로 독립적으로 계산되는지 확인.
        """
        x = torch.rand(30, 1, requires_grad=True)
        y = torch.rand(30, 1, requires_grad=True)
        f = x ** 2 + y ** 3

        df_dx = grad(f, x)
        df_dy = grad(f, y)

        assert torch.allclose(df_dx, 2 * x,      atol=1e-5), "∂f/∂x 오류"
        assert torch.allclose(df_dy, 3 * y ** 2, atol=1e-5), "∂f/∂y 오류"

    def test_laplacian_2d(self):
        """
        2D Laplacian 검증:
        f(x,y) = sin(πx)·sin(πy)
        ∇²f = ∂²f/∂x² + ∂²f/∂y² = -2π²·sin(πx)·sin(πy)
        """
        x = torch.rand(50, 1, requires_grad=True)
        y = torch.rand(50, 1, requires_grad=True)
        f = torch.sin(math.pi * x) * torch.sin(math.pi * y)

        f_xx = grad(grad(f, x), x)
        f_yy = grad(grad(f, y), y)
        laplacian = f_xx + f_yy

        exact = -2 * (math.pi ** 2) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        err   = torch.abs(laplacian - exact).mean().item()
        assert err < 1e-4, f"2D Laplacian 오차 {err:.2e} > 1e-4"


# ────────────────────────────────────────────────────────────────────────────
# 2. 물리적 특성 검증
# ────────────────────────────────────────────────────────────────────────────

class TestPhysics:
    """
    PDE의 정확한 해(exact solution)를 넣었을 때
    PDE residual이 0에 가까운지 확인.
    """

    def test_pde_residual_with_exact_solution(self):
        """
        D=1, 도메인 [0,1]²에서의 정확한 해:
          c(x,y,t) = sin(πx)·sin(πy)·exp(-2π²Dt)
        이 해는 ∂c/∂t = D·∇²c 를 정확히 만족함.

        이 함수를 흉내내는 람다를 만들어 PDE residual 계산.
        residual이 거의 0 이어야 함.
        """
        D = 1.0
        N = 50

        x = torch.rand(N, 1, requires_grad=True)
        y = torch.rand(N, 1, requires_grad=True)
        t = torch.rand(N, 1, requires_grad=True)

        # 정확한 해를 계산하는 임시 모델
        class ExactModel(torch.nn.Module):
            def forward(self, x, y, t):
                return (torch.sin(math.pi * x)
                        * torch.sin(math.pi * y)
                        * torch.exp(-2 * math.pi**2 * D * t))

        exact_model = ExactModel()
        residual = compute_pde_residual(exact_model, x, y, t, D=D)
        mean_res = residual.abs().mean().item()

        # 정확한 해이므로 residual ≈ 0 (float32 정밀도 허용)
        assert mean_res < 1e-4, f"정확한 해의 PDE residual {mean_res:.2e} 이 크다"

    def test_pde_residual_with_wrong_solution(self):
        """
        PDE를 만족하지 않는 함수(c=1 상수)는 residual이 커야 함.
        c=1이면 ∂c/∂t=0, ∇²c=0 → residual=0 이 되어버리는 특수 케이스.
        c = x*y*t (선형) 로 테스트.
          ∂c/∂t = x*y ≠ 0
          ∇²c = 0
          → residual = x*y ≠ 0
        """
        D = 1.0
        N = 50

        x = torch.rand(N, 1, requires_grad=True)
        y = torch.rand(N, 1, requires_grad=True)
        t = torch.rand(N, 1, requires_grad=True)

        class WrongModel(torch.nn.Module):
            def forward(self, x, y, t):
                # c = (x² + y²)·t
                # ∂c/∂t = x² + y²
                # ∂²c/∂x² = 2,  ∂²c/∂y² = 2
                # residual = (x²+y²) - D·4  → 일반적으로 ≠ 0
                return (x ** 2 + y ** 2) * t

        wrong_model = WrongModel()
        residual = compute_pde_residual(wrong_model, x, y, t, D=D)
        mean_res = residual.abs().mean().item()

        assert mean_res > 1e-3, f"잘못된 해의 residual {mean_res:.2e} 이 너무 작다"


# ────────────────────────────────────────────────────────────────────────────
# 3. 손실 함수 거동 검증
# ────────────────────────────────────────────────────────────────────────────

class TestFickLoss:

    def test_loss_is_nonnegative(self, model, sampler, criterion):
        """모든 손실 항은 MSE이므로 반드시 ≥ 0."""
        batch = sampler.sample_all()
        _, log = criterion(model, batch["pde"], batch["bc"], batch["ic"])

        assert log["pde"]   >= 0, "L_pde < 0"
        assert log["bc"]    >= 0, "L_bc < 0"
        assert log["ic"]    >= 0, "L_ic < 0"
        assert log["total"] >= 0, "L_total < 0"

    def test_total_equals_weighted_sum(self, model, sampler):
        """
        L_total = L_pde + λ_bc · L_bc + λ_ic · L_ic 수식 검증.
        직접 계산한 값과 forward() 결과가 일치해야 함.
        """
        lam = 5.0
        criterion = FickLoss(D=1e-2, lambda_bc=lam, lambda_ic=lam)
        batch = sampler.sample_all()

        _, log = criterion(model, batch["pde"], batch["bc"], batch["ic"])

        expected = log["pde"] + lam * log["bc"] + lam * log["ic"]
        assert abs(log["total"] - expected) < 1e-6, \
            f"L_total 불일치: {log['total']:.6f} ≠ {expected:.6f}"

    def test_lambda_scaling(self, model, sampler):
        """
        λ 증가 → L_total 증가 (단조 증가 확인).
        L_pde는 λ와 무관하게 동일해야 함.
        """
        batch = sampler.sample_all()
        pde_losses = []
        totals     = []

        for lam in [1.0, 10.0, 100.0]:
            crit = FickLoss(D=1e-2, lambda_bc=lam, lambda_ic=lam)
            _, log = crit(model, batch["pde"], batch["bc"], batch["ic"])
            pde_losses.append(log["pde"])
            totals.append(log["total"])

        # L_pde는 λ와 무관
        assert abs(pde_losses[0] - pde_losses[2]) < 1e-6, \
            "λ 변화에 따라 L_pde가 변함"

        # L_total 단조 증가
        assert totals[0] < totals[1] < totals[2], \
            f"L_total이 λ 증가에 단조 증가하지 않음: {totals}"

    def test_bc_loss_zero_for_zero_prediction(self, sampler):
        """
        경계조건 target이 0이고, 모델이 항상 0을 예측하면 L_bc = 0.
        """
        class ZeroModel(torch.nn.Module):
            def forward(self, x, y, t):
                return torch.zeros(x.shape[0], 1)

        criterion = FickLoss(D=1e-2, lambda_bc=10.0, lambda_ic=10.0)
        bc = sampler.sample_bc()  # c_target = 0

        loss = criterion.bc_loss(ZeroModel(), bc["x"], bc["y"], bc["t"], bc["c_target"])
        assert loss.item() < 1e-10, f"Zero 예측 BC loss = {loss.item():.2e} (0 이어야 함)"

    def test_backward_computes_gradients(self, model, sampler, criterion):
        """
        backward() 호출 후 모든 파라미터에 gradient가 존재해야 함.
        이게 없으면 학습 자체가 안 됨.
        """
        batch = sampler.sample_all()
        loss, _ = criterion(model, batch["pde"], batch["bc"], batch["ic"])
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"파라미터 '{name}'에 gradient 없음"
            assert not torch.isnan(param.grad).any(), f"파라미터 '{name}' gradient에 NaN"

    def test_no_nan_in_loss(self, model, sampler, criterion):
        """손실에 NaN/Inf 없는지 확인. 훈련 불안정성 조기 감지."""
        batch = sampler.sample_all()
        loss, log = criterion(model, batch["pde"], batch["bc"], batch["ic"])

        assert not torch.isnan(loss), "L_total에 NaN 발생"
        assert not torch.isinf(loss), "L_total에 Inf 발생"
        for key, val in log.items():
            assert not math.isnan(val), f"log['{key}']에 NaN"


# ────────────────────────────────────────────────────────────────────────────
# 4. 모델 아키텍처 검증
# ────────────────────────────────────────────────────────────────────────────

class TestModel:

    def test_output_shape(self):
        """출력 shape이 (N,1)인지 확인."""
        model = PINN(hidden_layers=[32, 32])
        N = 77
        x = torch.rand(N, 1, requires_grad=True)
        y = torch.rand(N, 1, requires_grad=True)
        t = torch.rand(N, 1, requires_grad=True)

        c = model(x, y, t)
        assert c.shape == (N, 1), f"출력 shape {c.shape} ≠ ({N}, 1)"

    def test_normalization_range(self):
        """
        입력 정규화 후 MLP 입력이 [-1,1] 범위 내인지 확인.
        domain 경계값을 입력하면 정규화 후 -1 또는 1이 되어야 함.
        """
        model = PINN(
            hidden_layers=[16],
            x_bounds=(0.0, 2.0),
            y_bounds=(0.0, 1.0),
            t_bounds=(0.0, 0.5),
        )
        # 경계값 입력
        x = torch.tensor([[0.0], [2.0]])
        y = torch.tensor([[0.0], [1.0]])
        t = torch.tensor([[0.0], [0.5]])

        # _normalize 결과 확인
        xn, yn, tn = model._normalize(x, y, t)
        assert torch.allclose(xn, torch.tensor([[-1.0], [1.0]])), "x 정규화 오류"
        assert torch.allclose(yn, torch.tensor([[-1.0], [1.0]])), "y 정규화 오류"
        assert torch.allclose(tn, torch.tensor([[-1.0], [1.0]])), "t 정규화 오류"

    def test_fourier_embedding_shape(self):
        """Fourier embedding 출력 shape 확인."""
        model = PINN(hidden_layers=[32], use_fourier=True, fourier_dim=64)
        x = torch.rand(20, 1, requires_grad=True)
        y = torch.rand(20, 1, requires_grad=True)
        t = torch.rand(20, 1, requires_grad=True)
        c = model(x, y, t)
        assert c.shape == (20, 1)


# ────────────────────────────────────────────────────────────────────────────
# 5. 샘플러 검증
# ────────────────────────────────────────────────────────────────────────────

class TestSampler:

    def test_pde_points_in_domain(self):
        """PDE 포인트가 도메인 [0,1]² × [0,T] 내에 있는지 확인."""
        sampler = DiffusionSampler(
            x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), T=0.5,
            n_pde=500, strategy="lhs"
        )
        pde = sampler.sample_pde()

        assert pde["x"].min() >= 0.0 and pde["x"].max() <= 1.0, "x 범위 초과"
        assert pde["y"].min() >= 0.0 and pde["y"].max() <= 1.0, "y 범위 초과"
        assert pde["t"].min() >= 0.0 and pde["t"].max() <= 0.5, "t 범위 초과"

    def test_bc_target_is_zero(self):
        """Dirichlet BC: c_target이 모두 0인지 확인."""
        sampler = DiffusionSampler(n_pde=100, n_bc=200, n_ic=100)
        bc = sampler.sample_bc()
        assert bc["c_target"].abs().max().item() == 0.0, "BC c_target이 0이 아님"

    def test_ic_gaussian_peak_at_source(self):
        """Gaussian IC: 소스 위치(0.5, 0.5)에서 농도가 최대인지 확인."""
        sampler = DiffusionSampler(n_pde=100, n_bc=100, n_ic=2000)
        ic = sampler.sample_ic(ic_type="gaussian", source_x=0.5, source_y=0.5,
                               sigma=0.05, c_max=1.0)

        # 소스 주변 포인트 추출
        x_np = ic["x"].detach().numpy().flatten()
        y_np = ic["y"].detach().numpy().flatten()
        c_np = ic["c_target"].detach().numpy().flatten()

        dist = (x_np - 0.5)**2 + (y_np - 0.5)**2
        c_at_source = c_np[dist < 0.01].max() if (dist < 0.01).any() else 0.0
        c_at_boundary = c_np[dist > 0.2].max()

        assert c_at_source > c_at_boundary, \
            "Gaussian IC: 소스 위치가 경계보다 농도가 낮음"

    def test_requires_grad_pde_true(self):
        """PDE 포인트는 requires_grad=True, IC target은 False."""
        sampler = DiffusionSampler(n_pde=100, n_bc=100, n_ic=100)
        batch = sampler.sample_all()

        assert batch["pde"]["x"].requires_grad, "PDE x requires_grad=False"
        assert not batch["ic"]["c_target"].requires_grad, "IC c_target requires_grad=True"

"""
analytical.py — 2D Fick 방정식 해석해

두 가지 해석해 구현:
  1. infinite_domain_gaussian : 무한 도메인, 단순 Gaussian 공식
     → c(x,y,t) = C₀/(4πDt) · exp(-r²/4Dt)
     → 빠르고 단순. 단, 경계조건 무시.

  2. finite_domain_series     : 유한 직사각형 도메인 [0,L]², Dirichlet BC (c=0)
     → Eigenfunction expansion (Fourier 급수)
     → 경계조건 정확 반영. N_terms²개 항 합산.

모델 오차 정의:
  L2 relative error = ||c_pred - c_exact||₂ / ||c_exact||₂
"""

import numpy as np
import torch
from typing import Tuple


# ────────────────────────────────────────────────────────────────────────────
# 해석해 1: 무한 도메인 Gaussian
# ────────────────────────────────────────────────────────────────────────────

def infinite_domain_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    D: float,
    x0: float = 0.5,
    y0: float = 0.5,
    C0: float = 1.0,
) -> np.ndarray:
    """
    무한 도메인에서 점 소스(point source) 확산의 해석해.

    수식 유도:
      초기조건: c(x,y,0) = C₀ · δ(x-x₀) · δ(y-y₀)
      해       : c(x,y,t) = C₀ / (4πDt) · exp(-[(x-x₀)²+(y-y₀)²] / 4Dt)

    물리적 의미:
      - 분모 4πDt: 시간이 흐를수록 퍼지므로 최대 농도 감소
      - exp 항: 소스 위치(x₀,y₀)에서 멀수록 농도 낮음

    ※ t=0에서 분모=0이므로 t > 0 에서만 호출할 것.

    Args:
        x, y : numpy array (any shape), 공간 좌표
        t    : float, 시각 (> 0)
        D    : float, 확산계수
        x0,y0: 초기 소스 위치
        C0   : 초기 총 질량 (농도 스케일)
    """
    if t <= 0:
        raise ValueError("t must be > 0 for infinite domain solution")

    r_sq = (x - x0) ** 2 + (y - y0) ** 2
    return (C0 / (4 * np.pi * D * t)) * np.exp(-r_sq / (4 * D * t))


# ────────────────────────────────────────────────────────────────────────────
# 해석해 2: 유한 도메인 Eigenfunction Expansion
# ────────────────────────────────────────────────────────────────────────────

def _compute_fourier_coefficients(
    x0: float,
    y0: float,
    sigma: float,
    L: float,
    N_terms: int,
    C0: float,
) -> np.ndarray:
    """
    Gaussian 초기조건에 대한 2D Fourier sine 계수 A_mn 계산.

    초기조건: c₀(x,y) = C₀ · exp(-[(x-x₀)²+(y-y₀)²] / 2σ²)

    Fourier 계수:
      A_mn = (4/L²) ∫₀ᴸ ∫₀ᴸ c₀(x,y) · sin(mπx/L) · sin(nπy/L) dx dy

    수치 적분(Simpson)으로 계산.

    Returns:
        A : shape (N_terms, N_terms), A[m-1, n-1] = A_mn
    """
    # 적분용 격자 (해상도 높을수록 정확, 메모리와 tradeoff)
    n_quad = 200
    xs = np.linspace(0, L, n_quad)
    ys = np.linspace(0, L, n_quad)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    XX, YY = np.meshgrid(xs, ys)  # (n_quad, n_quad)

    # 초기 농도장 계산
    c0_vals = C0 * np.exp(-((XX - x0) ** 2 + (YY - y0) ** 2) / (2 * sigma ** 2))

    A = np.zeros((N_terms, N_terms))
    for m in range(1, N_terms + 1):
        for n in range(1, N_terms + 1):
            # sin basis 함수
            phi_m = np.sin(m * np.pi * XX / L)  # (n_quad, n_quad)
            phi_n = np.sin(n * np.pi * YY / L)

            # 수치 적분: 사다리꼴 공식
            integrand = c0_vals * phi_m * phi_n
            integral = np.trapezoid(np.trapezoid(integrand, xs, axis=1), ys)

            A[m - 1, n - 1] = (4.0 / L ** 2) * integral

    return A


def finite_domain_series(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    D: float,
    L: float = 1.0,
    x0: float = 0.5,
    y0: float = 0.5,
    sigma: float = 0.05,
    C0: float = 1.0,
    N_terms: int = 20,
) -> np.ndarray:
    """
    유한 도메인 [0,L]² 에서 Dirichlet BC(c=0)를 만족하는 해석해.

    수식:
      c(x,y,t) = Σ_{m=1}^{N} Σ_{n=1}^{N} A_mn
                 · sin(mπx/L) · sin(nπy/L)
                 · exp(-D·π²·(m²+n²)·t / L²)

    물리적 의미:
      - 각 (m,n) 모드는 고유 감쇠율 D·π²·(m²+n²)/L²을 가짐
      - 고주파 모드(m,n 큰 것)는 빠르게 감쇠 → 시간 지나면 기저 모드만 남음
      - N_terms 클수록 정확하나 계산 시간 증가. N=20 이면 충분.

    Args:
        x, y   : numpy array (flattened), 공간 좌표
        t      : float, 시각 (≥ 0)
        D      : 확산계수
        L      : 도메인 크기
        N_terms: Fourier 급수 항 수 (m, n 각각)

    Returns:
        c : x, y와 동일한 shape의 농도 배열
    """
    A = _compute_fourier_coefficients(x0, y0, sigma, L, N_terms, C0)

    c = np.zeros_like(x, dtype=np.float64)
    for m in range(1, N_terms + 1):
        for n in range(1, N_terms + 1):
            # 시간 감쇠 지수항
            decay = np.exp(-D * np.pi ** 2 * (m ** 2 + n ** 2) * t / L ** 2)

            # 공간 basis
            phi_x = np.sin(m * np.pi * x / L)
            phi_y = np.sin(n * np.pi * y / L)

            c += A[m - 1, n - 1] * phi_x * phi_y * decay

    return c


# ────────────────────────────────────────────────────────────────────────────
# 오차 지표
# ────────────────────────────────────────────────────────────────────────────

def l2_relative_error(c_pred: np.ndarray, c_exact: np.ndarray) -> float:
    """
    L2 상대 오차:
      ε = ||c_pred - c_exact||₂ / ||c_exact||₂

    값 해석:
      0.01 → 1% 오차 (우수)
      0.05 → 5% 오차 (양호)
      0.10 → 10% 오차 (개선 필요)
    """
    numerator   = np.linalg.norm(c_pred.flatten() - c_exact.flatten())
    denominator = np.linalg.norm(c_exact.flatten()) + 1e-10  # 0 나눗셈 방지
    return float(numerator / denominator)


def max_absolute_error(c_pred: np.ndarray, c_exact: np.ndarray) -> float:
    """최대 절대 오차: max|c_pred - c_exact|"""
    return float(np.max(np.abs(c_pred.flatten() - c_exact.flatten())))


# ────────────────────────────────────────────────────────────────────────────
# 동작 검증
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = 1e-2
    L = 1.0
    x0, y0 = 0.5, 0.5
    sigma = 0.05

    nx, ny = 80, 80
    xs = np.linspace(0, L, nx)
    ys = np.linspace(0, L, ny)
    XX, YY = np.meshgrid(xs, ys)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    t_vals = [0.01, 0.05, 0.1]

    for col, t in enumerate(t_vals):
        # 유한 도메인 해석해 계산
        c_finite = finite_domain_series(
            XX.flatten(), YY.flatten(), t,
            D=D, L=L, x0=x0, y0=y0, sigma=sigma, N_terms=15
        ).reshape(ny, nx)

        # 무한 도메인 해석해 (비교용)
        c_inf = infinite_domain_gaussian(
            XX, YY, t, D=D, x0=x0, y0=y0, C0=1.0
        )

        # 상단: 유한 도메인 해석해
        im0 = axes[0, col].contourf(XX, YY, c_finite, levels=30, cmap="hot")
        axes[0, col].set_title(f"Finite Domain  t={t}")
        axes[0, col].set_xlabel("x"); axes[0, col].set_ylabel("y")
        plt.colorbar(im0, ax=axes[0, col])

        # 하단: 무한 도메인 해석해
        im1 = axes[1, col].contourf(XX, YY, c_inf, levels=30, cmap="hot")
        axes[1, col].set_title(f"Infinite Domain  t={t}")
        axes[1, col].set_xlabel("x"); axes[1, col].set_ylabel("y")
        plt.colorbar(im1, ax=axes[1, col])

    plt.suptitle("Analytical Solutions — Fick's 2nd Law (2D)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(
        "/home/claude/Fick-Diffusion-Simulator/outputs/figures/analytical_check.png",
        dpi=120, bbox_inches="tight"
    )
    print("해석해 비교 저장 → outputs/figures/analytical_check.png")

    # 수치 검증: t→0 에서 총 질량 보존 (유한 도메인)
    print("\n[ 질량 보존 확인 ]")
    c0_exact = finite_domain_series(
        XX.flatten(), YY.flatten(), 0.001,
        D=D, L=L, x0=x0, y0=y0, sigma=sigma, N_terms=15
    ).reshape(ny, nx)
    mass = np.trapezoid(np.trapezoid(np.maximum(c0_exact, 0), xs, axis=1), ys)
    # Gaussian 2D 적분 이론값: 2πσ² = 2π*(0.05)² ≈ 0.0157
    theory = 2 * np.pi * sigma ** 2
    print(f"  t≈0 총 질량: {mass:.4f}  (이론값 ≈ {theory:.4f})  오차: {abs(mass-theory)/theory*100:.1f}%")

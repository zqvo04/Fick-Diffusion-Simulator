"""
evaluate.py — 학습된 PINN 정량 평가

실행 방법:
    python scripts/evaluate.py

출력:
    outputs/figures/error_comparison.png  — 4-panel 비교 플롯
    outputs/figures/l2_error_vs_time.png  — 시간별 L2 오차 곡선
    콘솔에 시각별 L2 오차 테이블 출력
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.model      import PINN
from src.sampler    import DiffusionSampler
from src.trainer    import TrainConfig, Trainer
from src.analytical import (
    finite_domain_series,
    l2_relative_error,
    max_absolute_error,
)


# ────────────────────────────────────────────────────────────────────────────
# 평가 설정
# ────────────────────────────────────────────────────────────────────────────

EVAL_CONFIG = {
    "D"      : 1e-2,
    "T"      : 0.5,
    "L"      : 1.0,
    "x0"     : 0.5,
    "y0"     : 0.5,
    "sigma"  : 0.05,
    "N_terms": 20,       # 해석해 Fourier 급수 항 수
    "nx"     : 80,       # 평가 격자 해상도
    "ny"     : 80,
    "t_eval" : [0.02, 0.05, 0.10, 0.20, 0.30, 0.50],  # 평가할 시각들
}


# ────────────────────────────────────────────────────────────────────────────
# 핵심 함수: 특정 시각에서 PINN 예측 vs 해석해
# ────────────────────────────────────────────────────────────────────────────

def evaluate_at_time(
    model: torch.nn.Module,
    t_val: float,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    단일 시각 t_val에서 모델 예측과 해석해를 비교.

    Returns:
        dict with keys:
            'c_pred'  : (ny, nx) PINN 예측 농도
            'c_exact' : (ny, nx) 해석해 농도
            'error'   : (ny, nx) 절대 오차 |pred - exact|
            'l2_rel'  : float L2 상대 오차
            'max_abs' : float 최대 절대 오차
            'xs', 'ys': 격자 좌표
    """
    nx, ny = cfg["nx"], cfg["ny"]
    L = cfg["L"]

    xs = np.linspace(0, L, nx)
    ys = np.linspace(0, L, ny)
    XX, YY = np.meshgrid(xs, ys)  # (ny, nx)

    # ── PINN 예측 ──────────────────────────────────────────────────────────
    # 격자를 flatten하여 (N,1) 텐서로 변환 후 모델에 투입
    x_t = torch.tensor(XX.flatten()[:, None], dtype=torch.float32, device=device)
    y_t = torch.tensor(YY.flatten()[:, None], dtype=torch.float32, device=device)
    t_t = torch.full_like(x_t, t_val)

    with torch.no_grad():
        c_pred_flat = model(x_t, y_t, t_t).cpu().numpy().flatten()

    c_pred = c_pred_flat.reshape(ny, nx)

    # ── 해석해 (유한 도메인 Fourier) ──────────────────────────────────────
    # 계산 시간이 있으므로 N_terms 적당히 설정
    c_exact = finite_domain_series(
        XX.flatten(),
        YY.flatten(),
        t_val,
        D=cfg["D"],
        L=L,
        x0=cfg["x0"],
        y0=cfg["y0"],
        sigma=cfg["sigma"],
        N_terms=cfg["N_terms"],
    ).reshape(ny, nx)

    error = np.abs(c_pred - c_exact)

    return {
        "c_pred"  : c_pred,
        "c_exact" : c_exact,
        "error"   : error,
        "l2_rel"  : l2_relative_error(c_pred, c_exact),
        "max_abs" : max_absolute_error(c_pred, c_exact),
        "xs"      : xs,
        "ys"      : ys,
        "XX"      : XX,
        "YY"      : YY,
    }


# ────────────────────────────────────────────────────────────────────────────
# 4-panel 비교 플롯 (단일 시각)
# ────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    result: dict,
    t_val: float,
    save_path: str,
):
    """
    한 시각에 대한 4-panel 비교:
      [PINN 예측] [해석해]
      [절대 오차] [오차 분포 히스토그램]
    """
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    XX, YY    = result["XX"], result["YY"]
    c_pred    = result["c_pred"]
    c_exact   = result["c_exact"]
    error     = result["error"]

    # color scale 공통 고정 (PINN / 해석해 비교를 위해)
    vmin = min(c_pred.min(), c_exact.min())
    vmax = max(c_pred.max(), c_exact.max())
    levels = np.linspace(vmin, vmax, 40)

    # ── 패널 1: PINN 예측 ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.contourf(XX, YY, c_pred, levels=levels, cmap="hot", extend="both")
    plt.colorbar(cf1, ax=ax1)
    ax1.set_title(f"PINN Prediction  (t={t_val})", fontsize=11)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_aspect("equal")

    # ── 패널 2: 해석해 ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = ax2.contourf(XX, YY, c_exact, levels=levels, cmap="hot", extend="both")
    plt.colorbar(cf2, ax=ax2)
    ax2.set_title(f"Analytical Solution  (t={t_val})", fontsize=11)
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    # ── 패널 3: 절대 오차 ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    # 오차는 별도 colormap (낮을수록 좋으니 파란 계열)
    cf3 = ax3.contourf(XX, YY, error, levels=30, cmap="Blues")
    plt.colorbar(cf3, ax=ax3)
    ax3.set_title(
        f"|Pred - Exact|   "
        f"L2={result['l2_rel']:.4f}  "
        f"MaxAbs={result['max_abs']:.4f}",
        fontsize=10,
    )
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    ax3.set_aspect("equal")

    # ── 패널 4: 오차 분포 히스토그램 ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(error.flatten(), bins=50, color="steelblue", edgecolor="white", lw=0.3)
    ax4.axvline(error.mean(), color="red", linestyle="--", label=f"mean={error.mean():.4f}")
    ax4.axvline(error.max(),  color="orange", linestyle=":",  label=f"max={error.max():.4f}")
    ax4.set_xlabel("|Pred - Exact|")
    ax4.set_ylabel("Count")
    ax4.set_title("Error Distribution", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        f"PINN vs Analytical Solution — Fick's 2nd Law 2D\n"
        f"D={EVAL_CONFIG['D']},  t={t_val},  L2 rel. error={result['l2_rel']:.4f}",
        fontsize=12, y=1.01
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# L2 오차 vs 시간 곡선
# ────────────────────────────────────────────────────────────────────────────

def plot_l2_vs_time(
    l2_errors: list,
    t_vals: list,
    save_path: str,
):
    """
    여러 시각에 걸친 L2 상대 오차 추이.

    이상적인 패턴:
      - 전 시간대에서 낮고 안정적 → 시공간 전체를 잘 학습
      - t 작을 때 오차 큰 경우   → IC 학습 부족 (λ_ic 증가 필요)
      - t 클 때 오차 큰 경우     → 장기 PDE 전파 학습 부족
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_vals, l2_errors, "o-", color="steelblue", lw=2, markersize=7)

    # 5% 기준선 (양호 임계값)
    ax.axhline(0.05, color="tomato", linestyle="--", alpha=0.7, label="5% threshold")

    for t, err in zip(t_vals, l2_errors):
        ax.annotate(f"{err:.3f}", (t, err), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Time  t", fontsize=12)
    ax.set_ylabel("L2 Relative Error", fontsize=12)
    ax.set_title("PINN Accuracy over Time — Fick's 2nd Law", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  저장: {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────────────────────────────────────────

def run_evaluation(model: torch.nn.Module, device: torch.device):
    cfg = EVAL_CONFIG

    print("\n" + "=" * 55)
    print("  PINN 정량 평가")
    print("=" * 55)
    print(f"  {'시각 t':>8}  {'L2 상대오차':>12}  {'최대절대오차':>12}")
    print(f"  {'-'*38}")

    l2_list  = []
    max_list = []

    for t_val in cfg["t_eval"]:
        result = evaluate_at_time(model, t_val, cfg, device)
        l2_list.append(result["l2_rel"])
        max_list.append(result["max_abs"])
        print(f"  t={t_val:.2f}   {result['l2_rel']:>12.6f}  {result['max_abs']:>12.6f}")

        # 대표 시각에서 4-panel 저장 (t=0.10 선택)
        if abs(t_val - 0.10) < 1e-9:
            plot_comparison(
                result, t_val,
                save_path="outputs/figures/error_comparison_t010.png"
            )

    print(f"  {'-'*38}")
    print(f"  평균 L2:   {np.mean(l2_list):.6f}")
    print(f"  평균 MaxAbs: {np.mean(max_list):.6f}")

    # L2 vs time 곡선
    plot_l2_vs_time(
        l2_list, cfg["t_eval"],
        save_path="outputs/figures/l2_error_vs_time.png"
    )

    return l2_list


# ────────────────────────────────────────────────────────────────────────────
# 스크립트 직접 실행 (체크포인트 로드 후 평가)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 저장된 체크포인트 로드
    ckpt_path = "outputs/checkpoints/pinn_best.pt"
    if not os.path.exists(ckpt_path):
        print("[evaluate] 체크포인트 없음. 빠른 훈련 후 평가 실행.")
        cfg = TrainConfig(
            D=1e-2, T=0.5,
            hidden_layers=[64, 64, 64, 64],
            use_fourier=True,
            adam_epochs=3000,
            lbfgs_epochs=200,
            n_pde=8000, n_bc=800, n_ic=800,
        )
        trainer = Trainer(cfg)
        trainer.train()
        model = trainer.model
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        saved_cfg = ckpt["config"]
        model = PINN(
            hidden_layers=saved_cfg["hidden_layers"],
            use_fourier=saved_cfg.get("use_fourier", False),
            fourier_dim=saved_cfg.get("fourier_dim", 64),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[evaluate] 체크포인트 로드 완료 (epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f})")

    run_evaluation(model, device)

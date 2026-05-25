"""
visualize.py — 2D 농도 분포 animation 생성

출력:
    outputs/animation.gif           — PINN 예측 농도 분포 애니메이션
    outputs/figures/snapshot_grid.png — 시각별 스냅샷 격자 이미지
    outputs/figures/pinn_vs_exact_anim_frame.png — 나란히 비교 프레임

animation 구조:
    좌: PINN 예측 c(x,y,t)
    우: 해석해 c_exact(x,y,t)
    중간에 시간 t 표시 슬라이더
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

from src.model      import PINN
from src.trainer    import TrainConfig, Trainer
from src.analytical import finite_domain_series


# ────────────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────────────

VIZ_CONFIG = {
    "D"        : 1e-2,
    "T"        : 0.5,
    "L"        : 1.0,
    "x0"       : 0.5,
    "y0"       : 0.5,
    "sigma"    : 0.05,
    "N_terms"  : 20,
    "nx"       : 60,          # animation 격자 (너무 크면 느림)
    "ny"       : 60,
    "n_frames" : 40,          # animation 프레임 수
    "fps"      : 8,           # GIF frames per second
    "snapshot_times": [0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
}


# ────────────────────────────────────────────────────────────────────────────
# 모든 시각에서 예측값 사전 계산 (animation 용)
# ────────────────────────────────────────────────────────────────────────────

def precompute_frames(model, device, cfg):
    """
    animation 프레임별로 PINN 예측 + 해석해를 미리 계산.
    렌더링 시 실시간 계산 → 느려지는 것 방지.

    Returns:
        t_frames   : (n_frames,) 시간 배열
        pred_frames: (n_frames, ny, nx) PINN 예측
        exact_frames: (n_frames, ny, nx) 해석해
        XX, YY     : 격자 (ny, nx)
    """
    nx, ny = cfg["nx"], cfg["ny"]
    L = cfg["L"]
    n_frames = cfg["n_frames"]

    xs = np.linspace(0, L, nx)
    ys = np.linspace(0, L, ny)
    XX, YY = np.meshgrid(xs, ys)

    # t=0은 피함 (해석해 초기 특이점 방지)
    t_frames = np.linspace(0.005, cfg["T"], n_frames)

    x_t = torch.tensor(XX.flatten()[:, None], dtype=torch.float32, device=device)
    y_t = torch.tensor(YY.flatten()[:, None], dtype=torch.float32, device=device)

    pred_frames  = []
    exact_frames = []

    print(f"  프레임 사전 계산 중 ({n_frames}개)...")
    for i, t_val in enumerate(t_frames):
        # PINN 예측
        t_tensor = torch.full_like(x_t, float(t_val))
        with torch.no_grad():
            c_pred = model(x_t, y_t, t_tensor).cpu().numpy().flatten()
        pred_frames.append(c_pred.reshape(ny, nx))

        # 해석해
        c_exact = finite_domain_series(
            XX.flatten(), YY.flatten(), float(t_val),
            D=cfg["D"], L=L, x0=cfg["x0"], y0=cfg["y0"],
            sigma=cfg["sigma"], N_terms=cfg["N_terms"],
        ).reshape(ny, nx)
        exact_frames.append(c_exact)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_frames} 완료")

    return t_frames, np.array(pred_frames), np.array(exact_frames), XX, YY


# ────────────────────────────────────────────────────────────────────────────
# 1. GIF animation (PINN 예측 vs 해석해 나란히)
# ────────────────────────────────────────────────────────────────────────────

def make_animation(t_frames, pred_frames, exact_frames, XX, YY, cfg, save_path):
    """
    FuncAnimation으로 PINN | 해석해 나란히 비교 animation 생성.

    color scale 전략:
      vmax = 전체 프레임 최대값으로 고정
      → 농도가 시간에 따라 감소하는 시각적 효과가 그대로 보임
      (자동 조정 시 항상 같은 색 강도로 보여 확산이 안 보임)
    """
    vmax = max(pred_frames.max(), exact_frames.max())
    vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    plt.subplots_adjust(bottom=0.18, wspace=0.35)

    # 초기 프레임으로 contourf 초기화
    cf1 = ax1.contourf(XX, YY, pred_frames[0],  levels=levels, cmap="hot", norm=norm)
    cf2 = ax2.contourf(XX, YY, exact_frames[0], levels=levels, cmap="hot", norm=norm)
    plt.colorbar(cf1, ax=ax1, label="Concentration c")
    plt.colorbar(cf2, ax=ax2, label="Concentration c")

    ax1.set_title("PINN Prediction", fontsize=11)
    ax2.set_title("Analytical Solution", fontsize=11)
    for ax in [ax1, ax2]:
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")

    time_text = fig.text(0.5, 0.04,
                         f"t = {t_frames[0]:.4f}", ha="center", fontsize=12,
                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Fick's 2nd Law — 2D Diffusion  (PINN vs Analytical)", fontsize=11)

    def update(frame_idx):
        """
        FuncAnimation이 매 프레임마다 호출하는 함수.
        contourf는 incremental update 지원 안 함 → axes 클리어 후 재그림.
        """
        ax1.cla(); ax2.cla()

        ax1.contourf(XX, YY, pred_frames[frame_idx],  levels=levels, cmap="hot", norm=norm)
        ax2.contourf(XX, YY, exact_frames[frame_idx], levels=levels, cmap="hot", norm=norm)

        ax1.set_title("PINN Prediction", fontsize=11)
        ax2.set_title("Analytical Solution", fontsize=11)
        for ax in [ax1, ax2]:
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_aspect("equal")

        time_text.set_text(f"t = {t_frames[frame_idx]:.4f}")
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(t_frames),
        interval=1000 // cfg["fps"],  # ms per frame
        blit=False,
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    writer = animation.PillowWriter(fps=cfg["fps"])
    anim.save(save_path, writer=writer, dpi=100)
    plt.close()
    print(f"  Animation 저장: {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# 2. 스냅샷 격자 (정적 이미지, README 용)
# ────────────────────────────────────────────────────────────────────────────

def make_snapshot_grid(model, device, cfg, save_path):
    """
    여러 시각에서의 PINN 예측 스냅샷을 격자로 배치.
    README.md에 넣을 대표 이미지로 활용.
    """
    t_snaps = cfg["snapshot_times"]
    nx, ny  = cfg["nx"], cfg["ny"]
    L       = cfg["L"]

    xs = np.linspace(0, L, nx)
    ys = np.linspace(0, L, ny)
    XX, YY = np.meshgrid(xs, ys)

    x_t = torch.tensor(XX.flatten()[:, None], dtype=torch.float32, device=device)
    y_t = torch.tensor(YY.flatten()[:, None], dtype=torch.float32, device=device)

    n = len(t_snaps)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    # 전체 컬러바 범위 미리 계산
    all_pred = []
    all_exact = []
    for t_val in t_snaps:
        t_t = torch.full_like(x_t, float(t_val))
        with torch.no_grad():
            cp = model(x_t, y_t, t_t).cpu().numpy().flatten().reshape(ny, nx)
        ce = finite_domain_series(
            XX.flatten(), YY.flatten(), float(t_val),
            D=cfg["D"], L=L, x0=cfg["x0"], y0=cfg["y0"],
            sigma=cfg["sigma"], N_terms=cfg["N_terms"],
        ).reshape(ny, nx)
        all_pred.append(cp); all_exact.append(ce)

    vmax = max(max(c.max() for c in all_pred), max(c.max() for c in all_exact))
    levels = np.linspace(0, vmax, 30)

    for col, (t_val, cp, ce) in enumerate(zip(t_snaps, all_pred, all_exact)):
        im0 = axes[0, col].contourf(XX, YY, cp, levels=levels, cmap="hot")
        axes[0, col].set_title(f"PINN\nt={t_val}", fontsize=9)
        axes[0, col].set_aspect("equal")
        axes[0, col].axis("off")

        im1 = axes[1, col].contourf(XX, YY, ce, levels=levels, cmap="hot")
        axes[1, col].set_title(f"Exact\nt={t_val}", fontsize=9)
        axes[1, col].set_aspect("equal")
        axes[1, col].axis("off")

    # 공통 컬러바
    fig.colorbar(im0, ax=axes.ravel().tolist(),
                 label="Concentration c", shrink=0.6)
    fig.suptitle("2D Fick Diffusion — PINN vs Analytical (Snapshots)", fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  스냅샷 격자 저장: {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────

def run_visualization(model: torch.nn.Module, device: torch.device):
    cfg = VIZ_CONFIG
    model.eval()

    print("\n" + "=" * 55)
    print("  시각화 생성")
    print("=" * 55)

    # ① 스냅샷 격자 (빠름)
    print("\n[1/2] 스냅샷 격자 생성...")
    make_snapshot_grid(
        model, device, cfg,
        save_path="outputs/figures/snapshot_grid.png"
    )

    # ② GIF animation (시간 걸림)
    print("\n[2/2] Animation 생성...")
    t_frames, pred_frames, exact_frames, XX, YY = precompute_frames(model, device, cfg)
    make_animation(
        t_frames, pred_frames, exact_frames, XX, YY, cfg,
        save_path="outputs/animation.gif"
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "outputs/checkpoints/pinn_best.pt"
    if not os.path.exists(ckpt_path):
        print("[visualize] 체크포인트 없음. 빠른 훈련 후 시각화 실행.")
        cfg = TrainConfig(
            D=1e-2, T=0.5,
            hidden_layers=[64, 64, 64, 64],
            use_fourier=True,
            adam_epochs=3000, lbfgs_epochs=200,
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
        print(f"[visualize] 체크포인트 로드 (epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f})")

    run_visualization(model, device)

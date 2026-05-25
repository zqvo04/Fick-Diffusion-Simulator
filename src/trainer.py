"""
trainer.py — PINN 훈련 루프

전략: Adam → L-BFGS 2단계
  Stage 1 (Adam)  : 빠른 global 탐색. 좋은 초기값 확보.
  Stage 2 (L-BFGS): 2차 근사로 세밀한 수렴. PDE residual 최소화.

콜로케이션 포인트 리샘플링:
  매 epoch마다 새 포인트를 샘플링하면 도메인 커버리지 향상.
  단, 계산 비용 증가. 기본적으로 N epoch마다 한 번 리샘플.

체크포인트:
  최적 손실 갱신 시 자동 저장 (outputs/checkpoints/).
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 같은 src 패키지 내 모듈 임포트
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model   import PINN
from src.sampler import DiffusionSampler
from src.loss    import FickLoss


# ────────────────────────────────────────────────────────────────────────────
# 훈련 설정 dataclass
# ────────────────────────────────────────────────────────────────────────────

class TrainConfig:
    """
    모든 하이퍼파라미터를 한 곳에 모아둠.
    실험 재현성을 위해 configs/default.yaml과 연동 예정.
    """
    # 물리 파라미터
    D: float = 1e-2           # 확산계수 (물 속 소분자 ≈ 1e-9 m²/s, 무차원화 후 1e-2)
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    T: float = 0.5            # 최대 시뮬레이션 시간

    # 초기조건 파라미터
    ic_type: str = "gaussian"
    source_x: float = 0.5
    source_y: float = 0.5
    sigma: float = 0.05
    c_max: float = 1.0

    # 네트워크
    hidden_layers: list = None
    use_fourier: bool = True   # Gaussian IC처럼 뾰족한 패턴 → Fourier 도움
    fourier_dim: int = 64

    # 손실 가중치
    lambda_bc: float = 10.0
    lambda_ic: float = 10.0

    # 샘플링
    n_pde: int = 8000
    n_bc:  int = 800
    n_ic:  int = 800
    resample_every: int = 100  # N epoch마다 콜로케이션 포인트 리샘플

    # Adam 설정
    adam_epochs: int = 3000
    adam_lr: float = 1e-3

    # L-BFGS 설정
    lbfgs_epochs: int = 500    # max_iter
    lbfgs_lr: float = 0.5
    lbfgs_max_iter: int = 100  # 내부 line search 반복 수

    # 기타
    log_every: int = 100       # 손실 출력 주기
    save_dir: str = "outputs/checkpoints"

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 64, 64]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64, 64, 64]


# ────────────────────────────────────────────────────────────────────────────
# 메인 Trainer 클래스
# ────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    PINN 훈련 전체 파이프라인 관리.

    사용법:
        config  = TrainConfig()
        trainer = Trainer(config)
        trainer.train()
        trainer.plot_loss_history()
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] device: {self.device}")

        # ── 모델 초기화 ────────────────────────────────────────────────────
        self.model = PINN(
            hidden_layers=config.hidden_layers,
            x_bounds=config.x_bounds,
            y_bounds=config.y_bounds,
            t_bounds=(0.0, config.T),
            use_fourier=config.use_fourier,
            fourier_dim=config.fourier_dim,
        ).to(self.device)
        print(f"[Trainer] 파라미터 수: {self.model.count_parameters():,}")

        # ── 손실 함수 ──────────────────────────────────────────────────────
        self.criterion = FickLoss(
            D=config.D,
            lambda_bc=config.lambda_bc,
            lambda_ic=config.lambda_ic,
        )

        # ── 샘플러 ─────────────────────────────────────────────────────────
        self.sampler = DiffusionSampler(
            x_bounds=config.x_bounds,
            y_bounds=config.y_bounds,
            T=config.T,
            n_pde=config.n_pde,
            n_bc=config.n_bc,
            n_ic=config.n_ic,
            strategy="lhs",
            device=self.device,
        )

        self.ic_kwargs = {
            "ic_type": config.ic_type,
            "source_x": config.source_x,
            "source_y": config.source_y,
            "sigma": config.sigma,
            "c_max": config.c_max,
        }

        # ── 로그 초기화 ────────────────────────────────────────────────────
        self.history: Dict[str, List[float]] = {
            "total": [], "pde": [], "bc": [], "ic": [], "epoch": []
        }
        self.best_loss = float("inf")
        self.best_epoch = 0

        os.makedirs(config.save_dir, exist_ok=True)

    # ── 포인트 로드 (device 이동 포함) ─────────────────────────────────────
    def _get_batch(self):
        batch = self.sampler.sample_all(ic_kwargs=self.ic_kwargs)
        return batch["pde"], batch["bc"], batch["ic"]

    # ── 체크포인트 저장 ─────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, loss: float, tag: str = "best"):
        path = os.path.join(self.cfg.save_dir, f"pinn_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "model_state": self.model.state_dict(),
            "config": self.cfg.__dict__,
        }, path)

    # ── Stage 1: Adam ──────────────────────────────────────────────────────
    def _train_adam(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.adam_lr)

        # 학습률 스케줄러: 1000 epoch마다 0.5 감소
        # → 초반에는 크게 이동, 후반에는 세밀하게 수렴
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.5
        )

        print(f"\n{'='*55}")
        print(f"  Stage 1: Adam  ({self.cfg.adam_epochs} epochs)")
        print(f"{'='*55}")
        t0 = time.time()

        # 초기 배치 샘플링
        pde_b, bc_b, ic_b = self._get_batch()

        for epoch in range(1, self.cfg.adam_epochs + 1):

            # 주기적 리샘플링: 새 포인트로 교체
            # 이유: 매번 같은 포인트만 보면 특정 영역에 overfitting
            if epoch % self.cfg.resample_every == 0:
                pde_b, bc_b, ic_b = self._get_batch()

            optimizer.zero_grad()
            loss, log = self.criterion(self.model, pde_b, bc_b, ic_b)
            loss.backward()

            # Gradient clipping: 폭발적 gradient 방지
            # PINN은 고차 미분으로 인해 gradient가 클 수 있음
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # 로그 저장
            self.history["total"].append(log["total"])
            self.history["pde"].append(log["pde"])
            self.history["bc"].append(log["bc"])
            self.history["ic"].append(log["ic"])
            self.history["epoch"].append(epoch)

            # 최적 손실 체크포인트
            if log["total"] < self.best_loss:
                self.best_loss  = log["total"]
                self.best_epoch = epoch
                self._save_checkpoint(epoch, log["total"], tag="best")

            if epoch % self.cfg.log_every == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch:5d} | "
                    f"Total={log['total']:.5f}  "
                    f"PDE={log['pde']:.5f}  "
                    f"BC={log['bc']:.5f}  "
                    f"IC={log['ic']:.5f}  "
                    f"lr={lr_now:.2e}  "
                    f"({elapsed:.0f}s)"
                )

        print(f"\n  Adam 완료. 최적 손실: {self.best_loss:.5f} @ epoch {self.best_epoch}")

    # ── Stage 2: L-BFGS ────────────────────────────────────────────────────
    def _train_lbfgs(self):
        """
        L-BFGS는 PyTorch에서 특이한 closure 패턴 필요.

        이유:
          L-BFGS는 내부적으로 Wolfe conditions를 만족하는
          step size를 찾기 위해 손실을 여러 번 재평가함 (line search).
          → optimizer.step()에 closure 함수를 넘겨야 함.
          → closure 내부에서 zero_grad + forward + backward 모두 수행.
        """
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.cfg.lbfgs_lr,
            max_iter=self.cfg.lbfgs_max_iter,
            history_size=50,        # Hessian 근사에 사용할 과거 gradient 수
            line_search_fn="strong_wolfe",  # 안정적인 line search
        )

        print(f"\n{'='*55}")
        print(f"  Stage 2: L-BFGS  ({self.cfg.lbfgs_epochs} steps)")
        print(f"{'='*55}")
        t0 = time.time()

        # L-BFGS는 포인트 고정 (리샘플링 X)
        # 이유: 손실 지형이 바뀌면 2차 근사가 무효화됨
        pde_b, bc_b, ic_b = self._get_batch()
        step_counter = [0]

        def closure():
            optimizer.zero_grad()
            loss, log = self.criterion(self.model, pde_b, bc_b, ic_b)
            loss.backward()

            step_counter[0] += 1

            # 로그 저장 (max_iter번 호출되므로 일부만 저장)
            if step_counter[0] % 10 == 0:
                epoch_tag = self.cfg.adam_epochs + step_counter[0]
                self.history["total"].append(log["total"])
                self.history["pde"].append(log["pde"])
                self.history["bc"].append(log["bc"])
                self.history["ic"].append(log["ic"])
                self.history["epoch"].append(epoch_tag)

                if log["total"] < self.best_loss:
                    self.best_loss  = log["total"]
                    self.best_epoch = epoch_tag
                    self._save_checkpoint(epoch_tag, log["total"], tag="best")

            return loss

        for step in range(1, self.cfg.lbfgs_epochs + 1):
            optimizer.step(closure)

            if step % 50 == 0:
                # 현재 손실 출력 (closure 마지막 호출 기준)
                with torch.no_grad():
                    _, log = self.criterion(self.model, pde_b, bc_b, ic_b)
                elapsed = time.time() - t0
                print(
                    f"  Step  {step:4d} | "
                    f"Total={log['total']:.5f}  "
                    f"PDE={log['pde']:.5f}  "
                    f"BC={log['bc']:.5f}  "
                    f"IC={log['ic']:.5f}  "
                    f"({elapsed:.0f}s)"
                )

        print(f"\n  L-BFGS 완료. 최적 손실: {self.best_loss:.5f} @ step {self.best_epoch}")

    # ── 전체 훈련 실행 ──────────────────────────────────────────────────────
    def train(self):
        print(f"\n[Trainer] 훈련 시작")
        print(f"  D={self.cfg.D}, T={self.cfg.T}, λ_bc={self.cfg.lambda_bc}, λ_ic={self.cfg.lambda_ic}")
        total_start = time.time()

        self._train_adam()
        self._train_lbfgs()

        total_time = time.time() - total_start
        print(f"\n[Trainer] 훈련 완료. 총 소요시간: {total_time:.1f}s")
        print(f"  최종 최적 손실: {self.best_loss:.6f} @ epoch {self.best_epoch}")

        self._save_checkpoint(self.best_epoch, self.best_loss, tag="final")
        self.plot_loss_history()

    # ── 손실 곡선 플롯 ──────────────────────────────────────────────────────
    def plot_loss_history(self, save_path: str = "outputs/figures/loss_history.png"):
        epochs = self.history["epoch"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # 좌: 총 손실
        axes[0].semilogy(epochs, self.history["total"], "k-", lw=1.5, label="L_total")
        axes[0].axvline(self.cfg.adam_epochs, color="gray", linestyle="--",
                        alpha=0.6, label="Adam→L-BFGS")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
        axes[0].set_title("Total Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # 우: 각 항 분리
        axes[1].semilogy(epochs, self.history["pde"], label="L_pde",  color="steelblue")
        axes[1].semilogy(epochs, self.history["bc"],  label="L_bc",   color="tomato")
        axes[1].semilogy(epochs, self.history["ic"],  label="L_ic",   color="seagreen")
        axes[1].axvline(self.cfg.adam_epochs, color="gray", linestyle="--", alpha=0.6)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss (log scale)")
        axes[1].set_title("Loss Components")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[Trainer] 손실 곡선 저장 → {save_path}")
        plt.close()

    # ── 체크포인트 로드 ─────────────────────────────────────────────────────
    def load_best(self):
        path = os.path.join(self.cfg.save_dir, "pinn_best.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"[Trainer] 최적 체크포인트 로드 (epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f})")


# ────────────────────────────────────────────────────────────────────────────
# 스모크 테스트 (소규모 빠른 검증)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = TrainConfig(
        D=1e-2,
        T=0.5,
        hidden_layers=[32, 32, 32],
        use_fourier=False,
        adam_epochs=200,          # 테스트용 축소
        lbfgs_epochs=20,
        n_pde=1000,
        n_bc=200,
        n_ic=200,
        log_every=50,
        resample_every=50,
    )

    trainer = Trainer(cfg)
    trainer.train()
    trainer.load_best()
    print("\n[스모크 테스트] 완료 ✅")

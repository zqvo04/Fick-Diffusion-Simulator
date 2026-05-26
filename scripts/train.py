"""
train.py — 전체 파이프라인 실행 엔트리포인트

사용법:
    # 기본 설정으로 실행
    python scripts/train.py

    # YAML 설정 파일 지정
    python scripts/train.py --config configs/default.yaml

    # 커맨드라인으로 특정 파라미터 오버라이드
    python scripts/train.py --config configs/default.yaml --D 0.005 --adam_epochs 5000

    # 기존 체크포인트에서 이어서 훈련
    python scripts/train.py --resume outputs/checkpoints/pinn_best.pt

파이프라인 순서:
    1. 설정 로드 (YAML → TrainConfig)
    2. 훈련 (Adam → L-BFGS)
    3. 정량 평가 (L2 오차 테이블)
    4. 시각화 (스냅샷, animation)
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model   import PINN
from src.trainer import TrainConfig, Trainer
from scripts.evaluate  import run_evaluation
from scripts.visualize import run_visualization


# ────────────────────────────────────────────────────────────────────────────
# YAML → TrainConfig 변환
# ────────────────────────────────────────────────────────────────────────────

def load_config_from_yaml(yaml_path: str) -> TrainConfig:
    """
    YAML 파일을 읽어 TrainConfig 객체로 변환.
    PyYAML 사용. 없으면 pip install pyyaml.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pip install pyyaml 을 먼저 실행하세요.")

    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    p   = cfg_dict.get("physics", {})
    ic  = cfg_dict.get("initial_condition", {})
    net = cfg_dict.get("network", {})
    ls  = cfg_dict.get("loss", {})
    sp  = cfg_dict.get("sampling", {})
    ad  = cfg_dict.get("adam", {})
    lb  = cfg_dict.get("lbfgs", {})
    tr  = cfg_dict.get("training", {})

    return TrainConfig(
        # 물리
        D           = p.get("D", 1e-2),
        x_bounds    = tuple(p.get("x_bounds", [0.0, 1.0])),
        y_bounds    = tuple(p.get("y_bounds", [0.0, 1.0])),
        T           = p.get("T", 0.5),
        # IC
        ic_type     = ic.get("type", "gaussian"),
        source_x    = ic.get("source_x", 0.5),
        source_y    = ic.get("source_y", 0.5),
        sigma       = ic.get("sigma", 0.05),
        c_max       = ic.get("c_max", 1.0),
        # 네트워크
        hidden_layers = net.get("hidden_layers", [64, 64, 64, 64]),
        use_fourier   = net.get("use_fourier", True),
        fourier_dim   = net.get("fourier_dim", 64),
        # 손실
        lambda_bc   = ls.get("lambda_bc", 10.0),
        lambda_ic   = ls.get("lambda_ic", 10.0),
        # 샘플링
        n_pde          = sp.get("n_pde", 8000),
        n_bc           = sp.get("n_bc", 800),
        n_ic           = sp.get("n_ic", 800),
        resample_every = sp.get("resample_every", 100),
        # Adam
        adam_epochs    = ad.get("epochs", 3000),
        adam_lr        = ad.get("lr", 1e-3),
        # L-BFGS
        lbfgs_epochs   = lb.get("epochs", 200),
        lbfgs_lr       = lb.get("lr", 0.5),
        lbfgs_max_iter = lb.get("max_iter", 100),
        # 기타
        log_every   = tr.get("log_every", 200),
        save_dir    = tr.get("save_dir", "outputs/checkpoints"),
    )


# ────────────────────────────────────────────────────────────────────────────
# 재현성 시드 고정
# ────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """
    모든 난수 소스를 고정해서 실험 재현성 보장.
    PyTorch의 경우 cudnn 동작도 deterministic으로 설정.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────────────────────
# 이어서 훈련 (resume)
# ────────────────────────────────────────────────────────────────────────────

def resume_training(ckpt_path: str, extra_adam_epochs: int = 1000):
    """
    저장된 체크포인트에서 모델을 불러와 Adam으로 추가 훈련.
    손실이 수렴하지 않았을 때 사용.

    Args:
        ckpt_path         : 체크포인트 파일 경로
        extra_adam_epochs : 추가 훈련 epoch 수
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device)
    saved  = ckpt["config"]

    print(f"[resume] 체크포인트 로드: epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}")

    # 저장된 config 복원
    cfg = TrainConfig(
        D             = saved["D"],
        hidden_layers = saved["hidden_layers"],
        use_fourier   = saved.get("use_fourier", False),
        fourier_dim   = saved.get("fourier_dim", 64),
        T             = saved["T"],
        adam_epochs   = extra_adam_epochs,
        lbfgs_epochs  = 100,
    )
    trainer = Trainer(cfg)
    trainer.model.load_state_dict(ckpt["model_state"])
    trainer.best_loss  = ckpt["loss"]
    trainer.best_epoch = ckpt["epoch"]

    # Adam만 추가 실행
    trainer._train_adam()
    trainer._train_lbfgs()
    return trainer


# ────────────────────────────────────────────────────────────────────────────
# CLI 파서
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fick-Diffusion-Simulator: PINN 훈련 & 평가"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML 설정 파일 경로 (default: 코드 내 기본값 사용)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="이어서 훈련할 체크포인트 경로"
    )
    # 커맨드라인 오버라이드 (YAML보다 우선)
    parser.add_argument("--D",            type=float, default=None)
    parser.add_argument("--adam_epochs",  type=int,   default=None)
    parser.add_argument("--lbfgs_epochs", type=int,   default=None)
    parser.add_argument("--lambda_bc",    type=float, default=None)
    parser.add_argument("--lambda_ic",    type=float, default=None)
    parser.add_argument("--n_pde",        type=int,   default=None)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument(
        "--skip_viz", action="store_true",
        help="시각화 생략 (빠른 오차 확인만 할 때)"
    )
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Fick-Diffusion-Simulator")
    print(f"  device: {device}  |  seed: {args.seed}")
    print(f"{'='*55}\n")

    # ── 1. 이어서 훈련 모드 ────────────────────────────────────────────────
    if args.resume:
        trainer = resume_training(args.resume)
        model = trainer.model

    # ── 2. 새 훈련 모드 ────────────────────────────────────────────────────
    else:
        # 설정 로드
        if args.config and os.path.exists(args.config):
            cfg = load_config_from_yaml(args.config)
            print(f"[main] 설정 로드: {args.config}")
        else:
            # 기본값 사용
            cfg = TrainConfig()
            print("[main] 기본 설정 사용")

        # CLI 오버라이드 적용
        # 이유: YAML 기본 설정을 바탕으로 한 파라미터만 바꿔서 빠르게 실험 가능
        overrides = {
            "D"            : args.D,
            "adam_epochs"  : args.adam_epochs,
            "lbfgs_epochs" : args.lbfgs_epochs,
            "lambda_bc"    : args.lambda_bc,
            "lambda_ic"    : args.lambda_ic,
            "n_pde"        : args.n_pde,
        }
        for key, val in overrides.items():
            if val is not None:
                setattr(cfg, key, val)
                print(f"  [override] {key} = {val}")

        # 훈련 실행
        trainer = Trainer(cfg)
        trainer.train()
        model = trainer.model

    # ── 3. 정량 평가 ──────────────────────────────────────────────────────
    print("\n[main] 정량 평가 시작...")
    model.eval()
    l2_errors = run_evaluation(model, device)
    mean_l2   = sum(l2_errors) / len(l2_errors)
    print(f"\n  → 평균 L2 상대 오차: {mean_l2:.4f}")
    if mean_l2 < 0.05:
        print("  → ✅ 우수 (< 5%)")
    elif mean_l2 < 0.10:
        print("  → 🔶 양호 (< 10%)")
    else:
        print("  → ❗ 개선 필요. adam_epochs / lbfgs_epochs 증가 권장.")

    # ── 4. 시각화 ─────────────────────────────────────────────────────────
    if not args.skip_viz:
        print("\n[main] 시각화 생성...")
        run_visualization(model, device)

    print(f"\n{'='*55}")
    print("  완료. 결과물 위치:")
    print("    outputs/checkpoints/pinn_best.pt")
    print("    outputs/figures/")
    print("    outputs/animation.gif")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

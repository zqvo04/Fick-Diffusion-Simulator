# Fick-Diffusion-Simulator

**Physics-Informed Neural Networks (PINNs) for 2D Fick's Diffusion Equation**

PyTorch autograd로 편미분을 직접 계산하여 Fick의 제2법칙을 신경망으로 근사하고, 농도 분포 확산을 2D로 시각화하는 프로젝트입니다.

---

## 데모

![2D Diffusion Animation](outputs/animation.gif)

*PINN 예측(좌) vs 해석해(우) — Gaussian point source의 시간 경과에 따른 확산*

---

## 구현 핵심

### 1. 물리 방정식 — Fick's 2nd Law (2D)

$$\frac{\partial c}{\partial t} = D \left( \frac{\partial^2 c}{\partial x^2} + \frac{\partial^2 c}{\partial y^2} \right)$$

- $c(x, y, t)$ : 농도 (concentration)
- $D$ : 확산계수 (diffusion coefficient)
- 경계조건: $c = 0$ at $\partial\Omega$ (Dirichlet)
- 초기조건: $c(x, y, 0) = C_0 \exp\!\left(-\frac{(x-x_0)^2+(y-y_0)^2}{2\sigma^2}\right)$ (Gaussian point source)

### 2. PINN 손실 함수

$$\mathcal{L}_{total} = \mathcal{L}_{pde} + \lambda_{bc} \cdot \mathcal{L}_{bc} + \lambda_{ic} \cdot \mathcal{L}_{ic}$$

| 항 | 수식 | 역할 |
|----|------|------|
| $\mathcal{L}_{pde}$ | $\frac{1}{N_f}\sum r(x_i,y_i,t_i)^2$ | PDE 잔차 최소화 |
| $\mathcal{L}_{bc}$ | $\frac{1}{N_b}\sum [c_{pred} - 0]^2$ | 경계조건 강제 |
| $\mathcal{L}_{ic}$ | $\frac{1}{N_i}\sum [c_{pred} - c_0]^2$ | 초기조건 강제 |

PDE 잔차 $r = \partial c/\partial t - D(\partial^2 c/\partial x^2 + \partial^2 c/\partial y^2)$ 는 **PyTorch autograd로 직접 계산** (DeepXDE, NeuralPDE 미사용).

### 3. autograd 편미분 계산 방식

```python
c = model(x, y, t)              # forward pass
c_t  = grad(c, t)               # ∂c/∂t
c_x  = grad(c, x)               # ∂c/∂x  (create_graph=True)
c_xx = grad(c_x, x)             # ∂²c/∂x²
c_y  = grad(c, y)               # ∂c/∂y
c_yy = grad(c_y, y)             # ∂²c/∂y²
residual = c_t - D * (c_xx + c_yy)
```

---

## 레포 구조

```
Fick-Diffusion-Simulator/
├── configs/
│   └── default.yaml         # 하이퍼파라미터 설정
├── src/
│   ├── model.py             # PINN 네트워크 (MLP + Fourier Embedding)
│   ├── sampler.py           # Latin Hypercube 콜로케이션 포인트 샘플러
│   ├── loss.py              # ★ PDE 잔차 + 복합 손실 함수 (핵심 구현)
│   ├── trainer.py           # Adam → L-BFGS 2단계 훈련 루프
│   └── analytical.py        # 해석해 (유한 도메인 Fourier 급수)
├── scripts/
│   ├── train.py             # 전체 파이프라인 실행 엔트리포인트
│   ├── evaluate.py          # 정량 오차 평가 (L2 relative error)
│   └── visualize.py         # 2D animation + 스냅샷 생성
├── tests/
│   └── test_loss.py         # pytest 단위 테스트 (19개)
├── outputs/
│   ├── animation.gif
│   ├── checkpoints/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## 빠른 시작

### 설치

```bash
git clone https://github.com/YOUR_USERNAME/Fick-Diffusion-Simulator.git
cd Fick-Diffusion-Simulator
pip install -r requirements.txt
```

### 훈련 실행

```bash
# 기본 설정으로 전체 파이프라인 실행 (훈련 → 평가 → 시각화)
python scripts/train.py --config configs/default.yaml

# 특정 파라미터 오버라이드
python scripts/train.py --config configs/default.yaml --D 0.005 --adam_epochs 5000

# 시각화 생략 (오차만 빠르게 확인)
python scripts/train.py --skip_viz
```

### 개별 단계 실행

```bash
# 평가만 (체크포인트 필요)
python scripts/evaluate.py

# 시각화만
python scripts/visualize.py
```

### 단위 테스트

```bash
pytest tests/test_loss.py -v
```

---

## 결과

### 손실 곡선

![Loss History](outputs/figures/loss_history.png)

Adam(좌) → L-BFGS(우, 회색 점선 이후) 전환 시 급격한 수렴 확인.

### 시각별 스냅샷

![Snapshot Grid](outputs/figures/snapshot_grid.png)

*상단: PINN 예측 / 하단: 해석해 — 시간이 지남에 따라 경계조건을 만족하며 확산*

### 오차 비교

![Error Comparison](outputs/figures/error_comparison_t010.png)

![L2 Error vs Time](outputs/figures/l2_error_vs_time.png)

---

## 설계 결정 사항

| 항목 | 선택 | 이유 |
|------|------|------|
| 활성화 함수 | `tanh` | ∂²c/∂x² 계산 시 ReLU는 2차 미분 = 0 |
| 초기화 | Xavier Normal | tanh와 수학적으로 최적 페어링 |
| 콜로케이션 샘플러 | Latin Hypercube (LHS) | 균일 랜덤 대비 도메인 커버리지 우월 |
| 입력 정규화 | `[-1, 1]` 선형 스케일 | 도메인 크기 무관 안정적 학습 |
| Fourier Embedding | 선택적 적용 | 뾰족한 Gaussian IC의 고주파 성분 학습 가속 |
| 옵티마이저 | Adam → L-BFGS | Adam: 빠른 global 탐색 / L-BFGS: 세밀한 수렴 |
| 해석해 | Eigenfunction Expansion | 유한 도메인 Dirichlet BC 정확 반영 |

---

## 기술 스택

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

- **PyTorch** — autograd 기반 PDE 잔차 계산
- **NumPy / SciPy** — 수치 계산, Latin Hypercube Sampling
- **Matplotlib** — 2D 시각화, GIF animation

---

## 참고 문헌

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.** *Journal of Computational Physics*, 378, 686–707.

2. Tancik, M., et al. (2020). **Fourier features let networks learn high frequency functions in low dimensional domains.** *NeurIPS*.

3. Original PINNs implementation: https://github.com/maziarraissi/PINNs

---

## 라이선스

MIT License

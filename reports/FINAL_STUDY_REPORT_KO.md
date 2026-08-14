# Phase-Aware Memory Placement for Single-GPU RL Training

## 1. 연구 목적

LLM의 preference alignment에 사용되는 RL 학습은 일반적인 supervised fine
tuning보다 많은 memory object를 필요로 한다. 한 GRPO training step에는 Actor와
Reference model이 함께 등장하며, 학습 중에는 parameter, gradient, Adam state와
activation이 서로 다른 시점에 생성되고 사용된다.

이 연구의 질문은 다음과 같다.

> 모든 memory object를 항상 GPU에 상주시킬 수 없는 환경에서, 각 object의
> lifetime을 GRPO phase에 맞춰 관리하면 full-parameter RL 학습을 단일 소형
> GPU에서 실행할 수 있는가?

최종 목표는 단순히 어떤 phase의 memory를 줄이는 것이 아니라, 전체 training
step에서 가장 큰 peak를 만드는 원인을 순서대로 제거하는 것이다.

## 2. 배경: GRPO training step의 phase

이 문서에서는 학습 과정 내부의 구간을 모두 **phase**라고 부른다. 하나의
training step은 다음 phase로 구성된다.

```text
Rollout
-> Reward
-> Actor log-probability
-> Reference log-probability
-> Actor forward
-> Actor backward
-> Update
```

- Actor는 prompt로부터 response를 생성한다.
- Reward model 또는 reward function이 response를 평가한다.
- Reference model은 RL 시작 전 policy를 고정한 모델이며, Actor가 초기 policy에서
  지나치게 멀어지는 것을 막는 penalty 계산에 사용된다.
- Actor/Reference log-probability와 reward로 학습 objective를 구성한다.
- Actor forward와 backward로 gradient를 계산하고 Update에서 parameter를
  갱신한다.

## 3. Memory object와 lifetime

Qwen2.5-0.5B-Instruct FP32 조건에서 주요 model-state 크기는 다음과 같다.

| Memory object | 크기 | GPU 연산에 필요한 phase |
|---|---:|---|
| Actor parameter | 1.843 GiB | Rollout, Actor log-probability, forward, backward, Update |
| Reference parameter | 1.843 GiB | Reference log-probability |
| Gradient | 1.843 GiB | backward에서 순차 생성, Update까지 유지 |
| Adam state | 3.686 GiB | Update |
| 단순 합계 | 약 9.215 GiB | transient memory 제외 |

GPU에 모든 object를 고정 배치하면 phase 전환은 빠르지만, 현재 연산에 참여하지
않는 object도 제한된 GPU memory를 계속 차지한다. 반대로 필요한 phase에서만
GPU에 배치하면 GPU residency를 줄일 수 있지만 CPU–GPU transfer 비용이 생긴다.

## 4. 실험 환경

| 항목 | 설정 |
|---|---|
| GPU | NVIDIA TITAN Xp, 단일 GPU, CUDA 보고 capacity 11.90 GiB |
| 기본 모델 | Qwen2.5-0.5B-Instruct |
| capacity 모델 | Qwen2.5-1.5B-Instruct |
| Dataset | GSM8K |
| 학습 | Full-parameter GRPO, LoRA 사용 안 함 |
| Rollout | Hugging Face eager backend |
| Train batch | prompt 2개 |
| Prompt/response 제한 | 128/64 tokens |
| Compute/rollout dtype | FP32/float32 |
| PPO micro-batch/GPU | 4 |
| 반복 | 최종 성능 비교 구성별 3회 |

실행 길이는 최종 실험 묶음에 따라 다르다.

- GPU AdamW 대 CPU AdamW: 총 32 training steps, warm-up 2개 제외, 30개 측정
- no-stream 대 최종 16 MiB streaming: 총 30 training steps, warm-up 2개 제외,
  28개 측정
- 1.5B 성공 run: 총 32 training steps, warm-up 2개 제외, 30개 측정

따라서 모든 최종 실험이 32 steps라는 단일 조건으로 실행됐다고 쓰지 않는다.
각 run의 실제 측정 개수는 `all_runs.csv`의 `metric_steps`가 기준이다.

## 5. 실험 1: Phase offloading

### 5.1 방법

현재 phase에서 사용하지 않는 memory object를 CPU로 이동한다.

| Memory object | Phase offloading 정책 |
|---|---|
| Actor parameter | Actor가 필요한 phase에만 GPU 배치 |
| Reference parameter | Reference log-probability phase에만 GPU 배치 |
| Gradient | backward에서 생성된 뒤 Update까지 GPU 유지 |
| Adam state | GPU AdamW를 실행하는 Update에만 GPU 배치 |

### 5.2 결과

| Phase | All on GPU | Phase offload |
|---|---:|---:|
| Rollout | 7.41 GiB | 1.88 GiB |
| Actor log-probability | 7.56 GiB | 2.04 GiB |
| Reference log-probability | 7.53 GiB | 2.00 GiB |
| Actor forward | 7.92 GiB | 2.40 GiB |
| Actor backward | 10.24 GiB | 4.72 GiB |
| Update | 10.24 GiB | 8.40 GiB |

Phase offloading은 앞의 다섯 phase에서 불필요한 residency를 크게 줄였다.
하지만 전체 peak를 만드는 Update는 10.24 GiB에서 8.40 GiB로 1.84 GiB만
감소했다. 이 감소량은 Update에서 사용하지 않는 Reference parameter 크기와
일치한다.

GPU AdamW Update에는 Actor parameter, 전체 gradient와 3.686 GiB Adam state가
동시에 필요하다. Phase offloading은 사용하지 않는 object는 제거하지만, 현재
연산에 필요한 object가 만드는 peak까지 제거하지 못한다.

## 6. 실험 2: CPU AdamW

### 6.1 방법

GPU AdamW에서는 Adam state를 CPU에 보관하더라도 Update 전에 GPU로 다시
가져와야 한다. CPU AdamW는 FP32 master parameter, gradient와 Adam state를
CPU에 두고 parameter update 자체를 CPU에서 수행한다.

### 6.2 격리 비교 결과

| 구성 | Update peak | Update time |
|---|---:|---:|
| Phase offload + GPU AdamW | 8.408 GiB | 0.129 s |
| Phase offload + CPU AdamW | 2.365 GiB | 3.565 s |

CPU AdamW는 Update의 GPU optimizer-state peak를 제거하지만, serial CPU update와
parameter copy 때문에 Update가 크게 느려진다. 이는 memory와 speed의 명확한
trade-off다.

전체 phase를 보면 CPU AdamW 적용 후 전체 GPU peak는 Update가 아니라 Actor
backward의 4.72 GiB로 이동한다. CPU AdamW가 optimizer bottleneck을 제거하자
full gradient residency가 새로운 bottleneck으로 드러난 것이다.

## 7. 실험 3: Backward 중 gradient streaming

### 7.1 아이디어

일반적인 no-stream CPU Adam 경로는 모든 parameter의 gradient가 만들어질 때까지
GPU에 보관한 뒤, backward가 끝난 후 전체 gradient를 CPU로 보낸다.

하지만 gradient는 backward 중 layer 순서에 따라 생성된다. 준비된 gradient를
작은 bucket으로 묶어 즉시 CPU로 전송하고 GPU 원본을 해제하면, 전체 gradient가
GPU에 동시에 존재하는 것을 피할 수 있다. 이는 ZeRO-Offload의 gradient
offloading 아이디어를 GRPO의 phase-aware 실행 경로에 적용한 것이다.

최종 0.5B 설정은 다음과 같다.

```text
bucket size                 16 MiB
staging slots               3
asynchronous D2H            enabled
early gradient release      enabled
reusable packing buffers    enabled
direct CPU gradient buffers enabled
CPU gradient accumulation   enabled
Adam–parameter H2D overlap  enabled
telemetry                   disabled for performance runs
```

### 7.2 결과

| 구성 | Backward peak | Backward time | Update time | Actor update | Training step |
|---|---:|---:|---:|---:|---:|
| No-stream | 4.721 GiB | 0.212 s | 3.551 s | 5.152 s | 9.814 s |
| 16 MiB streaming | 3.402 GiB | 0.292 s | 3.020 s | 3.964 s | 8.629 s |

- Backward peak: 1.319 GiB, 27.9% 감소
- Backward time: 약 80 ms 증가
- Update time: 약 531 ms 감소
- End-to-end training step: 1.186 s, 12.1% 단축

Backward time은 hook 호출, gradient packing, D2H enqueue, CUDA event 관리,
조기 release와 memory-bandwidth 경쟁 때문에 증가한다. 기록된 staging-slot
backpressure는 약 41 ms/training step이다.

반면 최종 경로는 gradient를 미리 CPU에 준비하고, CPU Adam bucket update와
updated parameter H2D를 overlap한다. 이 때문에 Update 감소가 backward 증가를
상쇄하고 전체 training step도 짧아진다.

발표 PDF 22번 slide의 “only 1.7% end-to-end overhead” 문구는 같은 slide의
막대 및 최종 raw data와 맞지 않는다. 최종 데이터 기준 결론은 **overhead가
아니라 12.1% training-step 단축**이다.

## 8. 실험 4: Qwen2.5-1.5B capacity

Qwen2.5-1.5B FP32의 model-state 크기는 다음과 같다.

| Memory object | 크기 |
|---|---:|
| Actor parameter | 5.751 GiB |
| Reference parameter | 5.751 GiB |
| Gradient | 5.751 GiB |
| Adam state | 11.502 GiB |
| 단순 최대 model-state 합 | 약 28.755 GiB |

| 구성 | 결과 | 관찰값 |
|---|---|---:|
| All GPU | OOM | 오류 시 PyTorch allocated 11.34 GiB |
| CPU Adam, no-stream | OOM | backward 중 54 MiB 추가 allocation 실패 |
| CPU Adam + gradient stream | 성공 | phase/detail peak 8.44 GiB |

CPU Adam만 사용하면 Update에 도달하기 전에 backward에서 gradient가 누적되어
OOM이 발생한다. OOM snapshot에는 총 338 parameter와 11.34 GiB allocated가
기록되어 있지만, 현재 저장된 event만으로 “gradient byte의 정확히 79%가
누적됐다”는 값을 직접 재계산할 수는 없다. 따라서 GitHub 보고서에서는 OOM이
backward 중 발생했다는 검증 가능한 결론만 사용한다.

Gradient streaming은 backward 중 full-gradient residency를 제한하고, CPU
AdamW는 Update의 optimizer-state residency를 제거한다. 두 기법은 서로 다른
peak 원인을 해결하므로 함께 적용했을 때만 1.5B training이 성공했다.

## 9. 최종 결론

1. Phase offloading은 현재 phase에서 사용하지 않는 Actor, Reference와 optimizer
   state의 불필요한 GPU residency를 제거한다.
2. GPU AdamW를 유지하면 Update에 optimizer state가 다시 필요하므로 전체 peak가
   충분히 줄지 않는다.
3. CPU AdamW는 Update peak를 제거하지만 full gradient가 남아 Actor backward가
   새로운 bottleneck이 된다.
4. Gradient streaming은 backward 중 full-gradient residency를 제거한다.
5. CPU Adam bucket update와 updated-parameter H2D overlap은 serial CPU optimizer
   비용을 완화한다.
6. 두 최적화를 결합해 11.90 GiB GPU에서 Qwen2.5-1.5B FP32 full-parameter GRPO를
   8.44 GiB phase peak로 실행했다.

## 10. 증거와 재현 경로

- 최종 figure provenance: `reports/final-figure-data/README.md`
- 최종 raw data 사본: `reports/final-figure-data/collected/`
- 전체 512-run 원장: `reports/final-figure-data/experiment-history/all_runs.csv`
- 전체 시행착오: `reports/final-figure-data/experiment-history/EXPERIMENT_HISTORY.md`
- 실행 방법과 측정 규칙: `docs/EXPERIMENTS.md`

최종 performance 수치는 telemetry/detail/Nsight를 끈 반복 run에서 읽고,
memory 수치는 phase 시작 시 peak reset이 적용된 memory run에서 읽는다.

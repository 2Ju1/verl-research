# 단일 GPU ZeRO-Offload형 자체 엔진 실험 기획서

## 1. 문서 목적

이 문서는 VERL의 단일 GPU GRPO full-parameter 학습 경로에서 FSDP의 parameter/gradient lifecycle에 의존하지 않는 전용 CPU optimizer offload engine을 설계하고, gradient streaming이 GPU peak memory와 end-to-end 성능에 미치는 효과를 검증하기 위한 구현 및 실험 계획을 정의한다.

여기서 말하는 자체 엔진은 다중 GPU ZeRO sharding 구현이 아니다. 단일 GPU에서는 shard가 전체 모델과 같으므로 분산 sharding에 의한 메모리 절감이 없다. 연구 대상은 ZeRO-Offload의 단일 GPU 실행 원리를 재현하는 다음 배치와 스케줄이다.

```text
GPU: 저정밀 compute parameter, activation, 현재 gradient bucket
CPU: FP32 master parameter, accumulated gradient, AdamW states

Forward
→ Backward 중 gradient bucket별 비동기 D2H
→ GPU gradient storage 회수
→ 전체 backward/accumulation 종료
→ CPU FP32 AdamW
→ 갱신된 parameter bucket별 H2D 및 저정밀 변환
→ 다음 PPO minibatch 또는 다음 step
```

기존 C/G/M 연구에서 M00/M01은 GPU AdamW working set을 제거하여 전체 peak를 약 10.15 GiB에서 약 6.47 GiB로 낮췄지만, 전체 FSDP flat parameter와 gradient를 backward 이후 동기식으로 CPU에 옮긴다. 본 연구는 그 다음 질문에 답한다.

> Gradient를 backward 중 bucket 단위로 CPU에 보내고 GPU storage를 조기에 회수하면, Actor forward/backward peak를 추가로 낮추면서 D2H 비용을 backward에 숨길 수 있는가?

## 2. 연구 질문과 가설

### 2.1 핵심 연구 질문

1. 전체 FP32 gradient를 GPU에 누적하지 않고 bucket 단위로 제한하면 Actor update의 peak allocated memory가 얼마나 감소하는가?
2. Gradient D2H를 backward와 overlap하면 동기식 전체 D2H 대비 step time 증가를 얼마나 숨길 수 있는가?
3. Bucket 크기와 staging-buffer 개수는 memory, PCIe 효율, backward stall 사이에 어떤 trade-off를 만드는가?
4. PPO gradient accumulation에서도 CPU에 누적한 gradient가 기존 학습과 수치적으로 동등한가?
5. CPU AdamW 및 updated-parameter H2D가 새로운 주요 병목이 되는가?

### 2.2 가설

**H1 — GPU peak 감소.** 현재 약 1.84 GiB인 FP32 Actor gradient가 전체로 GPU에 남지 않고 현재 전송 중/생성 중인 bucket으로 제한되므로 Actor backward peak가 유의하게 감소한다.

**H2 — 감소 상한.** 64 MiB double buffer라면 streaming용 gradient storage의 하한은 약 128 MiB이며, 단순 구성요소 계산상 최대 약 1.7 GiB의 allocated-memory 절감 여지가 있다. 실제 절감은 activation, autograd temporary, pack 중 원본 gradient와 bucket의 중첩 때문에 이보다 작다.

**H3 — D2H overlap.** Bucket이 backward 준비 순서대로 구성되고 pinned memory와 별도 CUDA stream을 사용하면 gradient D2H 일부가 나머지 backward와 겹친다.

**H4 — CPU update 병목 유지.** Gradient streaming은 gradient residency와 D2H stall을 개선하지만 CPU AdamW 자체와 updated-parameter H2D는 그대로 남는다. 따라서 memory는 줄어도 step time은 CPU update에 의해 제한될 수 있다.

**H5 — backend 독립 효과.** 동일 자체 엔진 내부에서 동기식 전송과 streaming 전송을 비교하면 FSDP 제거, parameter layout, optimizer 구현 변화의 영향을 통제할 수 있다.

## 3. 연구 범위

### 3.1 포함 범위

- 단일 GPU, single process
- Qwen2.5-0.5B-Instruct full-parameter Actor 학습
- VERL GRPO의 Actor update 경로
- FP16 또는 BF16 GPU compute parameter
- CPU FP32 master parameter와 FP32 AdamW states
- Backward-ready 순서 기반 gradient bucket
- Pinned CPU memory, 별도 CUDA D2H/H2D stream, CUDA event
- PPO micro-batch gradient accumulation
- Global gradient norm 및 clipping
- 기존 rollout/reference 경로와의 parameter 동기화
- Full checkpoint 저장 및 복원
- Memory, phase time, PCIe transfer, overlap 계측

### 3.2 제외 범위

- 다중 GPU parameter/gradient/optimizer sharding
- tensor, pipeline, context parallelism
- NVMe offload
- One-step delayed/stale parameter update
- layerwise parameter evict/prefetch
- optimizer quantization 또는 8-bit Adam
- custom C++/AVX fused CPU Adam kernel의 초기 구현
- activation offload 알고리즘 재설계

이 제외 항목들은 baseline 정확성과 gradient streaming 효과가 확인된 뒤 후속 연구로 다룬다.

## 4. 실험 원칙

### 4.1 독립 변수 분리

FSDP baseline과 자체 엔진 streaming 버전만 직접 비교하면 다음 효과가 섞인다.

- FSDP 제거
- parameter flattening 변경
- optimizer parameter identity 변경
- mixed-precision 구현 변경
- gradient storage 변경
- streaming 추가

따라서 반드시 자체 엔진 안에 동기식 대조군을 둔다.

```text
기존 FSDP GPU Adam
기존 FSDP synchronous CPU Adam
자체 엔진 synchronous CPU Adam
자체 엔진 bucketed synchronous D2H
자체 엔진 bucketed overlapped D2H
```

Streaming의 인과 효과는 마지막 세 구성 사이에서 판단한다. 기존 FSDP 구성은 외부 기준점으로만 사용한다.

### 4.2 정확성 우선 순서

다음 기능을 한 번에 활성화하지 않는다.

1. CPU master parameter와 동기식 CPU AdamW
2. Contiguous bucket layout
3. Bucket별 동기식 D2H
4. 비동기 D2H와 event
5. Double buffering과 backpressure
6. Gradient accumulation의 CPU 누적
7. CPU update와 H2D overlap

각 단계는 이전 단계와 parameter delta 및 loss가 허용 오차 내에서 일치해야 다음 단계로 진행한다.

## 5. 목표 메모리 배치

| 객체 | GPU 배치 | CPU 배치 | 정밀도 |
|---|---|---|---|
| Actor compute parameter | 상주 | 없음 | FP16/BF16 |
| Master parameter | 없음 | 전체, contiguous bucket | FP32 |
| Backward gradient | 현재 생성/전송 bucket만 | accumulated bucket 전체 | GPU 계산 dtype 또는 FP32, CPU FP32 |
| Adam `exp_avg` | 없음 | 전체 | FP32 |
| Adam `exp_avg_sq` | 없음 | 전체 | FP32 |
| Adam step counter | 없음 | bucket 또는 parameter별 | FP32/int |
| Activation | 기존 정책 유지 | 기존 정책에 따름 | 연산 정책에 따름 |
| Reference model | 기존 matched policy 유지 | 기존 정책에 따름 | 기존 설정 유지 |

Qwen2.5-0.5B의 parameter 수를 약 0.5B로 볼 때 CPU 측 model-state 예산은 대략 다음과 같다.

```text
FP32 master parameter    약 1.84 GiB
FP32 accumulated grad    약 1.84 GiB
FP32 Adam exp_avg        약 1.84 GiB
FP32 Adam exp_avg_sq     약 1.84 GiB
합계                     약 7.36 GiB + metadata/staging
```

실제 CPU RSS에는 원본 checkpoint 로딩, tokenizer, Ray worker, rollout/reference storage가 추가된다. 실험 전 시스템 available memory를 검사하고 CPU OOM guard를 둔다.

## 6. 자체 엔진 아키텍처

가칭은 `SingleGPUZeroOffloadEngine`으로 한다. 엔진은 Actor 학습 backend와 optimizer 사이에서 다음 책임을 갖는다.

```text
SingleGPUZeroOffloadEngine
├── ParameterRegistry
├── BucketPlanner
├── GradientOffloadManager
├── CPUGradientAccumulator
├── CPUAdamController
├── ParameterReloadManager
├── StepStateMachine
├── CheckpointAdapter
└── OffloadTelemetry
```

### 6.1 `ParameterRegistry`

모든 trainable parameter에 안정적인 canonical name과 metadata를 부여한다.

```text
name
shape
numel
compute dtype
CPU master dtype
requires_grad
tied/shared group
bucket id
bucket offset
padding length
```

Parameter 순서는 Python 객체 순서에 의존하지 않고 canonical name과 명시적 registry로 고정한다. Tied embedding/output weight는 중복 등록하지 않고 동일 storage group으로 처리한다.

### 6.2 `BucketPlanner`

목표 byte 크기에 맞춰 parameter를 contiguous bucket으로 배치한다.

초기 구현에서는 module 역순, 즉 일반적인 backward 순서를 사용한다. 첫 iteration의 hook timestamp를 기록한 뒤 실제 ready 순서 기반 재계획을 선택적으로 지원한다.

규칙은 다음과 같다.

- Parameter 하나가 목표 bucket보다 크면 독립 oversized bucket으로 둔다.
- Parameter 하나를 임의로 분할하는 tensor slicing은 1차 구현에서 하지 않는다.
- Tied parameter는 마지막 기여 gradient가 끝나는 bucket에 둔다.
- 정렬/padding을 명시적으로 기록한다.
- Bucket layout hash를 checkpoint metadata에 저장한다.

초기 bucket 크기는 64 MiB이며 32/128/256 MiB를 ablation한다.

### 6.3 `GradientOffloadManager`

각 parameter의 최종 accumulated gradient가 준비되는 시점을 hook으로 받는다. Bucket의 모든 entry가 준비되면 별도 D2H stream에서 pinned CPU staging buffer로 복사하고 completion event를 기록한다.

상태 전이는 다음과 같다.

```text
EMPTY → FILLING → READY → COPYING_D2H → OFFLOADED → REUSABLE
```

불변조건은 다음과 같다.

- `READY` 이전에는 D2H를 시작하지 않는다.
- D2H stream은 compute-stream ready event를 기다린다.
- `COPYING_D2H` storage는 completion event 이전에 덮어쓰지 않는다.
- GPU 원본 gradient는 D2H가 안전하게 완료되거나 storage ownership이 보장된 뒤에만 해제한다.
- 다음 bucket을 받을 staging slot이 없으면 backward에 bounded backpressure를 건다.

### 6.4 `CPUGradientAccumulator`

PPO micro-batch accumulation이 있는 경우 CPU에 step gradient를 누적한다.

```text
GPU gradient bucket
→ pinned CPU staging bucket
→ FP32 accumulated-gradient bucket에 add
```

첫 micro-batch는 copy, 이후 micro-batch는 add를 사용한다. Loss normalization과 accumulation scale은 기존 Actor 구현과 동일해야 한다. 마지막 micro-batch 이전에는 optimizer update를 허용하지 않는다.

CPU staging과 accumulation이 직렬 병목이 되지 않도록 CPU thread pool은 초기에는 1개로 고정하고, correctness 이후 2개까지 ablation한다. 과도한 CPU thread는 AdamW 및 Ray와 core/cache bandwidth를 경쟁할 수 있다.

### 6.5 `CPUAdamController`

초기 버전은 correctness를 위해 PyTorch AdamW를 CPU FP32 master parameter에 적용한다. `foreach=False`를 기준으로 사용하여 temporary-memory와 실행 경로의 변동을 줄인다.

Optimizer step 순서는 다음과 같다.

1. 모든 gradient bucket D2H와 CPU accumulation 완료
2. unscale 및 non-finite 검사
3. 모든 bucket의 squared norm 합산
4. global clip coefficient 결정
5. CPU gradient에 clipping 적용
6. AdamW update
7. scheduler step은 기존 VERL 의미와 동일한 위치에서 실행
8. CPU gradient reset

Bucket별 독립 AdamW 인스턴스는 동일한 수학을 단순히 구현할 수 있지만 checkpoint와 parameter group 의미가 복잡해질 수 있다. 1차 버전에서는 하나의 optimizer가 CPU master views 전체를 소유하고 동기식 `step()`을 수행한다. Streaming CPU update는 별도 후속 단계로 둔다.

### 6.6 `ParameterReloadManager`

CPU master parameter update 후 GPU compute parameter에 bucket별 H2D를 수행한다.

```text
CPU FP32 master bucket
→ H2D stream
→ GPU FP16/BF16 compute parameter view
```

초기 구현은 CPU optimizer 전체 종료 후 모든 H2D를 실행하고 synchronize한다. 후속 구현에서 CPU bucket update와 이전 bucket H2D를 overlap한다.

다음 forward 또는 rollout weight export는 모든 관련 H2D completion event 이후에만 시작한다. PPO mini-batch가 여러 개라면 각 optimizer step 후 동일 규칙을 적용한다.

### 6.7 `StepStateMachine`

잘못된 lifecycle 진입을 조기에 검출하도록 step 상태를 명시한다.

```text
IDLE
→ FORWARD
→ BACKWARD_ACCUMULATING
→ GRADIENTS_OFFLOADED
→ CPU_OPTIMIZING
→ PARAMETERS_RELOADING
→ READY_FOR_NEXT_STEP
```

각 public method는 허용 상태를 assertion으로 검사한다. 예외 발생 시 outstanding CUDA event를 정리하고 해당 run을 실패로 기록한다.

### 6.8 `CheckpointAdapter`

Checkpoint에는 다음을 저장한다.

- GPU compute parameter 또는 CPU FP32 master parameter 중 하나의 canonical model state
- CPU AdamW state
- scheduler state
- global training step
- loss-scaler state
- bucket layout version/hash
- compute dtype
- accumulation 설정

복원 시 model parameter name/shape/layout hash를 검증한다. Bucket 크기가 달라도 canonical parameter 단위로 재배치할 수 있게 저장 format은 bucket implementation detail과 분리하는 것이 바람직하다.

## 7. VERL 통합 설계

### 7.1 통합 원칙

Rollout, reference policy, reward, data pipeline은 유지하고 Actor training backend만 선택 가능하게 만든다.

```yaml
actor_rollout_ref:
  actor:
    strategy: single_gpu_zero_offload
    offload_engine:
      enabled: true
      bucket_mb: 64
      num_staging_buffers: 2
      async_d2h: true
      cpu_grad_accumulation: true
      cpu_optimizer: adamw
      master_dtype: fp32
      compute_dtype: fp16
      overlap_h2d_with_cpu_update: false
```

기존 `fsdp`와 새 전략은 동일 실행에서 동시에 Actor parameter를 소유하지 않는다. Reference에는 기존 FSDP 또는 일반 inference 경로를 유지할 수 있으나 비교 구성 전반에서 고정한다.

### 7.2 예상 코드 경계

신규 모듈은 다음과 같이 분리한다.

```text
verl/workers/actor/single_gpu_offload/
├── engine.py
├── registry.py
├── buckets.py
├── grad_offload.py
├── cpu_optimizer.py
├── reload.py
├── checkpoint.py
└── telemetry.py
```

통합 지점은 다음 세 곳으로 제한하는 것을 목표로 한다.

1. Actor model 생성 시 FSDP wrapping 대신 engine 등록
2. Actor backward/optimizer step을 engine lifecycle 호출로 대체
3. Rollout 진입 전 updated Actor parameter export/synchronization

기존 `dp_actor.py`의 PPO loss 계산, minibatch iterator, metrics 계산은 최대한 유지한다. Engine은 model-state lifecycle만 담당한다.

### 7.3 Mixed precision 의미

현재 실험에는 FSDP `param_dtype=fp16`, `reduce_dtype=fp32`, buffer FP32와 forward autocast BF16이 혼재할 수 있다. 새 엔진에서는 dtype을 암묵적으로 상속하지 않고 다음을 명시한다.

- `compute_dtype`: FP16 또는 BF16 중 하나
- `master_dtype`: FP32 고정
- `cpu_grad_dtype`: FP32 고정
- loss computation의 FP32 승격 지점
- loss scaling 사용 여부

첫 matched 실험은 기존 실제 forward dtype을 instrumentation으로 확정한 뒤 동일하게 맞춘다. Dtype이 다르면 memory와 수치 결과 비교를 같은 실험으로 해석하지 않는다.

## 8. 구현 단계와 통과 기준

### 단계 P0 — Baseline 고정 및 재현

목표:

- 기존 FSDP GPU Adam 및 M00/M01 CPU Adam 재실행
- software commit, environment, seed, GPU clock/전력 상태 기록
- Actor parameter/gradient/optimizer memory breakdown 재확인

통과 기준:

- 기존 보고서의 memory peak와 ±0.1 GiB
- step time은 동일 run batch 내 변동 범위 기록
- 동일 입력에서 baseline loss/gradient checksum 확보

### 단계 P1 — 비-FSDP 동기식 CPU master optimizer

목표:

- 일반 Actor parameter에 CPU FP32 master copy 생성
- backward 종료 후 전체 gradient 동기식 D2H
- CPU AdamW 후 전체 parameter 동기식 H2D
- streaming 없이 자체 엔진의 수치 정확성 검증

통과 기준:

- 1 step 후 FP32 master parameter delta가 reference와 `rtol/atol` 기준 충족
- 10 step loss, grad norm, parameter checksum trajectory가 허용 범위 내 일치
- checkpoint save/resume 후 다음 step 결과 일치
- GPU에 Adam state가 남지 않음

### 단계 P2 — Contiguous bucket, 동기식 전송

목표:

- 64 MiB bucket registry/layout 구현
- 전체 backward 후 bucket별 순차 D2H
- P1과 동일 수학을 다른 storage layout에서 재현

통과 기준:

- P1과 gradient/parameter 결과 일치
- bucket coverage 100%, 중복/누락 0
- tied parameter test 통과
- padding 영역 update 없음

### 단계 P3 — Backward 중 비동기 D2H

목표:

- gradient-ready hook
- compute/D2H stream event dependency
- pinned staging buffer
- double buffering과 bounded backpressure
- D2H 완료 후 GPU gradient storage 회수

통과 기준:

- Compute Sanitizer 또는 반복 checksum에서 race 징후 없음
- P2와 결과 일치
- Nsight Systems에서 backward와 D2H overlap 관측
- 전체 gradient 크기와 D2H byte 합 일치
- Actor backward peak가 P2보다 감소

### 단계 P4 — CPU gradient accumulation

목표:

- 여러 PPO micro-batch gradient를 CPU FP32 bucket에 누적
- accumulation boundary에서만 optimizer step
- 기존 loss normalization 유지

통과 기준:

- micro-batch 1과 accumulation 구성별 reference 일치
- accumulation step 중 GPU gradient가 전체 크기로 누적되지 않음
- CPU staging buffer backlog가 bounded 상태 유지

### 단계 P5 — H2D pipeline

목표:

- parameter bucket별 H2D
- CPU Adam bucket 진행과 완료된 bucket H2D overlap 가능성 검증
- 다음 forward/rollout과 정확한 event dependency

통과 기준:

- stale/mixed-version parameter 사용 없음
- parameter H2D byte 합이 예상 모델 크기와 일치
- 동기식 H2D 대비 step time 개선 또는 동일
- GPU peak가 허용 범위 이상 증가하지 않음

### 단계 P6 — 성능 최적화 선택 사항

- Fused/flat CPU Adam kernel
- NUMA pinning 및 CPU affinity
- backward-ready profile 기반 동적 bucket 재배치
- FP16 gradient D2H 후 CPU FP32 승격 ablation
- 3-buffer pipeline
- One-step delayed update는 별도 연구로 분리

## 9. 실험 구성

### 9.1 주 대조군

| ID | Actor backend | Gradient D2H | CPU update | 목적 |
|---|---|---|---|---|
| Z00 | 기존 FSDP | 없음 | GPU AdamW | all-GPU 외부 기준 |
| Z01 | 기존 FSDP CPU mode | backward 후 전체 동기식 | 전체 CPU AdamW | 현재 M형 외부 기준 |
| Z02 | 자체 엔진 | backward 후 전체 동기식 | CPU AdamW | FSDP 제거 효과 분리 |
| Z03 | 자체 엔진 | backward 후 bucket별 동기식 | CPU AdamW | layout/packing 비용 분리 |
| Z04 | 자체 엔진 | backward 중 bucket별 async | CPU AdamW | gradient streaming 핵심 구성 |
| Z05 | 자체 엔진 | Z04 + CPU accumulation | CPU AdamW | 실제 PPO accumulation 구성 |
| Z06 | 자체 엔진 | Z05 | CPU AdamW + bucket H2D overlap | 최종 최적화 구성 |

핵심 인과 비교는 다음과 같다.

- Z01 vs Z02: FSDP lifecycle 제거 효과
- Z02 vs Z03: bucket layout 자체 비용
- Z03 vs Z04: D2H overlap과 조기 gradient 회수 효과
- Z04 vs Z05: CPU accumulation 비용
- Z05 vs Z06: H2D pipeline 효과

### 9.2 Bucket ablation

Z04가 정확성 검증을 통과하면 다음 matrix를 실행한다.

| 변수 | 값 |
|---|---|
| Bucket size | 32, 64, 128, 256 MiB |
| Staging buffers | 1, 2, 3 |
| D2H mode | synchronous, asynchronous |
| CPU grad dtype | FP32 기준; FP16은 별도 탐색 |
| Micro-batch accumulation | 1, 현재 PPO 설정 |

전체 조합을 무조건 실행하지 않는다. 먼저 bucket size × double buffer를 실행해 Pareto 후보를 고르고, 후보에 대해서만 buffer count와 accumulation을 확장한다.

### 9.3 Workload 고정

기존 보고서와 비교할 기본 workload는 다음을 유지한다.

| 항목 | 값 |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Training | Full parameter, LoRA rank 0 |
| Algorithm | GRPO |
| Dataset | GSM8K |
| Train batch | 2 |
| PPO mini-batch | 2 |
| Micro-batch/GPU | 1 |
| Prompt/response max | 128/64 |
| Rollout samples | `n=2` |
| GPU | 동일 단일 GPU |
| Compile | 비활성화 |
| Attention | eager 기준 |

Reference residency, activation checkpointing/offload, rollout engine 설정은 모든 Z 구성에서 동일하게 고정한다. GPU OOM 때문에 Z00만 다른 설정을 쓰게 되면 matched 비교에서 제외하고 외부 참고값으로만 둔다.

### 9.4 반복 프로토콜

- Correctness smoke: 1 step, 고정 입력, seed 고정
- Stability: 10 step, parameter/loss trajectory 기록
- Performance: 구성별 30 measured step
- Warm-up: 앞 5 step 제외
- 독립 실행: 최소 3회
- 구성 실행 순서: 가능한 경우 randomized 또는 interleaved
- 실패 run도 OOM, assertion, NaN, timeout 원인을 보존

## 10. 계측 계획

### 10.1 End-to-end 지표

- Step time
- Tokens/s/GPU
- Actor update time
- Rollout/reference phase time
- CPU RSS peak
- GPU device-used peak
- PyTorch peak allocated/reserved

### 10.2 Engine 내부 지표

Bucket별로 다음 timestamp와 byte를 기록한다.

```text
first_grad_ready
last_grad_ready
d2h_enqueued
d2h_completed
gpu_storage_released
cpu_accumulation_started/completed
cpu_adam_started/completed
h2d_enqueued/completed
bytes
staging_slot
stall_reason/stall_duration
```

파생 지표:

- Total D2H/H2D bytes per step
- Effective PCIe bandwidth
- D2H가 backward와 겹친 시간/비율
- Backpressure stall time
- CPU accumulation time
- CPU AdamW time
- H2D reload time
- 최대 동시 live GPU gradient bytes
- 최대 staging queue depth
- Bucket ready skew

### 10.3 Memory snapshot 지점

```text
before_actor_forward
after_actor_forward
before_backward
각 bucket ready
각 D2H enqueue/complete
각 GPU bucket release
after_backward
before/after CPU AdamW
before/after parameter H2D
after_actor_update
```

`allocated`와 `reserved`를 분리한다. Gradient storage 조기 해제의 주 평가지표는 `torch.cuda.max_memory_allocated()`이다. Allocator cache 때문에 reserved나 `nvidia-smi` device-used가 즉시 감소하지 않는 현상을 실패로 해석하지 않는다.

### 10.4 Timeline 도구

- NVTX range: bucket state 및 phase
- Nsight Systems: CUDA compute, memcpy, CPU Adam thread timeline
- PyTorch CUDA memory history: 대표 smoke run에만 사용
- Engine JSONL: 모든 performance run에서 경량 기록

Profiler overhead가 performance 수치를 오염시키므로 profile run과 최종 timing run을 분리한다.

## 11. 정확성 검증

### 11.1 단위 테스트

- Bucket offset/shape round-trip
- Parameter coverage 및 중복 검출
- Oversized parameter
- Tied/shared parameter
- Padding 보호
- State-machine illegal transition
- Event 완료 전 buffer 재사용 차단
- Gradient accumulation scale
- Global norm/clipping
- AdamW weight decay 및 bias correction
- Checkpoint layout 변경 복원

### 11.2 Reference 비교

작은 deterministic 모델에서 CPU-only reference AdamW와 다음을 비교한다.

- Raw gradient
- Accumulated gradient
- Global grad norm
- Clip coefficient
- `exp_avg`, `exp_avg_sq`
- FP32 master parameter delta
- GPU compute parameter reload 결과

FP16/BF16 cast가 개입하므로 비교 허용 오차는 객체별로 분리한다. FP32 optimizer states는 엄격한 오차를 적용하고 저정밀 compute parameter는 dtype에 맞춘 오차를 적용한다.

### 11.3 장기 안정성

- 100 step smoke에서 NaN/Inf 없음
- Baseline과 loss/reward trajectory 방향성 비교
- Save/resume 경계 전후 동일성
- CPU/GPU memory leak 여부
- Bucket queue가 step마다 초기 상태로 복귀하는지 확인

본 연구는 동일 seed의 장기 training trajectory가 bitwise 동일할 것을 요구하지 않는다. 다만 초기 deterministic step의 optimizer 수학은 reference와 일치해야 한다.

## 12. 성공 기준

### 12.1 필수 성공 기준

1. Z04가 correctness test와 100-step stability를 통과한다.
2. Z04의 Actor update peak allocated memory가 Z03보다 최소 0.5 GiB 감소한다.
3. GPU에 전체 FP32 gradient가 동시에 상주하지 않음을 telemetry로 입증한다.
4. Step당 D2H/H2D byte가 예상 범위와 일치한다.
5. Checkpoint save/resume가 동작한다.

### 12.2 목표 성공 기준

- Z03 대비 Actor update peak 1.0 GiB 이상 감소
- Gradient D2H 시간의 50% 이상을 backward와 overlap
- Z02 동기식 자체 엔진 대비 step time overhead 10% 이내
- 기존 M CPU-Adam 대비 같은 또는 낮은 전체 GPU peak

### 12.3 Stretch 목표

- Actor update peak 5 GiB 이하
- CPU update와 H2D pipeline으로 Z05 대비 유의한 step-time 개선
- 최적 bucket 설정이 3회 반복에서 일관된 Pareto 우위를 보임

목표치는 현재 약 6.47 GiB CPU-Adam peak와 약 1.84 GiB FP32 gradient 관측을 바탕으로 한 사전값이며, P0/P1 instrumentation 결과에 따라 수정할 수 있다.

## 13. 실패 판정과 해석 규칙

### 13.1 Memory가 줄지 않는 경우

다음을 순서대로 확인한다.

1. 원본 `parameter.grad`와 pack bucket이 동시에 살아 있는가?
2. Autograd accumulator가 gradient storage reference를 유지하는가?
3. Peak가 gradient 누적 전 activation/workspace에서 이미 발생하는가?
4. D2H completion 전 staging 부족으로 여러 GPU bucket이 누적되는가?
5. 측정값이 allocated가 아니라 reserved/device-used인가?

원인이 1 또는 2라면 direct-to-bucket gradient storage가 후속 구현 대상이다. 원인이 3이면 gradient streaming의 메모리 상한 자체가 workload에서 낮은 것이므로 유효한 부정 결과다.

### 13.2 속도가 느린 경우

다음을 분해한다.

- Gradient pack kernel 시간
- D2H bandwidth
- Backpressure stall
- CPU accumulation
- CPU AdamW
- FP32→FP16/BF16 cast
- H2D
- Rollout weight sync

CPU AdamW가 지배하면 gradient streaming 실패로 해석하지 않는다. Memory 효과와 D2H overlap 효과를 별도로 보고하고 fused CPU Adam은 후속 과제로 둔다.

### 13.3 수치 불일치

- Loss scaling/unscale 순서
- PPO accumulation normalization
- Gradient clipping 위치
- AdamW weight decay 적용 순서
- Tied parameter 중복 update
- Asynchronous buffer race
- Parameter reload 완료 전 다음 계산 진입
- Compute dtype 불일치

Async를 끈 Z03에서도 불일치하면 수학/layout 문제이고, Z03은 맞지만 Z04만 틀리면 stream/event ownership 문제로 분류한다.

## 14. 주요 위험과 완화책

| 위험 | 영향 | 완화 |
|---|---|---|
| Autograd가 원본 gradient를 유지 | peak 감소 실패 | post-accumulate hook, reference 추적, direct bucket 후속 구현 |
| Async D2H race | silent corruption | stream event, buffer state assertion, checksum |
| CPU AdamW 병목 | GPU idle, 긴 step | 단계별 시간 분해, CPU affinity, fused kernel 후속 |
| CPU RSS 증가 | 시스템 OOM | 사전 memory budget, bounded staging, run guard |
| PPO accumulation 오류 | 학습 의미 변경 | deterministic micro-batch reference test |
| Tied weights 중복 update | 모델 손상 | storage identity registry |
| Dtype 혼재 | 비교 불가능 | 명시적 dtype config와 runtime assertion |
| Backend 변경이 rollout sync에 영향 | stale policy | update version counter, rollout 진입 assertion |
| 너무 작은 bucket | launch/packing overhead | 32–256 MiB ablation |
| 너무 큰 bucket | peak/overlap 악화 | ready skew와 live bytes 계측 |

## 15. 의사코드

```python
engine.begin_step()

for micro_batch_idx, batch in enumerate(minibatches):
    engine.begin_micro_batch(micro_batch_idx)

    loss = actor_forward_and_loss(batch)
    loss.backward()

    # Parameter hooks invoked during backward:
    #   1. write/copy grad into current GPU bucket
    #   2. when bucket becomes ready, enqueue D2H
    #   3. record completion event
    #   4. release/recycle GPU grad storage safely

    engine.end_micro_batch()

engine.finish_gradient_offload()       # wait for outstanding D2H
grad_norm = engine.compute_global_norm()
engine.clip_gradients(grad_norm)
engine.cpu_optimizer_step()            # FP32 master + Adam states on CPU
engine.reload_compute_parameters()     # bucketed H2D + low-precision cast
engine.end_step()
```

Hook와 D2H의 개념적 흐름은 다음과 같다.

```python
def on_parameter_grad_ready(entry, grad):
    bucket = buckets[entry.bucket_id]
    bucket.pack(entry, grad)
    bucket.mark_ready(entry)

    if bucket.is_complete():
        compute_ready = record_event(current_compute_stream())
        with use_stream(d2h_stream):
            d2h_stream.wait_event(compute_ready)
            bucket.cpu_staging.copy_(bucket.gpu_buffer, non_blocking=True)
            bucket.d2h_done.record(d2h_stream)

        bucket.state = COPYING_D2H
```

실제 storage 회수는 `d2h_done`과 autograd ownership을 모두 만족할 때 수행한다.

## 16. 예상 결과 표 형식

### 16.1 핵심 결과

| ID | Step time (s) | Throughput | 전체 peak (GiB) | Actor update peak | CPU RSS | 비고 |
|---|---:|---:|---:|---:|---:|---|
| Z00 | | | | | | GPU Adam |
| Z01 | | | | | | 기존 FSDP CPU Adam |
| Z02 | | | | | | 자체 sync |
| Z03 | | | | | | bucket sync |
| Z04 | | | | | | bucket async |
| Z05 | | | | | | CPU accumulation |
| Z06 | | | | | | H2D pipeline |

### 16.2 Bucket 결과

| Bucket MiB | Buffers | Peak grad bytes | Actor peak | D2H BW | Overlap % | Stall ms | Step time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2 | | | | | | |
| 64 | 2 | | | | | | |
| 128 | 2 | | | | | | |
| 256 | 2 | | | | | | |

## 17. 최종 산출물

1. `SingleGPUZeroOffloadEngine` 구현
2. Engine config schema와 example launch script
3. Unit/integration/correctness tests
4. Z00–Z06 benchmark runner
5. Bucket-level JSONL telemetry와 summary script
6. Nsight Systems 대표 timeline
7. Memory composition 및 overlap 그래프
8. 기존 C/G/M 결과와 연결된 한국어/영어 최종 보고서
9. 구현 한계와 후속 최적화 목록

## 18. 의사결정 기준

P3 종료 시 다음 중 하나로 진행 방향을 결정한다.

### 계속 최적화

- Z04에서 memory peak가 명확히 감소하고
- 수치 정확성이 유지되며
- D2H overlap이 관측되는 경우

이 경우 P4–P6으로 진행한다.

### Direct-to-bucket storage로 재설계

- Gradient pack 중 원본 gradient와 bucket의 중첩이 peak를 지배하거나
- autograd가 원본 gradient storage를 오래 유지하는 경우

이 경우 gradient accumulator storage를 bucket view로 직접 제공하는 설계를 검토한다.

### 부정 결과로 종료

- GPU gradient live bytes는 줄었지만 전체 peak가 activation/workspace 시점에서 결정되고
- sequence/batch 확대 없이 연구 workload에서 유의한 memory 개선이 없는 경우

이 경우 gradient streaming은 정상 작동했으나 현재 workload의 지배 객체가 아니라는 결론을 낸다. 실패한 구현으로 분류하지 않는다.

## 19. 최종 연구 주장 범위

성공 시 주장할 수 있는 범위는 다음과 같다.

> 단일 GPU VERL GRPO full-parameter 학습에서 FP32 optimizer state와 master parameter를 CPU에 유지하고, backward-ready gradient를 bounded bucket으로 streaming offload하면 전체 gradient의 GPU 동시 상주를 피할 수 있다. 이로 인한 Actor backward peak 감소와 D2H overlap의 정도는 bucket 크기, staging depth, autograd storage lifecycle에 의해 결정된다.

다음 주장은 별도 검증 없이 하지 않는다.

- 다중 GPU ZeRO/FSDP보다 항상 우수하다.
- 모든 모델 크기와 sequence length에서 같은 절감률을 보인다.
- CPU AdamW 비용이 완전히 숨겨진다.
- Reserved/device-used memory도 allocated와 같은 폭으로 감소한다.
- Delayed update 없이 CPU optimizer 전체가 GPU compute와 완전히 overlap된다.

이 기획의 핵심은 FSDP를 제거하는 것 자체가 아니라, **gradient가 생성되고 전송되고 해제되는 시점을 자체 엔진이 소유하여 gradient streaming의 효과와 한계를 독립적으로 측정하는 것**이다.

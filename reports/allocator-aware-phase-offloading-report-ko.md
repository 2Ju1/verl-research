# Allocator-Aware Phase Offloading for Memory-Efficient Single-GPU GRPO Training

## 초록

단일 GPU에서 full-parameter GRPO를 수행하면 Rollout, Actor log-prob, Reference log-prob 및 Actor update가 순차 실행되며, Actor/Reference parameter, optimizer state, gradient, activation과 CUDA allocator cache가 서로 다른 phase의 GPU peak를 형성한다. 본 연구는 Actor와 Reference를 phase별로 교대 상주시켜 불필요한 동시 residency를 제거하고, CPU optimizer로 GPU optimizer state를 제거하며, backward 중 gradient를 bounded bucket 단위로 CPU에 비동기 전송하고 조기 해제하는 실행 경로를 구현했다.

Gradient streaming은 live CUDA allocation을 약 1 GiB 줄였지만 PyTorch native caching allocator에서는 reserved 및 driver-visible peak가 거의 감소하지 않았다. 원인 분석 결과, 해제된 gradient 자체보다 forward/backward 동안 형성된 allocator segment의 high-water mark가 실제 device peak를 유지하고 있었다. `expandable_segments`를 gradient streaming과 결합하자 동일 allocator 조건에서 no-streaming 대비 reserved/device peak가 실제로 감소했다. 12개 group/slot 조합을 screening한 결과 1–16 MiB와 slot 1개가 동일한 최소 reserved/device 영역을 형성했으며, 그중 8 MiB·slot 1개인 `PA-MEM-B08-S1`을 대표 최소-memory 설정으로 선택했다. 1회 smoke 측정값은 allocated/reserved/device peak 3.214/3.473/3.792 GiB다.

현재 결과는 allocator-aware gradient lifecycle 관리가 알고리즘적 tensor-memory 감소를 실제 GPU capacity 감소로 변환하는 데 필요함을 보여준다. 최종 발표를 위해서는 동일 실행 조건의 반복 측정, OOM frontier 및 correctness 검증이 추가로 필요하다.

## 1. 연구 문제

일반적인 pretraining step과 달리 GRPO에는 다음 phase가 존재한다.

```text
Rollout
→ Actor log-prob
→ Reference log-prob
→ Actor update forward
→ Actor update backward
→ Optimizer update
```

단일 GPU에서는 다음 메모리가 peak에 기여한다.

- Actor 및 frozen Reference parameter
- GPU 또는 CPU optimizer state
- Actor gradient
- activation 및 checkpoint recomputation temporary
- CUDA kernel/library workspace
- PyTorch caching allocator가 보유한 inactive segment

본 연구의 핵심 질문은 다음과 같다.

> Full-parameter GRPO를 단일 GPU에서 실행할 때 Actor, Reference, optimizer 및 gradient의 residency와 이동을 어떻게 스케줄링해야 실제 GPU peak를 최소화할 수 있는가?

## 2. 제안 실행 구조

### 2.1 Actor–Reference phase-exclusive residency

```text
Rollout / Actor log-prob: Actor GPU 상주
Reference log-prob:       Actor 제거, Reference GPU 상주
Actor update:             Reference 제거, Actor GPU 상주
```

별도 frozen Reference가 존재하고 Actor와 Reference가 순차 실행된다는 GRPO phase semantics를 이용해 두 모델의 불필요한 동시 residency를 제거한다.

### 2.2 CPU optimizer

CPU에 FP32 master parameter와 Adam state를 유지하고 GPU에는 update forward/backward에 필요한 FP16 Actor parameter만 둔다. Backward gradient는 CPU로 전달되고 CPU AdamW가 master parameter를 갱신한 뒤 GPU FP16 parameter를 갱신한다.

### 2.3 Bounded asynchronous gradient streaming

```text
parameter gradient 생성
→ bounded GPU packing/group buffer
→ pinned CPU staging buffer로 async D2H
→ parameter.grad early release
→ CPU FP32 gradient/optimizer update
```

핵심 속성은 다음과 같다.

- backward hook 기반 gradient readiness 추적
- deterministic bucket layout
- bounded pinned CPU staging slots
- 별도 D2H CUDA stream과 CUDA event lifetime 관리
- 전송 enqueue 이후 원본 `parameter.grad` 조기 해제
- 작은 group 설정에서 oversized parameter gradient direct D2H

### 2.4 Allocator-aware execution

Native caching allocator에서는 크기와 lifetime이 다른 forward/backward allocation이 독립 segment를 형성하고, gradient가 해제돼도 해당 segment가 cache에 남을 수 있다. 본 연구는 다음 allocator 설정을 gradient streaming과 함께 사용한다.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

이 설정은 변동하는 allocation을 확장 가능한 segment에 배치해, streaming으로 해제된 공간이 이후 요청에 재사용될 가능성을 높인다.

## 3. 측정 방법

메모리 지표를 다음처럼 구분한다.

| 지표 | 의미 | 역할 |
|---|---|---|
| Peak allocated | 살아 있는 PyTorch CUDA tensor allocation | 알고리즘 및 tensor lifecycle 평가 |
| Peak reserved | PyTorch allocator가 확보한 CUDA segment | allocator footprint 평가 |
| Device-used peak | CUDA context와 PyTorch 외 allocation을 포함한 driver-visible 사용량 | 실제 GPU 요구량의 주 proxy |
| OOM frontier | 설정별 최소 실행 가능 memory limit | 최종 capacity 증거 |

측정 phase는 다음과 같이 분리한다.

- Rollout
- Actor log-prob
- Reference log-prob
- Actor update forward
- Actor update backward
- Actor optimizer

현재 결과는 Qwen2.5-0.5B-Instruct, FP16 compute, eager attention, gradient checkpointing, 단일 NVIDIA TITAN Xp 환경에서 수집됐다. 별도로 명시하지 않은 최신 allocator 비교는 7 training step, 2 warm-up step, 5 measured step, 1회 실행의 smoke 결과다.

## 4. 결과

### 4.1 GPU optimizer에서 CPU optimizer로의 이전

| 설정 | Peak allocated | Peak reserved | Device peak |
|---|---:|---:|---:|
| GPU optimizer: R-CCC | 8.410 | 9.264 | 9.599 |
| CPU optimizer: O-PART | 4.222 | 4.889 | 5.206 |
| 감소 | 4.188 | 4.375 | 4.393 |

CPU optimizer는 optimizer state 및 update allocation을 GPU에서 제거해 device peak를 약 4.4 GiB 줄였다. 다만 이 비교는 서로 다른 기존 실험 묶음에서 취합됐으므로 최종 발표 전 동일 실행 세대에서 재측정해야 한다.

### 4.2 Native allocator가 gradient streaming 효과를 가리는 현상

| 설정 | Peak allocated | Peak reserved | Device peak |
|---|---:|---:|---:|
| CPU optimizer: O-PART | 4.222 | 4.889 | 5.206 |
| Native PA-ASYNC | 3.235 | 4.875 | 5.196 |
| 감소 | 0.987 | 0.014 | 0.010 |

Gradient streaming은 live allocation을 약 0.99 GiB 줄였지만 reserved/device peak에는 거의 반영되지 않았다. Backward 종료 시 gradient가 해제되어 current allocated가 약 1.857 GiB까지 감소했음에도 reserved는 4.875 GiB에 남았다.

Fixed GPU packing buffer를 2×64 MiB로 제한한 실험에서도 peak는 줄지 않았으며 오히려 고정 buffer 크기만큼 증가했다. 따라서 64 MiB packing buffer 개수 자체가 native peak의 주원인은 아니었다.

### 4.3 동일 allocator에서 분리한 bucket streaming 효과

`expandable_segments:True`를 고정하고 gradient streaming 유무만 비교했다.

| 설정 | Gradient 정책 | Allocated | Reserved | Device | Step |
|---|---|---:|---:|---:|---:|
| PA-PART-EXPAND | 전체 GPU gradient 유지 | 4.206 | 4.342 | 4.659 | 11.975 s |
| PA-EXPAND | 64 MiB async streaming | 3.208 | 3.570 | 3.892 | 12.076 s |
| 감소 |  | 0.999 | 0.771 | 0.768 | +0.8% |

동일 allocator 기준에서 gradient streaming은 reserved peak를 17.8%, device peak를 16.5% 줄였다. 따라서 bucket streaming은 live tensor allocation뿐 아니라 실제 device-visible GPU 요구량에도 유의미한 영향을 준다.

### 4.4 현재 메모리 최적 후보

| 설정 | Gradient group | Allocated | Reserved | Device | Step |
|---|---:|---:|---:|---:|---:|
| PA-NATIVE | 64 MiB, native allocator | 3.235 | 4.875 | 5.196 | 12.757 s |
| PA-EXPAND | 64 MiB | 3.208 | 3.570 | 3.892 | 12.076 s |
| PA-MEM-B08-S1 | 8 MiB, slot 1 | 3.214 | 3.473 | 3.792 | 12.445 s |
| PA-MEM-B32-S1 | 32 MiB, slot 1 | 3.231 | 3.492 | 3.812 | **11.780 s** |

현재 대표 최소-memory 설정은 `PA-MEM-B08-S1`이다. Native PA-ASYNC 대비 reserved는 1.402 GiB(28.8%), device peak는 1.404 GiB(27.0%) 감소했다. 동일 expandable allocator의 no-streaming 통제군 `PA-PART-EXPAND`와 비교하면 reserved/device가 각각 0.869/0.867 GiB 감소한다.

`PA-MEM-B32-S1`은 device peak가 최소 설정보다 0.020 GiB만 높으면서 screening step time이 가장 짧다. 따라서 8 MiB·slot 1개를 최소-memory 대표점으로, 32 MiB·slot 1개를 memory–performance Pareto 후보로 유지한다. 성능 차이는 1회 smoke 결과이므로 반복 실행 전에는 통계적 우위를 주장하지 않는다.

### 4.5 Group size 및 staging slot sweep

`expandable_segments:True`, async D2H, early release 및 reusable packing buffer를 고정하고 6개 group size와 staging slot 1/2개를 비교했다.

| Group | Slots | Allocated | Reserved | Device | Step |
|---:|---:|---:|---:|---:|---:|
| 1 MiB | 1 | 3.208 | **3.473** | **3.792** | 12.606 s |
| 1 MiB | 2 | 3.208 | **3.473** | 3.794 | 12.605 s |
| 4 MiB | 1 | 3.211 | **3.473** | **3.792** | 12.840 s |
| 4 MiB | 2 | 3.214 | **3.473** | 3.794 | 12.639 s |
| 8 MiB | 1 | 3.214 | **3.473** | **3.792** | 12.445 s |
| 8 MiB | 2 | 3.221 | **3.473** | 3.794 | 12.396 s |
| 16 MiB | 1 | 3.214 | **3.473** | **3.792** | 12.498 s |
| 16 MiB | 2 | 3.221 | **3.473** | 3.794 | 12.641 s |
| 32 MiB | 1 | 3.231 | 3.492 | 3.812 | **11.780 s** |
| 32 MiB | 2 | 3.254 | 3.512 | 3.833 | 11.886 s |
| 64 MiB | 1 | 3.270 | 3.531 | 3.851 | 12.300 s |
| 64 MiB | 2 | 3.332 | 3.980 | 4.302 | 12.011 s |

주요 관찰은 다음과 같다.

- 1–16 MiB에서는 slot 1개의 reserved/device peak가 모두 3.473/3.792 GiB로 동일했다.
- 같은 group size에서 slot 2개는 최소 영역에서 약 2 MiB, 32 MiB에서는 약 22 MiB, 64 MiB에서는 약 451 MiB의 device 증가를 보였다.
- 64 MiB·slot 2개는 bounded buffer와 allocator segment의 중첩으로 다른 조합보다 명확히 불리했다.
- 절대 최소 group size를 1 MiB까지 줄일 필요는 없으며, 8 MiB·slot 1개가 같은 최소 peak에서 더 적절한 대표점이다.
- 32 MiB·slot 1개는 20 MiB의 device 증가와 더 짧은 screening time을 교환하는 Pareto 후보다.

### 4.6 실패하거나 채택하지 않은 방법

| 방법 | 결과 | 판정 |
|---|---|---|
| 2×64 MiB fixed GPU packing buffer | reserved/device 감소 없음 | packing buffer 수가 native peak의 주원인이 아님 |
| Forward 종료 후 `empty_cache()` | 단독 효과 약 0.1 GiB, expandable과 결합 시 peak 증가 | 최종 기법에서 제외 |
| `cudaMallocAsync` + trim | reserved/device 4.062/4.384 GiB | expandable보다 불리 |
| 1 MiB direct groups without expandable | reserved/device 4.877/5.199 GiB | allocator 설정 없이는 효과가 가려짐 |

## 5. 현재 발표 가능한 기여

1. **GRPO phase-aware Actor–Reference residency**  
   Actor와 frozen Reference의 교대 실행을 이용해 불필요한 동시 GPU residency를 제거했다.

2. **단일 GPU full-parameter CPU optimizer 경로**  
   FSDP sharding에 의존하지 않고 CPU FP32 master parameter와 Adam state를 관리하는 실행 엔진을 구현했다.

3. **Bounded asynchronous gradient streaming**  
   Backward hook, bounded staging slots, async D2H 및 early gradient release를 결합해 전체 GPU gradient 누적을 제거했다.

4. **Allocator-aware offloading 분석**  
   Allocated 감소가 reserved/device 감소를 보장하지 않음을 보이고, allocator segment 정책을 통제해야 gradient lifecycle 개선이 실제 GPU capacity 개선으로 전환됨을 정량화했다.

5. **Phase/subphase memory methodology**  
   Rollout, Actor log-prob, Reference log-prob와 Actor update forward/backward/optimizer를 분리하고 allocated/reserved/device-used를 동시에 보고한다.

## 6. 주장 범위와 한계

현재 결과로 다음을 주장해서는 안 된다.

- CPU optimizer, async gradient offload 또는 expandable segments 자체를 최초 제안했다.
- 8 MiB가 모든 GPU와 모델에서 전역 최적이다.
- 최소 실행 가능 GPU 용량이 정확히 3.792 GiB다.
- Actor–Reference reload 시간 최적화가 완료됐다.
- 1회 smoke 결과만으로 성능 차이가 통계적으로 유의하다.

최종 주장은 기존 기법의 개별 발명이 아니라, **GRPO phase schedule에서 이들을 결합하고 allocator 상호작용을 규명한 시스템 설계와 측정 결과**에 두어야 한다.

## 7. 재실험 계획

### P0. 완료: bucket-memory sweep

`expandable_segments`를 고정한 상태에서 메모리 최소 group size와 staging slot 수를 찾는 screening을 완료했다.

| 변수 | 값 |
|---|---|
| Bucket/group size | 1, 4, 8, 16, 32, 64 MiB |
| D2H staging slots | 1, 2 |
| 총 설정 | 12 |
| 반복 | 1회 screening 완료 |
| Steps | 7, warm-up 2, measured 5 |

결과는 `outputs/pa-memory-sweep-v1/summary/memory_phase_table.csv`에 저장돼 있다. 최소-memory 대표점은 `PA-MEM-B08-S1`, Pareto 후보는 `PA-MEM-B32-S1`이다. 최종 발표용 반복 실행에서는 이 두 설정만 재검증하고 나머지 10개 조합은 screening 결과로 사용한다.

### P1. 동일 조건 핵심 baseline 재실행

최종 비교표는 서로 다른 과거 실험 디렉터리의 수치를 혼합하지 않아야 한다. 다음 설정을 같은 코드 revision, GPU, 데이터, seed, step 수 및 allocator 정책에서 새 프로세스로 실행한다.

| ID | 목적 |
|---|---|
| FINAL-GPU-OPT | GPU optimizer baseline |
| FINAL-CPU-NOSTREAM | CPU optimizer, 전체 GPU gradient 유지 |
| FINAL-CPU-STREAM64 | 64 MiB async gradient streaming |
| FINAL-CPU-BEST | `PA-MEM-B08-S1`: 8 MiB, slot 1개 |
| FINAL-CPU-PARETO | `PA-MEM-B32-S1`: 32 MiB, slot 1개 |

두 종류의 allocator 질문을 분리한다.

1. Native allocator에서 알고리즘적 allocated 변화
2. Expandable allocator에서 실제 reserved/device 변화

발표용 최종 실행은 설정별 30 step, warm-up 5, 3회 반복을 권장한다. 현재 요청대로 우선 1회 실행한 뒤 결과가 안정적이면 최종 반복 수를 늘린다. 메모리 주장은 `FINAL-CPU-BEST`, 성능과의 trade-off는 `FINAL-CPU-PARETO`를 사용한다.

### P2. 실제 capacity 검증

Device peak만으로 최소 실행 가능 GPU 용량을 확정할 수 없다. 다음 중 하나가 필요하다.

- MIG 또는 더 작은 물리 GPU에서 성공/OOM 경계 측정
- `torch.cuda.set_per_process_memory_fraction()`을 이용한 allocator-limit sweep
- 고정 memory pressure 아래 설정별 성공/OOM 비교

최소한 `FINAL-CPU-NOSTREAM`과 `FINAL-CPU-BEST`를 비교해 baseline은 실패하지만 best가 성공하는 memory limit 구간을 찾아야 한다.

### P3. Correctness 검증

메모리 및 속도보다 먼저 다음을 검증한다.

- 동일 seed와 batch에서 CPU master gradient 비교
- no-streaming과 streaming의 gradient norm 비교
- optimizer 1-step 후 FP32 master parameter 차이
- GPU FP16 parameter reload 후 차이
- NaN/Inf 및 missing/duplicate gradient 검사
- 전송 byte 수가 전체 parameter gradient 크기와 일치하는지 확인

권장 허용 오차는 FP16 backward 및 FP32 CPU accumulation을 고려해 별도로 정의한다.

### P4. 대표 설정 allocator trace

다음 두 설정에 memory history를 활성화해 representative trace를 수집한다.

- FINAL-CPU-NOSTREAM
- FINAL-CPU-BEST

필요 산출물:

- segment allocation/free timeline
- active/inactive block 크기 분포
- backward 중 reserved 증가 시점
- allocation retry 및 OOM event

이 trace는 allocator-aware 주장의 원인 증거로 사용한다.

### P5. 속도 후속 실험

메모리 최종 설정을 고정한 후에만 수행한다.

| 설정 | 내용 |
|---|---|
| T0 | 순차 Actor reload 및 순차 updated-parameter H2D |
| T1 | bounded double-buffer Actor reload |
| T2 | CPU Adam group update와 updated-parameter H2D overlap |

Actor–Reference phase 전환 최적화와 일반 ZeRO-Offload형 CPU Adam/H2D overlap을 구분해서 보고한다. Layer-wise Ref eviction과 Actor prefetch는 reload 시간이 전체 step에서 충분히 큰 경우에만 후속 연구로 진행한다.

## 8. 발표용 필수 표와 그림

1. 설정별 전체 allocated/reserved/device peak 표
2. Rollout, Actor log-prob, Reference log-prob, Update forward/backward/optimizer 통합 표
3. Native PA-ASYNC memory timeline
4. Expandable allocator 적용 후 동일 timeline
5. Bucket size–device peak–step time Pareto plot
6. CPU optimizer 및 gradient streaming 단계별 waterfall chart
7. Representative allocator segment trace

## 9. 현재 결론

현재 결과는 CPU optimizer가 단일 GPU GRPO update의 가장 큰 model-state peak를 제거하고, bounded gradient streaming이 남은 전체 gradient 누적을 줄인다는 것을 보여준다. Native allocator에서는 이 감소가 cache high-water mark에 가려졌으나, expandable-segment allocator와 결합하면 동일 allocator의 no-streaming baseline 대비 device peak가 약 0.77 GiB 감소했다.

따라서 본 연구의 핵심 결론은 다음과 같다.

> 단일 GPU GRPO의 실제 memory capacity를 줄이려면 Actor–Reference phase residency와 gradient lifecycle뿐 아니라 CUDA allocator의 segment reuse까지 함께 설계하고 측정해야 한다.

# VERL 단일 GPU 오프로딩 연구: C, G, M 최종 결과

## 1. 보고 범위와 연구 질문

이 보고서는 단일 GPU에서 VERL GRPO full-parameter 학습을 수행한 **C, G, M 시리즈의 최종 결과**를 통합한다. 각 시리즈는 서로 다른 질문에 답한다.

- **C 시리즈 — 순방향 ablation:** 모든 객체가 GPU에 있는 기준점에서 Reference 파라미터, AdamW optimizer state, saved activation을 하나씩 offload하면 어떤 변화가 생기는가?
- **G 시리즈 — 역방향 residency ablation:** Actor 파라미터를 GPU에 고정해 Actor 이동이라는 교란요인을 제거한 상태에서, 추가로 어떤 객체를 GPU에 상주시켜야 하는가?
- **M 시리즈 — optimizer 실행 위치:** optimizer state뿐 아니라 AdamW 계산 자체를 CPU로 옮기면 GPU 메모리가 얼마나 줄고 성능 비용이 얼마나 발생하는가?

세 시리즈는 함께 해석해야 한다. C는 all-GPU endpoint에서 offload를 도입하는 비용을 측정한다. G는 Actor residency를 고정한 상태에서 같은 정책을 반대 방향으로 검증한다. M은 optimizer 실행 장치를 바꿔 GPU AdamW 병목을 제거한 뒤 어떤 메모리 병목이 새로 드러나는지 보여준다.

최종 정책 결론은 C00–C03, G00–G03, M00–M02만 사용한다. 조합형 C 구성과 allocator 진단용 A 시리즈는 원인 분석에는 유용하지만 최종 비교 집합에서는 제외한다.

## 2. 실험 환경과 지표 정의

| 항목 | 설정 |
|---|---|
| Framework | VERL single-controller GRPO, Ray worker, PyTorch FSDP1 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, full-parameter training (`LoRA rank=0`) |
| Dataset | GSM8K |
| Hardware | 약 11.9 GiB 메모리를 가진 단일 GPU |
| Batch | train batch 2, PPO mini-batch 2, micro-batch/GPU 1 |
| Sequence | 최대 prompt 128, 최대 response 64, prompt당 rollout sample `n=2` |
| 반복 프로토콜 | 구성별 30 step, 독립 실행 3회, 앞 5 step 제외 |
| M detail에서 관찰된 정밀도 | FSDP flat Actor parameter 및 gradient FP32 저장 |
| 주요 지표 | 평균 step time, tokens/s/GPU, peak PyTorch CUDA allocation, phase time, phase peak allocation |

그래프의 `(n=3)`은 rollout sample 수가 아니라 **독립 반복 실행 3회**를 뜻한다. rollout 설정은 `n=2`이다.

### 지표의 측정 범위

- **Step time**은 warm-up 제외 후 end-to-end 학습 step의 평균 시간이다.
- **Throughput**은 GPU당 초당 token 수이며 높을수록 좋다.
- **Peak allocated GPU memory**는 PyTorch가 추적한 live CUDA tensor allocation의 최댓값으로, 이 보고서의 주 메모리 지표다.
- **GPU device-used memory**에는 CUDA context, allocator reserved block, library 및 driver-visible allocation이 추가로 포함된다. Tensor가 CPU로 이동해도 이 값은 높게 남을 수 있다.
- Phase-level `Actor update` peak는 Actor forward/backward부터 optimizer까지 전체 update phase의 최댓값이다. 반드시 AdamW 산술 연산 중 발생한 peak라는 뜻은 아니다.
- Phase 값은 성공한 3회 실행의 평균이다. Tensor shape와 allocation 경로가 고정되어 있어 메모리 표준편차는 거의 0이다.

C, G, M은 서로 다른 실행 batch에서 수집되었으므로 시리즈 사이 절대 시간에는 실행 시점 차이가 포함될 수 있다. 가장 강한 인과 비교는 각 시리즈 내부의 matched configuration이며, 시리즈 간 결론은 반복적으로 나타난 방향성에 기반한다.

## 3. C 시리즈: 순방향 오프로딩 ablation

네 구성 모두 Actor는 GPU에 상주하고 AdamW 계산도 GPU에서 수행한다. C01–C03은 C00에서 객체 하나만 변경한다.

| ID | Reference 파라미터 | Optimizer state | Saved activation | Actor 파라미터 | 의미 |
|---|---|---|---|---|---|
| C00 | GPU 상주 | GPU 상주 | GPU | GPU 상주 | all-GPU 기준점 |
| C01 | phase 사이 CPU offload | GPU 상주 | GPU | GPU 상주 | Reference만 offload |
| C02 | phase별 CPU↔GPU swap | 사용하지 않을 때 CPU | GPU | GPU 상주 | optimizer-state만 offload |
| C03 | saved tensor GPU→CPU offload | GPU 상주 | CPU-backed | GPU 상주 | activation만 offload |

### 3.1 End-to-end 결과

| ID | Step time (s) | Throughput (tokens/s/GPU) | 전체 GPU peak (GiB) | CPU RSS peak (GiB) | C00 대비 |
|---|---:|---:|---:|---:|---|
| C00 | 6.656 | 100.92 | 11.065 | 1.237 | 기준 |
| C01 | 6.923 | 97.06 | 10.149 | 2.486 | GPU −0.916 GiB; step +4.0% |
| C02 | 7.505 | 89.39 | 11.065 | 5.233 | 전체 peak 감소 없음; step +12.8% |
| C03 | 7.146 | 94.09 | 11.066 | 1.762 | 전체 peak 감소 없음; step +7.4% |

### 3.2 Phase 실행시간

| 구성 | Rollout (s) | Actor log-prob (s) | Reference log-prob (s) | Actor update (s) |
|---|---:|---:|---:|---:|
| C00 | 4.562 | 0.321 | 0.311 | 1.452 |
| C01 | 4.548 | 0.322 | 0.568 | 1.472 |
| C02 | 4.987 | 0.325 | 0.325 | 1.857 |
| C03 | 4.559 | 0.326 | 0.328 | 1.923 |

### 3.3 Phase별 peak allocated GPU memory

| 구성 | Rollout (GiB) | Actor log-prob (GiB) | Reference log-prob (GiB) | Actor update (GiB) |
|---|---:|---:|---:|---:|
| C00 | 7.409 | 7.687 | 7.021 | 11.064 |
| C01 | 6.507 | 6.767 | 7.021 | 10.149 |
| C02 | 3.728 | 4.007 | 3.340 | 11.064 |
| C03 | 7.409 | 7.687 | 7.021 | 11.066 |

### 3.4 해석

#### Reference offload만 C 시리즈 전체 peak를 낮춘다

C01은 전체 allocated peak를 11.065 GiB에서 10.149 GiB로 0.916 GiB 낮춘다. Reference가 사용되지 않는 Rollout과 Actor log-prob에서도 약 0.90–0.92 GiB가 줄어든다. Reference log-prob에서는 계산을 위해 Reference를 다시 적재해야 하므로 peak가 C00과 같은 7.021 GiB다.

비용은 특정 phase에 집중된다. Reference log-prob 시간이 0.311 s에서 0.568 s로 증가하고 나머지 phase는 거의 변하지 않는다. 따라서 Reference offload는 원인과 비용이 명확한 memory–time trade-off다.

#### Optimizer-state offload는 비-update phase 메모리는 줄이지만 전체 peak는 못 줄인다

C02의 Rollout peak는 7.409→3.728 GiB, Actor log-prob은 7.687→4.007 GiB, Reference log-prob은 7.021→3.340 GiB로 크게 감소한다. 그러나 GPU AdamW를 수행하려면 optimizer state를 Actor update 때 다시 GPU로 가져와야 하므로 Actor update peak는 11.064 GiB로 그대로다. 따라서 전체 peak도 줄지 않는다.

Actor update 시간은 1.452→1.857 s, Rollout은 4.562→4.987 s로 증가한다. 즉 현재 구성은 state 전송 및 residency transition 비용을 지불하면서 최대 메모리를 결정하는 phase는 해결하지 못한다.

#### Activation offload는 현재 workload의 측정 peak를 줄이지 못한다

C03은 모든 phase의 peak가 C00과 사실상 동일하지만 Actor update 시간은 1.452→1.923 s로 증가한다. Saved-tensor offload는 activation이 생성된 이후 CPU로 복사하고 backward에서 다시 가져오는 방식이다. 계산 중인 layer의 activation, gradient, parameter, logits, workspace는 여전히 GPU에 동시에 존재한다. 전송 도중에는 원본과 복사본 수명이 겹칠 수도 있다.

따라서 saved activation offload가 Actor F/B working set 전체를 제거하는 것은 아니다. 이 결과는 activation offload가 항상 무용하다는 뜻이 아니라, 짧은 sequence와 micro-batch 1인 현재 workload에서는 peak 절감 없이 시간 비용만 발생했다는 뜻이다.

## 4. G 시리즈: Actor-resident 역방향 residency ablation

최종 G는 원래의 **Actor-resident baseline**을 사용한다. G00–G03 모두 Actor 파라미터를 GPU에 유지하므로, Actor 상주 여부를 별도의 변수로 추가하지 않고 Reference, optimizer, activation만 비교한다. AdamW는 모든 구성에서 GPU에서 수행한다.

G00은 Reference를 phase-offload하고 optimizer state를 swap하며 saved activation을 offload한다. G01–G03은 G00에서 객체 하나만 추가로 GPU에 상주시킨다.

| ID | Actor | Reference | Optimizer state | Saved activation | G00 대비 GPU에 추가 상주하는 객체 |
|---|---|---|---|---|---|
| G00 | GPU 상주 | phase-offload | 사용하지 않을 때 CPU | CPU-backed | 없음; Actor-resident base |
| G01 | GPU 상주 | GPU 상주 | 사용하지 않을 때 CPU | CPU-backed | Reference parameter |
| G02 | GPU 상주 | phase-offload | GPU 상주 | CPU-backed | optimizer state |
| G03 | GPU 상주 | phase-offload | 사용하지 않을 때 CPU | GPU | saved activation |

### 4.1 End-to-end 결과

| ID | Step time (s) | Throughput (tokens/s/GPU) | 전체 GPU peak (GiB) | GPU device peak (GiB) | CPU RSS peak (GiB) | G00 대비 |
|---|---:|---:|---:|---:|---:|---|
| G00 | 7.985 | 84.14 | 10.149 | 9.546 | 7.017 | 기준 |
| G01 | 7.872 | 85.31 | 11.066 | 10.497 | 5.763 | step −1.4%; GPU +0.917 GiB |
| G02 | 7.330 | 91.20 | 10.149 | 10.292 | 3.018 | step −8.2%; allocated peak 증가 없음 |
| G03 | 7.584 | 88.63 | 10.149 | 9.544 | 6.486 | step −5.0%; allocated peak 증가 없음 |

### 4.2 Phase 실행시간

| 구성 | Rollout (s) | Actor log-prob (s) | Reference log-prob (s) | Actor update (s) |
|---|---:|---:|---:|---:|
| G00 | 4.948 | 0.314 | 0.559 | 2.155 |
| G01 | 4.922 | 0.322 | 0.321 | 2.298 |
| G02 | 4.529 | 0.326 | 0.581 | 1.885 |
| G03 | 4.856 | 0.322 | 0.571 | 1.824 |

### 4.3 Phase별 peak allocated GPU memory

| 구성 | Rollout (GiB) | Actor log-prob (GiB) | Reference log-prob (GiB) | Actor update (GiB) |
|---|---:|---:|---:|---:|
| G00 | 2.826 | 3.086 | 3.340 | 10.149 |
| G01 | 3.728 | 4.007 | 3.340 | 11.066 |
| G02 | 6.507 | 6.767 | 7.021 | 10.149 |
| G03 | 2.826 | 3.086 | 3.340 | 10.149 |

### 4.4 해석

#### Reference 상주는 작은 성능 이득에 비해 peak 비용이 직접적이다

G01은 Reference 전송을 제거해 Reference log-prob을 0.559→0.321 s로 줄인다. 그러나 Actor update가 0.143 s 느려져 end-to-end 개선은 1.4%에 그친다. 반면 전체 peak는 0.917 GiB 증가해 11.066 GiB가 된다.

Phase 메모리 패턴은 C01의 정확한 역방향이다. Reference 상주는 Rollout, Actor log-prob, Actor update에 약 0.90–0.92 GiB를 추가한다. 따라서 C01과 G01은 독립적으로 같은 결론을 준다. 메모리가 부족할 때 가장 먼저 offload할 persistent object는 Reference parameter다.

#### 메모리 여유가 있으면 optimizer 상주가 가장 큰 속도 개선을 준다

G02는 G 중 가장 빠르다. G00보다 step time은 8.2% 짧고 throughput은 약 8.4% 높다. 주요 개선은 Rollout과 Actor update에서 나타나며, optimizer load/offload transition과 GPU AdamW 직전 state swap을 제거한 효과와 일치한다.

Optimizer 상주는 비-update phase allocation을 약 3.68 GiB 높인다. 이는 두 FP32 Adam moment buffer의 크기와 일치한다. 그런데 전체 allocated peak는 증가하지 않는다. G00도 Actor update에서는 optimizer state를 GPU에 적재해야 하므로 이미 같은 크기를 peak에서 지불하기 때문이다. 즉 residency가 길어져 step 전반 메모리는 증가해도, 기존 peak에서 동일 객체가 필요했다면 최대값은 증가하지 않을 수 있다.

#### Activation은 GPU에 두는 편이 빠르고 측정 peak도 증가하지 않는다

G03은 G00보다 step time이 5.0% 짧다. 특히 Actor update가 2.155→1.824 s로 감소한다. Phase peak는 G00과 사실상 동일하다. 이는 C03과 완전히 일관된 결과다. Activation offload는 pack/copy/unpack 비용을 만들지만 현재 workload의 phase maximum은 줄이지 못한다.

따라서 현재 조건에서는 saved activation을 GPU에 유지하는 것이 합리적이다. 단, sequence가 길거나 micro-batch가 커지면 saved activation 비중이 커질 수 있으므로 그 조건에서는 다시 검증해야 한다.

## 5. M 시리즈: CPU AdamW와 새로 드러난 Actor F/B 병목

M은 optimizer-state residency만 바꾸는 실험이 아니다. M00과 M01은 FSDP flat Actor parameter와 gradient를 CPU로 옮기고 일반 PyTorch AdamW를 CPU에서 실행한다. 갱신된 Actor parameter는 다시 GPU로 복사된다. M02는 Actor-resident GPU-Adam matched control이다.

발표에서 사용할 정확한 이름은 다음과 같다.

- **M00 — CPU Adam / Partial actor reuse:** Rollout→Actor log-prob 구간에서만 Actor를 유지하고 나머지 phase에서는 offload하며 CPU AdamW 실행
- **M01 — CPU Adam / Actor resident:** RL phase 사이에는 Actor를 GPU에 유지하되 CPU AdamW 순간에는 구현상 CPU로 이동
- **M02 — GPU Adam / Actor resident:** Actor를 GPU에 유지하고 optimizer state를 swap하여 GPU AdamW 실행

| ID | RL phase 사이 Actor residency | Reference | AdamW 장치 | Optimizer state | Saved activation |
|---|---|---|---|---|---|
| M00 | 부분 재사용 후 phase-offload | phase-offload | CPU | CPU | CPU-backed |
| M01 | CPU optimizer 순간 외 GPU 상주 | phase-offload | CPU | CPU | CPU-backed |
| M02 | GPU 상주 | phase-offload | GPU | 사용하지 않을 때 CPU | CPU-backed |

### 5.1 End-to-end 결과

| ID | Step time (s) | Throughput (tokens/s/GPU) | 전체 GPU peak (GiB) | GPU device peak (GiB) | CPU RSS peak (GiB) |
|---|---:|---:|---:|---:|---:|
| M00 | 12.979 | 51.76 | 6.469 | 8.972 | 8.667 |
| M01 | 12.307 | 54.26 | 6.468 | 8.359 | 8.670 |
| M02 | 8.226 | 81.73 | 10.151 | 10.816 | 7.010 |

M02 대비 M00은 peak allocation을 3.682 GiB(36.3%) 줄이지만 step time은 57.8% 증가한다. M01은 peak를 3.683 GiB(36.3%) 줄이고 step time은 49.6% 증가한다. M01은 M00과 peak가 사실상 같으면서 5.2% 빠르다.

### 5.2 Phase 실행시간

| 구성 | Rollout (s) | Actor log-prob (s) | Reference log-prob (s) | Actor update (s) |
|---|---:|---:|---:|---:|
| M00 | 4.763 | 0.487 | 0.571 | 7.145 |
| M01 | 4.583 | 0.314 | 0.575 | 6.823 |
| M02 | 4.904 | 0.325 | 0.583 | 2.405 |

### 5.3 Phase별 peak allocated GPU memory

| 구성 | Rollout (GiB) | Actor log-prob (GiB) | Reference log-prob (GiB) | Actor update (GiB) |
|---|---:|---:|---:|---:|
| M00 | 2.823 | 3.085 | 1.499 | 6.469 |
| M01 | 2.810 | 3.086 | 3.340 | 6.468 |
| M02 | 2.826 | 3.086 | 3.340 | 10.151 |

### 5.4 CPU AdamW가 실제로 제거한 메모리

M00 step 10의 detail instrumentation에는 다음 placement 전이가 기록되어 있다.

| Snapshot | CUDA allocated (GiB) | Actor parameter | Gradient | CPU Adam states |
|---|---:|---|---|---|
| Actor backward 종료 | 3.697 | GPU 1.840 GiB | GPU 1.840 GiB | CPU 3.681 GiB |
| parameter/gradient D2H 후 | 0.016 | CPU 1.840 GiB | CPU 1.840 GiB | CPU 3.681 GiB |
| CPU AdamW 후 | 0.016 | CPU 1.840 GiB | CPU 1.840 GiB | CPU 3.681 GiB |
| Actor GPU 재적재 후 | 1.856 | GPU 1.840 GiB | 해제 | CPU 3.681 GiB |

따라서 남은 6.47 GiB phase peak의 원인은 CPU AdamW가 아니다. 실제 CPU optimizer 계산 중 live CUDA tensor allocation은 약 16 MiB까지 내려간다.

### 5.5 Actor-update peak가 6.47 GiB로 남는 이유

그래프의 `Actor update` 막대는 update phase 전체의 최댓값이다. GPU AdamW pressure를 제거하면 peak 발생 시점이 더 앞의 Actor forward/backward로 이동한다. F/B에서는 다음 객체가 동시에 필요하다.

- Actor parameter
- backward에서 생성되는 gradient
- autograd에 필요한 현재 activation과 saved activation
- logits, loss tensor, attention workspace 및 연산 임시 buffer

CPU AdamW는 optimizer의 위치와 실행 장치만 바꾼다. Actor F/B를 계산하는 데 필요한 working set은 제거하지 못한다. 즉 M00/M01은 optimizer-state peak를 해결하고, 그 아래에 가려져 있던 Actor F/B 병목을 드러낸다.

### 5.6 현재 CPU AdamW가 느린 이유

현재 구현은 전체 모델 단위의 동기식 경로다.

1. GPU에서 Actor backward 전체 완료
2. FSDP flat Actor parameter storage 전체를 CPU로 이동
3. GPU synchronize
4. 모든 gradient를 CPU로 동기 전송
5. 일반 PyTorch AdamW를 CPU에서 실행
6. CPU gradient 해제
7. 갱신된 Actor parameter 전체를 GPU로 재적재

Backward 중 gradient streaming, bucket-level overlap, DeepSpeed식 최적화 CPU Adam kernel, double buffering, one-step-delayed update가 없다. 따라서 이 성능 손실은 **naive synchronous CPU-Adam endpoint**의 비용이며 최적화된 ZeRO-Offload의 성능 상한으로 해석하면 안 된다.

M01이 M00보다 빠른 이유는 주변 RL phase 사이에서 불필요한 Actor swap을 제거하기 때문이다. 그러나 현재 CPU optimizer step 안에서 발생하는 필수 parameter D2H/H2D 왕복은 그대로 남는다.

## 6. C, G, M 통합 해석

### 6.1 C와 G는 같은 정책을 양방향으로 검증한다

| 객체 | C: all-GPU에서 offload | G: Actor-resident offload base에서 상주 | 통합 결론 |
|---|---|---|---|
| Reference parameter | 전체 0.916 GiB 절감, step +4.0% | 0.917 GiB 증가, 속도 개선은 1.4% | persistent object 중 가장 먼저 offload |
| Optimizer state | 전체 peak 감소 없음, step +12.8% | 전체 peak 동일, step 8.2% 개선 | 여유 메모리가 있으면 상주; offload는 phase residency는 낮추지만 GPU-Adam peak는 못 낮춤 |
| Saved activation | peak 감소 없음, step +7.4% | peak 증가 없이 step 5.0% 개선 | 현재 workload에서는 GPU 유지 |

Optimizer 결과는 모순이 아니다. Offload하면 Rollout과 log-prob phase의 메모리는 크게 줄지만 GPU AdamW 시점에는 state가 다시 필요하므로 전체 maximum은 줄지 않는다. 반대로 상주시키면 이미 optimizer가 필요했던 기존 peak는 증가시키지 않으면서 전송만 제거할 수 있다.

### 6.2 M은 메모리 병목의 종류를 바꾼다

C와 G는 GPU AdamW를 유지한다. 이때 전체 peak는 약 10.15–11.07 GiB이며 optimizer state가 존재하는 Actor update가 지배한다. M00/M01은 AdamW 계산과 state를 CPU로 옮겨 peak를 약 6.47 GiB로 낮춘다. 이후 지배 객체는 GPU optimizer working set에서 Actor F/B working set으로 바뀐다.

따라서 병목은 두 단계로 설명할 수 있다.

1. **GPU-Adam regime:** optimizer state와 AdamW temporary가 Actor-update 메모리를 지배한다.
2. **CPU-Adam regime:** optimizer pressure가 사라지고 Actor parameter + gradient + activation + transient F/B memory가 peak가 된다.

### 6.3 현재 workload의 권장 정적 정책

1. **Actor parameter는 RL phase 사이 GPU에 유지한다.** Rollout, Actor log-prob, Actor update에서 반복 사용하며 M01이 같은 CPU-Adam peak에서 M00보다 빠르다.
2. **메모리가 부족하면 Reference parameter를 먼저 offload한다.** Read-only이고 한 phase에서만 사용하며, 시간 비용 대비 가장 명확한 전체 peak 절감을 제공한다.
3. **현재 workload에서는 saved activation을 GPU에 유지한다.** Offload가 측정 peak를 낮추지 못하고 Actor update만 느리게 한다.
4. **메모리가 충분하면 optimizer-state GPU residency를 우선한다.** G02가 G 중 가장 빠르며 G00 대비 전체 peak도 증가하지 않는다. 단, 비-update phase 상시 점유량은 증가한다.
5. **약 3.68 GiB peak 절감이 반드시 필요할 때만 CPU AdamW를 사용한다.** 측정된 CPU-Adam 구성 중에는 M01이 M00보다 실용적이다.

정책 우선순위는 다음과 같다.

> Actor 상주 → activation 상주 → 여유가 있을 때 optimizer 상주 → Reference부터 offload

이 순서는 현재 workload에 한정된다. Sequence 또는 micro-batch가 커지면 activation 메모리 증가로 정책 순서가 바뀔 수 있다.

## 7. 한계와 피해야 할 주장

- 결과는 약 11.9 GiB 단일 GPU, FSDP1, 0.5B model 조건이다. Multi-GPU scaling 결과로 일반화할 수 없다.
- M00/M01은 일반 PyTorch CPU AdamW를 동기식으로 사용한다. 이 실행시간을 모든 CPU optimizer 또는 완성된 ZeRO-Offload의 본질적 비용이라고 주장하면 안 된다.
- Activation 결론은 현재 prompt/response 길이와 micro-batch에만 적용된다. Long-context 조건에는 별도 sweep이 필요하다.
- 서로 다른 시리즈는 별도 batch에서 실행되어 절대시간 drift가 있을 수 있다. 인과 비교는 시리즈 내부 matched configuration을 우선해야 한다.
- CUDA `allocated`, CUDA `reserved`, driver-visible device usage는 서로 다른 지표다. CPU AdamW 중 device-used가 높다고 Actor tensor가 GPU에 남아 있다는 뜻은 아니다.
- 최종 G는 제공된 최종 그래프와 일치하는 `offload-gpu-adam-residency-v1-performance`의 Actor-resident G00–G03이다. 이후 CPU-residency로 재설계한 G와 섞어 해석하면 안 된다.

## 8. 최종 결론

1. Offload 기능을 더 많이 켠다고 항상 이득이 커지는 것은 아니다. 전송 비용은 증가하면서 전체 peak를 결정하는 phase는 그대로일 수 있다.
2. Reference parameter는 가장 강한 첫 offload 후보이다. C01과 G01 모두 Reference 상주의 메모리 비용이 약 0.92 GiB이며 성능 이득은 제한적임을 보여준다.
3. 메모리가 허용하면 optimizer-state residency는 유효한 성능 최적화다. G에서 가장 큰 속도 개선을 제공하지만, optimizer-state offload만으로 GPU AdamW의 전체 peak를 낮출 수는 없다.
4. 현재 workload에서 saved-activation offload는 역효과다. 측정 peak를 줄이지 못하면서 Actor-update overhead를 증가시킨다.
5. AdamW 계산을 CPU로 옮기는 방법만 GPU optimizer working set을 제거하여 peak를 약 10.15 GiB에서 6.47 GiB로 낮췄다.
6. GPU AdamW를 제거하면 Actor forward/backward가 새로운 병목이 된다. 추가 절감에는 activation recomputation, micro-batch 축소, gradient streaming, layer/bucket 단위 parameter scheduling 같은 F/B 전용 기법이 필요하다.
7. 현재 CPU-Adam 구현은 동기 전송과 CPU 계산 비용이 크다. 다음 시스템 연구 방향은 Actor의 phase 간 residency를 보존하면서 gradient D2H, CPU update, updated-parameter H2D를 bucket화하고 서로 overlap하는 것이다.

## 9. 최종 데이터 출처

- C: `outputs/offload-fullft-v5-performance` — C00–C03
- G: `outputs/offload-gpu-adam-residency-v1-performance` — 제공된 최종 그래프의 Actor-resident G00–G03
- M: `outputs/offload-residency-v2-performance` — M00–M02
- M 세부 placement 근거: `outputs/offload-residency-v1-detail/M00_phase_min-r1/events/memory-actor_update_detail-rank0-pid1232276.jsonl`

최종 C00–C03, G00–G03, M00–M02의 모든 구성은 성공한 독립 실행 3회의 평균을 사용했다.

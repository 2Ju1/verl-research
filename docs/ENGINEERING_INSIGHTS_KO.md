# Phase-aware offloading 실험에서 얻은 인사이트와 디버깅 기록

이 문서는 phase-aware model offload, CPU AdamW, backward gradient streaming을
구현하고 측정하는 과정에서 얻은 기술적 교훈을 정리한다. 현재 대화에 남아 있는
작업 맥락, Git 이력, 보존된 결과와 구현 코드를 교차 확인했다. 별도의 외부 대화
캐시를 직접 조회한 기록은 아니므로, 수치가 남지 않은 시행착오는 정성적 사실로만
기록하고 추정과 확정 결론을 구분한다.

## 1. 문제를 바라보는 방식이 어떻게 바뀌었는가

처음 질문은 단순히 “RL 학습의 GPU 메모리를 줄일 수 있는가”였다. 실험을 거치며
이 질문은 다음 세 문제로 분해되었다.

1. 각 phase에서 어떤 memory object가 실제 peak를 만드는가?
2. 불필요한 residency를 제거한 뒤 새로 나타나는 peak는 무엇인가?
3. 메모리 이동을 연산과 겹쳤을 때 critical path가 어디로 이동하는가?

핵심 교훈은 전체 step의 최대값 하나만 보는 것으로는 부족하다는 것이다. Rollout,
Actor log-prob, Reference log-prob, Actor forward, Actor backward, Update는 서로
다른 tensor lifetime을 가진다. 최적화 하나를 적용하면 병목이 사라지는 것이 아니라
다른 phase로 이동한다.

## 2. 확정된 결과

### 2.1 Phase offload

All-on-GPU 대비 phase-exclusive Actor/Reference placement는 앞쪽 phase의 불필요한
parameter residency를 크게 제거했다. Actor backward peak는 10.24 GiB에서
4.72 GiB로 감소했다. 반면 GPU AdamW Update는 10.24 GiB에서 8.40 GiB로만
줄었다. 감소분은 Update에 필요하지 않은 Reference parameter에 해당한다.

따라서 phase offload만으로는 Update 중 동시에 필요한 Actor parameter, gradient,
optimizer state를 제거할 수 없다. “모든 phase의 메모리를 조금씩 줄이는 것”보다
전체 step의 최대 peak 원인을 직접 제거해야 더 큰 모델을 실행할 수 있다.

### 2.2 CPU AdamW

phase placement를 고정하고 optimizer 구현만 바꾼 matched comparison에서 Update
peak는 8.408 GiB에서 2.365 GiB로 감소했다. GPU Adam state를 늦게 load하는
방식과 달리 CPU AdamW는 FP32 master parameter, gradient, Adam first/second
moment를 CPU에 유지하고 CPU에서 update한다.

대신 serial CPU AdamW의 Update 시간은 0.129초에서 3.565초로 증가했다. 이는
구현 오류가 아니라 GPU optimizer state residency를 제거하기 위해 CPU 연산과
parameter reload 비용을 지불한 기본 trade-off다.

### 2.3 Gradient streaming과 pipelined Update

16 MiB bucket, 3 staging slots, asynchronous D2H, early gradient release,
CPU gradient accumulation, Adam–H2D overlap을 함께 사용한 결과는 다음과 같다.

| metric | No-stream | 16 MiB streaming |
|---|---:|---:|
| Actor backward peak | 4.721 GiB | 3.402 GiB |
| Actor backward time | 0.212 s | 0.292 s |
| Update time | 3.551 s | 3.020 s |
| Step time | 9.814 s | 8.629 s |

Backward는 약 80 ms 느려졌지만, Update가 약 531 ms 줄고 전체 step은
1.186초(12.1%) 감소했다. streaming의 목적은 단순히 D2H를 빠르게 만드는 것이
아니다. gradient lifetime을 줄이는 동시에 이후 CPU Adam과 parameter H2D를
bucket 단위 pipeline으로 바꾸는 데 의미가 있다.

### 2.4 1.5B capacity boundary

11.90 GiB TITAN Xp에서 Qwen2.5-1.5B FP32 full-parameter workload는 All-GPU와
CPU AdamW no-stream 모두 OOM이었다. CPU AdamW만으로 optimizer state는 GPU에서
제거되지만 backward에서 full gradient가 누적되기 때문이다. CPU AdamW와
gradient streaming을 함께 사용했을 때만 학습이 완료됐고 phase-local peak는
8.44 GiB였다.

이 결과는 두 최적화가 같은 메모리를 중복해서 줄이는 것이 아님을 보여준다.

- CPU AdamW: Update의 optimizer-state residency 제거
- Gradient streaming: Backward의 full-gradient residency 제한

## 3. 주요 오류와 디버깅 과정

### 3.1 cumulative peak를 phase peak로 잘못 해석한 문제

초기 그래프에서는 이전 phase에서 발생한 CUDA peak가 다음 phase에도 남아 있었다.
그 결과 Update memory가 실제 optimizer-only peak처럼 보이지 않거나, gradient를
해제했는데도 Update bar가 줄지 않는 모순이 생겼다.

수정 원칙은 다음과 같다.

- 각 phase 시작 전에 `torch.cuda.reset_peak_memory_stats()`를 호출한다.
- phase 종료 시점의 allocated와 그 phase 내부 peak allocated를 분리한다.
- `reserved`, `nvidia-smi` device-used, PyTorch `allocated`를 혼용하지 않는다.
- memory probe와 performance run을 분리한다.

결론: “Update가 줄지 않았다”는 일부 과거 결과는 최적화 실패가 아니라 measurement
boundary 오류였다.

### 3.2 gradient 해제 시점과 Update peak

CPU optimizer가 gradient를 소비한 뒤에도 GPU parameter의 `.grad` 참조가 남으면
Update 측정에 gradient가 포함된다. 그래서 gradient 해제를 Update 전에 수행하고,
CUDA D2H 완료가 보장된 뒤 source gradient를 제거하도록 경계를 수정했다.

여기서 순서가 중요하다. 비동기 D2H가 끝나기 전에 `.grad`를 해제하면 correctness
문제가 생길 수 있다. 반대로 모든 복사를 동기화한 뒤 한꺼번에 해제하면 안전하지만
streaming의 memory 이점과 overlap을 잃는다. CUDA event로 staging slot의 완료를
추적하고 완료된 source만 해제하는 방식이 필요했다.

### 3.3 `Late-load GPU Adam`의 의미

이 표현은 GPU AdamW 상태를 학습 내내 GPU에 두는 것이 아니라 Update 직전에
GPU로 가져와 update한 뒤 다시 내리는 비교군을 뜻한다. CPU AdamW와 동일하지 않다.

- Late-load GPU AdamW: optimizer 연산은 GPU, Update 시 state가 GPU에 존재
- CPU AdamW: optimizer 연산과 state가 CPU, GPU에는 updated parameter만 reload

따라서 둘의 phase placement가 비슷해 보여도 Update peak와 시간의 의미는 다르다.

### 3.4 residual Optimize timer를 직접 timer처럼 사용한 문제

일부 과거 그래프는 `actor update - forward - backward`로 Optimize 시간을
역산했다. 이 residual에는 offload, synchronization, cleanup, parameter movement가
섞일 수 있어 Adam update 자체와 같지 않다. 같은 설정인데 3.56초, 4.26초,
4.77초처럼 서로 다른 값이 보였던 주요 이유 중 하나였다.

이후에는 다음 direct metric만 사용했다.

- `perf/actor_backward_total_wall_s`
- `perf/actor_adam_step_total_wall_s`

결론: phase label이 같아도 timer boundary가 다르면 같은 수치로 비교하면 안 된다.

### 3.5 streaming을 켰더니 Rollout memory가 증가한 문제

Gradient streaming은 backward에만 필요하므로 Rollout memory가 증가해서는 안
된다는 것이 정상적인 기대였다. 실제 증가가 관측되어 lifetime을 추적한 결과,
packing/staging buffer가 engine 초기화 시점에 미리 할당되어 앞 phase에도 상주하는
문제였다.

수정은 packing buffer와 관련 storage를 최초 backward 사용 시점에 lazy allocation
하도록 바꾸는 것이었다. 이 사례는 “기능상 backward 전용”이라는 설계 의도만으로
실제 allocator lifetime이 보장되지 않는다는 점을 보여준다.

### 3.6 backward hook이 하는 일

여기서 gradient hook은 gradient 값을 바꾸기 위한 callback이 아니라, 각 parameter의
gradient accumulation이 끝난 순간을 감지하는 callback이다. 구현은
`register_post_accumulate_grad_hook`을 사용한다.

hook이 수행하는 작업은 다음과 같다.

1. 해당 parameter gradient가 ready임을 기록한다.
2. 같은 bucket의 gradient가 모두 준비되면 packing을 시작한다.
3. reusable GPU staging slot에 contiguous하게 복사한다.
4. 전용 CUDA stream에 pinned CPU buffer로 D2H를 enqueue한다.
5. CUDA event로 완료를 추적한다.
6. 안전한 시점에 원본 GPU gradient를 해제한다.

hook 자체, Python dispatch, packing kernel, event 기록과 allocator 작업이 backward
critical path에 들어갈 수 있다. 따라서 streaming이 CPU에서 진행된다는 사실만으로
GPU backward 시간이 완전히 같을 것이라고 기대하면 안 된다.

### 3.7 backward가 느려진 원인과 backpressure

staging slot 수는 유한하다. GPU가 다음 완성 bucket을 생산했는데 모든 slot이 D2H
중이면 producer인 backward가 가장 오래된 slot의 완료를 기다린다. 이것이 bounded
backpressure다. slot 수를 무제한으로 늘리면 기다림은 줄 수 있지만 pinned CPU와
GPU staging memory가 증가해 bounded-memory 설계가 무너진다.

관측된 근거는 대략 다음과 같다.

- staging-slot backpressure: 약 41 ms/step
- 겹쳐 실행된 D2H CUDA activity 합계: 약 154 ms
- 최종 backward 증가: 약 80 ms

D2H activity의 합계를 backward 증가와 직접 더하면 안 된다. 상당 부분이 다른 작업과
겹친다. 남은 차이는 hook dispatch, packing, event enqueue, early release, GPU memory
bandwidth contention과 일치하지만, matching streaming Nsight trace가 완성되지 않아
각 항목의 정확한 비율까지 확정하지는 않았다.

### 3.8 2-slot과 3-slot

2-slot 검증은 backpressure 가설을 빠르게 확인하기 위한 실험이었다. slot이 적을수록
bounded memory는 작지만 producer stall 가능성이 커진다. 16 MiB layout에서는 3-slot이
memory 증가를 제한하면서 pipeline progress를 개선해 채택됐다.

중요한 제한은 3-slot이 보편적인 최적값이 아니라는 점이다. bucket size, PCIe 속도,
GPU compute 속도, parameter shape에 따라 최적 slot 수는 달라진다. “3-slot 최적”이
아니라 “이 하드웨어와 16 MiB 설정에서 선택된 값”이라고 해석해야 한다.

### 3.9 bucket size sweep의 해석 한계

0.5B에서 16, 32, 64, 128, 256, 512 MiB를 각각 3회 측정했다. 작은 bucket은
gradient를 일찍 보낼 수 있지만 hook/packing/event 횟수가 늘고, 큰 bucket은 overhead가
줄지만 gradient release가 늦어지고 burst가 커진다.

그러나 보존된 sweep은 `overlap_h2d_with_cpu_update=false`로 수행됐다. 따라서 이
sweep으로 optimized pipeline 전체의 최적 bucket을 단정할 수 없다. 특히 3-slot이
128 MiB에 맞춘 값인지 묻는 문제도 bucket과 slot을 독립 변수로 함께 sweep하지
않으면 답할 수 없다.

### 3.10 “streaming인데 Update가 더 느림”이라는 모순

한 그래프에서는 no-stream Update 3.55초보다 16 MiB streaming Update 3.68초가
느렸다. 처음에는 노이즈, CPU Adam 구현 차이, no-stream에만 적용된 최적화 등을
의심했다. direct timer로 재측정하고 configuration을 대조한 결과 핵심 차이는
`overlap_h2d_with_cpu_update`였다.

통제 실험의 구조는 다음과 같았다.

- A: no-stream, serial CPU Adam + H2D
- B: 16 MiB gradient stream, serial CPU Adam + H2D
- C: 16 MiB gradient stream, bucket CPU Adam + H2D overlap

telemetry-on 진단값에서 Update는 각각 3.579, 3.716, 3.407초였다. B는 gradient를
미리 보냈어도 parameter reload가 serial이라 Update가 줄지 않았다. C에서만 CPU
Adam 완료 bucket의 parameter H2D를 다음 bucket CPU update와 겹쳤다. telemetry-off
3회 결과에서는 optimized Update가 3.020초까지 감소했다.

결론: gradient D2H overlap과 updated-parameter H2D overlap은 서로 다른 pipeline
구간이다. 앞의 것만 켜고 뒤의 것을 끄면 Update 단축을 기대할 수 없다.

## 4. 실험 설계에서 얻은 일반 원칙

### 4.1 한 번에 하나의 변수만 바꾼다

phase placement, CPU Adam, gradient streaming, bucket size, slot 수, telemetry,
foreach, H2D overlap을 한 matrix에서 동시에 바꾸면 원인을 식별할 수 없다. 최종적으로
설득력이 생긴 비교는 matched A/B 또는 A/B/C였다.

### 4.2 memory run과 performance run을 분리한다

`--detail`, JSON event logging, 강제 synchronize, CUDA memory history, Nsight는
모두 시간을 교란한다. 메모리/원인 분석 run의 wall time을 성능 bar에 쓰지 않고,
telemetry-off 반복 run의 direct timer를 사용해야 한다.

### 4.3 OOM은 실패 샘플이 아니라 capacity evidence가 될 수 있다

OOM run은 평균 시간에 포함하면 안 된다. 하지만 “이 구성이 GPU에 들어가는가”가
질문이면 return code, CUDA OOM 로그, 당시 allocated/capacity는 유효한 증거다.

### 4.4 평균값만으로 pipeline 원인을 설명하지 않는다

Update가 줄었다는 bar만으로 overlap이 원인이라고 단정할 수 없다. configuration flag,
bucket event 순서, CPU Adam completion, H2D enqueue/completion, backpressure event가
예상 순서로 나타나는지 함께 보여야 한다.

### 4.5 allocator 지표의 의미를 명시한다

- allocated: live tensor가 차지하는 PyTorch memory
- reserved: allocator가 재사용을 위해 확보한 pool
- device-used: CUDA context와 외부 library까지 포함한 driver 관점

이 세 지표는 동일하지 않다. tensor lifetime 최적화의 주장은 allocated peak로,
실제 카드에 들어가는지에 관한 capacity 주장은 OOM과 device capacity로 설명한다.

## 5. 폐기하거나 제한적으로만 사용하는 해석

- cumulative peak 기반의 phase graph는 사용하지 않는다.
- residual로 역산한 Optimize time은 direct Adam timer와 섞지 않는다.
- telemetry/Nsight run의 wall time은 performance 결과로 사용하지 않는다.
- overlap이 꺼진 bucket sweep으로 optimized pipeline의 최적 bucket을 단정하지 않는다.
- 3-slot을 다른 bucket, GPU, PCIe 환경에도 보편적인 최적값이라고 주장하지 않는다.
- D2H activity 합계만으로 backward 증가 전체를 설명하지 않는다.
- CPU AdamW만으로 1.5B가 가능하다고 주장하지 않는다. 실제 병목은 backward gradient
  accumulation으로 이동했다.

## 6. 남아 있는 미해결 질문

1. 동일한 환경에서 no-stream과 streaming의 matching Nsight trace를 모두 확보해
   hook, packing, copy-engine contention, allocator 비용을 정량 분해할 필요가 있다.
2. Adam–H2D overlap을 켠 상태에서 bucket size × slot count의 factorial sweep이
   필요하다.
3. TITAN Xp 외 GPU와 PCIe 세대에서 16 MiB/3-slot 선택이 어떻게 변하는지 확인해야 한다.
4. 더 긴 sequence, 다른 micro-batch, 다른 model size에서도 bottleneck 이동이 같은지
   검증해야 한다.
5. CPU NUMA placement, pinned-memory budget, optimizer vectorization/foreach가 CPU Adam
   throughput에 미치는 영향을 별도로 분리해야 한다.

## 7. 현재 보존된 근거

- 결과와 수치: `results/README.md`
- 핵심 그림: `results/figures/`
- figure-to-data mapping: `results/manifest.csv`
- 채택 run의 설정과 측정값: `results/data/`
- 실험 및 재현 규칙: `docs/EXPERIMENTS.md`
- gradient streaming 구현:
  `src/verl/verl/workers/actor/single_gpu_offload/streaming.py`
- bucket CPU Adam/H2D pipeline:
  `src/verl/verl/workers/actor/single_gpu_offload/pipeline.py`
- engine configuration and lifecycle:
  `src/verl/verl/workers/actor/single_gpu_offload/engine.py`

과거의 중간 출력 전체는 정리했으므로, 이 문서는 남아 있는 근거로 확인 가능한 결론과
대화 맥락에서 복원한 시행착오를 구분해 보존하는 역할을 한다.

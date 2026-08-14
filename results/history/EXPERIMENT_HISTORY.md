# Offload 연구 전체 실험·시행착오 기록

## 0. 문서 범위와 증거 수준

이 문서는 `/mnt/sda/juwon/verl-research/outputs` 아래 남아 있는 benchmark run을
전수 조사해 작성했다. 2026-07-27부터 2026-08-13까지 95개 output group,
512개 run이 확인됐다.

함께 생성된 파일은 다음과 같다.

- `all_runs.csv`: 512개 run의 설정, 상태, 시간, 메모리 metric
- `all_runs.json`: 같은 내용을 JSON으로 보존
- `output_groups.csv`: 95개 실험군의 성공/실패/미완료 수와 실행 시각
- `GROUP_CATALOG.md`: 모든 실험군과 개별 run의 표
- 생성기: `results/build_experiment_inventory.py`

이 문서의 판정 표기:

- **채택 결과**: 연구 결과 그래프에 사용한 데이터
- **유효 진단**: 원인 규명에는 유효하지만 보고 성능 수치에는 사용하지 않음
- **폐기/대체**: 설정 오류, 측정 경계 오류, 실패, 또는 더 정확한 후속 실험으로 대체
- **실패가 결과**: OOM이나 기능 실패 자체가 capacity/호환성 결론의 증거

JSON telemetry, `--detail`, `--sync`, Nsight는 실행 시간을 교란할 수 있다.
따라서 원인 분석용 trace와 telemetry-off 성능 수치를 구분한다.

## 1. 초기 offload 조합 탐색: full fine-tuning matrix

### 1.1 `offload-measurement-v2`

- 5 runs: 성공 4, 실패 1
- 초기 계측 체계가 phase, transfer, memory를 실제로 기록하는지 확인한 실험군이다.
- 이후 full matrix의 기반이 되었지만 최종 수치로 사용하지 않았다.

### 1.2 `offload-fullft-v3`, `v4`

- v3: 24 runs, 성공 21, 실패 3
- v4: 15 runs, 성공 12, 실패 3
- actor parameter, reference parameter, optimizer, activation placement 조합을 넓게 탐색했다.
- 실패 run을 포함해 메모리 한계와 잘못된 placement 조합을 확인했다.
- 동일 목적의 후속 v5에서 계측과 구성 정의가 정리되어 v3/v4 수치는 대체됐다.

### 1.3 activation 문제: `activation-fallback-smoke`, `activation-fixed-full`

- fallback smoke: 2/2 성공
- fixed full: 9 runs 중 7 성공, 2 실패
- activation offload가 요청되었지만 실제 handler가 적용되지 않는 fallback 가능성을 확인했다.
- feature가 실제 활성화됐는지 기록하도록 보완한 뒤 full matrix를 재실행했다.
- 단순 config 값만 보고 activation offload 효과라고 주장하면 안 된다는 교훈을 남겼다.

### 1.4 `offload-fullft-v5-smoke`, `v5-detail`, `v5-performance`

- smoke: 5 runs, 성공 4, 실패 1
- detail: 8 runs, 성공 7, 실패 1
- performance: 24 runs, 성공 21, 실패 3
- smoke → detail → performance를 분리했다.
- detail run은 phase-local memory와 전송 원인 확인용, performance run은 반복 성능용이다.
- 이 분리 원칙은 이후 모든 최종 실험에 적용됐다.

## 2. Residency 효과 분리

### 2.1 `offload-residency-*`

- `offload-residency-smoke`: 3/3 성공
- `offload-residency-v1-smoke`: 3/3 성공
- `offload-residency-v1-detail`: 6/6 성공
- `offload-residency-v1-performance`: 18/18 성공
- `offload-residency-v2-smoke`: 3/3 성공
- `offload-residency-v2-performance`: 9/9 성공
- actor/reference/optimizer/activation 중 하나를 GPU에 유지하거나 CPU로 내릴 때의 독립 효과를 분리했다.
- 반복과 detail을 나누면서 초기 matrix의 혼합 효과를 해석 가능한 단위로 바꿨다.

### 2.2 GPU Adam residency 계열

- `offload-gpu-adam-residency-v1-performance`: 18/18 성공
- `v2-performance`: 15 runs 중 12 성공, 3 실패
- `v2-noforeach-performance`: 1 실패
- `v2-performance-final`: 1 실패
- `v3-performance`: 15/15 성공
- optimizer resident/late-load와 foreach 구현 차이를 점검했다.
- no-foreach 단독 재현은 실패했고, v3에서 실행 가능한 비교군을 다시 확정했다.
- 실패한 v2-final/no-foreach 결과는 성능 결론에 사용하지 않는다.

### 2.3 `offload-c04-diagnostics-v1`, `offload-c04-diagnostics-final`

- v1: 10 runs 중 6 성공, 4 실패
- final: 27 runs 중 18 성공, 9 실패
- 특정 actor-parameter/offload 조합(C04)의 memory/transfer 이상을 여러 allocator 및 detail 조건으로 진단했다.
- 성공과 실패가 혼재하므로 개별 run 설정을 보지 않고 평균을 합치면 안 된다.

## 3. Single-GPU ZeRO-style CPU offload engine 진화

### 3.1 Z00–Z03: 기본 구조

- `zero-offload-z00-smoke`: 실패 1
- `zero-offload-z01-z03-smoke`: 3/3 성공
- `zero-offload-z01-z03-smoke-v2`: 1/1 성공
- Z00의 실패 후 CPU master parameter/optimizer와 gradient 수집 경로를 phase별로 분리했다.
- Z01–Z03에서 최소 기능과 checkpoint/precision 경로가 동작하는지 확인했다.

### 3.2 Z03M–Z06: precision, streaming, accumulation, overlap

- `zero-offload-z03m-z04-precision-v1`: 2/2 성공
- `zero-offload-z04-z06-smoke`: 4/4 성공
- `zero-offload-z06-stability-smoke`: 1/1 성공
- Z04: backward hook 기반 bucket D2H
- Z05: CPU gradient accumulation
- Z06: CPU Adam 후 parameter H2D overlap
- smoke 결과는 기능 확인용이며 최종 성능 수치가 아니다.

## 4. Non-FSDP unified engine 탐색

- `non-fsdp-unified-smoke-v1`: 9/9 성공
- `non-fsdp-unified-fp16-smoke-v1`: 1/1 성공
- `non-fsdp-unified-v1`: 48/48 성공
- sync/async, release, bucket, accumulation, pipeline 등 기능 조합을 통합 경로에서 비교했다.
- 48개 run은 구현 옵션의 기능적 유효 범위를 넓게 확인한 데이터다.
- 이 시기의 O-ASYNC/O-BKT/O-GACC/O-PIPE 등은 후속 PA 실험의 후보 설정을 만드는 데 사용됐다.

## 5. Phase-aware(PA) streaming과 allocator 시행착오

### 5.1 초기 async 확인

- `actor-phase-async-smoke`: 1/1 성공
- `pa-async-quick`: 1/1 성공
- `pa-async-fixed2-quick`: 1/1 성공
- `pa-async-fixed2-memory-check`: 1/1 성공
- async D2H, 고정 staging slot, bounded memory가 실제 동작하는지 빠르게 확인했다.

### 5.2 allocator 및 empty-cache 계열

- `pa-allocator-opt-smoke`: 7 runs, 성공 6, 미완료 1
- `pa-allocator-opt-smoke-remain`: 2/2 성공
- `pa-allocator-best-v1`: 4/4 성공
- `pa-allocator-native-confirm`: 1/1 성공
- `pa-part-expand-control`: 1/1 성공
- native allocator, expandable segments, trim/empty-cache, direct buffer 조합을 비교했다.
- allocator 설정은 reserved/device peak에 영향을 주지만 allocated live peak와 동일 개념이 아님을 확인했다.
- 최종 그래프는 allocated peak를 사용하고 allocator 진단의 reserved 수치를 섞지 않았다.

### 5.3 memory sweep와 core diagnostics

- `pa-memory-sweep-v1`: 12/12 성공
- `all-gpu-memory-diagnostic-v1/v2/v3`: 각 1 성공
- `core-memory-diagnostics-v1`: 2 runs 중 1 성공, 1 실패
- bucket/slot 변화에 따른 memory high-water와 all-GPU baseline 측정 경계를 점검했다.
- 이 과정에서 global cumulative peak와 phase-local peak가 혼동될 수 있음을 확인했다.

## 6. FP32 재현과 phase-local peak 교정

### 6.1 `pa-final-v1`

- 15/15 성공
- CPU best, Pareto, streaming 등 후보를 반복 측정한 중간 최종군이다.
- 이후 phase-local probe와 FP32 재현 데이터가 생겨 일부 수치는 대체됐다.

### 6.2 `pa-repro-fp32-v1`

- 48 runs: 성공 42, 실패 6
- FP32 조건에서 placement와 streaming 후보를 재현했다.
- 과거 placement 2×3 진단 그림의 주요 집계 원천이었다. 해당 그림은 핵심 결과
  흐름에서 제외했지만 raw run은 역사 자료로 보존한다.
- 실패 6개는 catalog에서 개별 설정을 확인해야 하며 평균에 포함하지 않는다.

### 6.3 activation/late optimizer probe

- `pa-repro-fp32-late-optimizer-smoke`: 실패 1
- `pa-repro-fp32-late-optimizer-smoke-v2`: 2/2 성공
- `pa-activation-probe-fp32-v1`: 2개 중 1 성공, 1 실패
- `pa-saved-activation-probe-fp32-v1`: 1 성공
- 초기 late optimizer smoke 실패 후 설정을 수정해 GGG/CCC 비교를 확보했다.
- `allgpu_vs_phase_offload_05b`는 성공한 v2 aggregate를 사용한다.

### 6.4 phase-local peak 수정 이력

- `pa-phase-local-peak-v1`: 성공
- `pa-phase-local-peak-lazy-buffer-v1`: 성공
- `pa-phase-local-peak-backward-lazy-v1`: 성공
- `pa-phase-local-peak-nostream-v1`: 성공
- `nostream-v2`: 실패
- `nostream-v3`: 실패
- `nostream-v4`: 성공
- phase 시작 전 peak reset, optimizer 직전 gradient 해제, lazy buffer 시점 등을 반복 교정했다.
- v2/v3 실패 후 v4 결과를 no-stream phase-local 기준으로 사용했다.
- 이 과정 이전의 Update peak가 gradient까지 포함했다면 CPU Adam 효과를 잘못 표현할 수 있다.

## 7. CPU Adam 효과 분리

### 7.1 isolation 실패와 수정

- `pa-cpu-adam-isolation-fp32-v1`: 2/2 실패
- `pa-cpu-adam-isolation-fp32-v2`: 2/2 성공
- 첫 isolation matrix의 구성/호환 문제를 수정한 뒤 GPU Adam과 CPU Adam을 분리 측정했다.

### 7.2 all-phase 및 late GPU Adam

- `pa-cpu-adam-all-phase-fp32-v1`: 2/2 성공
- `pa-late-gpu-adam-30steps-v1`: 1/1 성공
- 여섯 phase 시간과 memory tag를 동일 형식으로 비교했다.
- GPU optimizer load/offload 시간은 Update에 포함해야 하며 단순 `optimizer.step()`만 비교하지 않았다.

### 7.3 phase-best vs CPU AdamW 최종 비교

- `phase-best-vs-cpu-adamw-performance-v1`: 6/6 성공, 각 구성 3회
- `phase-best-vs-cpu-adamw-memory-v1`: 2/2 성공, 각 구성 phase-memory 1회
- 최종 수치:
  - GPU AdamW Update peak 8.408 GiB, Update 0.129 s
  - CPU AdamW Update peak 2.365 GiB, Update 3.565 s
  - backward peak은 양쪽 모두 약 4.72 GiB
- `late-load`는 GPU optimizer state를 항상 resident로 두지 않고 Update 직전에 올리는 구성이다.
- 최종 그래프는 `phase_best_vs_cpu_adamw*.png/pdf`이다.

## 8. Streaming bucket/slot 최적화 시행착오

### 8.1 64/128 MiB 빠른 검사

- `pa-stream64-no-telemetry-quick-v1`: 성공
- `pa-stream64-2slot-no-telemetry-quick-v1`: 성공
- `pa-stream128-direct-nt-v1`: 성공
- telemetry를 끄고 slot 수와 direct CPU gradient buffer 효과를 빠르게 확인했다.

### 8.2 bucket 및 direct-buffer 후보 sweep

- `pa-stream-bucket-sweep-nt-v1`: 4/4 성공
- `pa-stream-direct-final-sweep-v1`: 3/3 성공
- `pa-stream-foreach-test-v1`: 1/1 성공
- `pa-stream-optimized-final-v1`: 3/3 성공
- `pa-stream-optimized-phase-v1`: 1/1 성공
- bucket 크기, staging slot, direct CPU grad, foreach를 순차 비교했다.
- foreach가 항상 이득이라는 가정은 채택하지 않았고 최종 AdamW는 foreach=false 기준으로 고정했다.

### 8.3 16 MiB lazy buffer

- `pa-stream16-lazy-final-v1`: 3/3 성공
- `pa-stream16-lazy-phase-v1`: 1/1 성공
- GPU packing buffer를 필요한 phase에만 활성화해 rollout 등 앞 phase memory 증가를 제거했다.
- 초기 그래프에서 rollout memory가 증가한 문제를 이 lazy allocation으로 교정했다.

### 8.4 16–512 MiB 최적화 sweep

- `pa-optimized-bucket-sweep-v1`: 18/18 성공
- 16, 32, 64, 128, 256, 512 MiB, 각 3회, 3 slots
- 공통: async D2H, early release, reusable packing, direct CPU gradients
- 중요한 제한: `overlap_h2d_with_cpu_update=false`
- 따라서 bucket에 따른 backward/memory trade-off 진단에는 유효하지만 최종 Update 3.02 s와 직접 비교하면 안 된다.

## 9. No-stream vs 16 MiB: 측정 오류와 최종 교정

### 9.1 잘못된 residual 그래프

- 과거 `nostream_vs_16mib_optimized_legacy_residual.*`는
  `timing_s/update_actor - forward - backward`로 Optimize를 역산했다.
- streaming이 작업의 귀속 위치를 바꾸기 때문에 두 구성에서 같은 범위를 비교하지 못했다.
- 4.77→4.26 s라는 막대는 최종 근거에서 제외했다.

### 9.2 직접 optimizer timer 재측정

- `nostream-vs-16mib-direct-remeasure-v1`: 6/6 성공
- no-stream 3회 + streaming direct 3회
- `perf/actor_adam_step_total_wall_s`를 직접 사용했다.
- 당시 streaming matrix는 `overlap_h2d_with_cpu_update=false`였고 Update가 약 3.68 s로 증가했다.
- 이 결과는 “overlap을 켠 최종 streaming” 결과가 아니라 serial H2D 대조군이다.

### 9.3 GPU와 telemetry 조건 통제

- `nostream-vs-16mib-telemetry-once-v1`: 2/2 성공
- `nostream-vs-16mib-telemetry-once-gpu1-v1`: 2/2 성공
- GPU0의 외부 프로세스 점유를 발견한 뒤 GPU1에서 같은 조건으로 재검증했다.
- telemetry 결과에서 no-stream과 streamed CPU Adam/H2D 구간을 분리했다.

### 9.4 기각된 gradient storage 가설

- `stream16-s3-pageable-telemetry-once-v1`: 성공
- `stream16-s3-materialized-telemetry-once-v1`: 실패
- `stream16-s3-materialized-telemetry-once-gpu1-v1`: 성공
- `stream16-s3-fresh-clone-telemetry-once-gpu1-v1`: 성공
- pageable staging, persistent materialization, 매-step fresh clone을 각각 시험했다.
- fresh clone은 약 0.63 s/step 복제 비용을 추가했지만 CPU Adam을 유의미하게 회복시키지 못했다.
- pinned/pageable 또는 “no-stream만 fresh owning gradient”가 주원인이라는 가설을 기각했다.
- 실험용 clone/materialization 옵션은 최종 코드 경로에서 제거했다.

## 10. Adam–H2D overlap 누락 발견과 최종 해결

### 10.1 A/B/C 병렬 기능 검증

- `rootcause-abc-v1`: 3/3 성공
- A: no-stream, serial Adam/H2D
- B: 16 MiB gradient streaming, serial Adam/H2D
- C: 16 MiB gradient streaming, bucket Adam + H2D overlap
- 병렬 실행이라 CPU bandwidth 간섭이 있으므로 기능 확인용으로만 사용했다.

### 10.2 동일 GPU 순차 검증

- `rootcause-abc-sequential-gpu1-v1`: 3/3 성공
- telemetry 직접 Update:
  - A no-stream 3.579 s
  - B stream, overlap off 3.716 s
  - C stream, overlap on 3.407 s
- B→C 개선으로 parameter H2D overlap이 실제로 동작함을 확인했다.
- bucket별 JSON telemetry 자체가 C에 큰 오버헤드를 주므로 최종 성능에는 쓰지 않았다.

### 10.3 telemetry-off 최종 성능

- `stream16-pipeline-performance-once-gpu1-v1`: 1/1 성공
- `stream16-pipeline-performance-3x-gpu1-v1`: 3/3 성공
- 최종 3회 평균:
  - no-stream backward 0.212 s, Update 3.551 s, peak 4.721 GiB
  - 16 MiB overlap backward 0.292 s, Update 3.020 s, peak 3.402 GiB
  - streaming actor update 3.964 s, total step 8.629 s
- 핵심 설정:
  - `bucket_mb=16`
  - `num_staging_buffers=3`
  - `cpu_grad_accumulation=true`
  - `overlap_h2d_with_cpu_update=true`
- 최종 그래프는 `nostream_vs_16mib_optimized.*`이다.

## 11. Backward 증가 원인 분석

- 최종 telemetry-off 차이: 0.212→0.292 s, 약 +80 ms
- 동일 GPU telemetry에서 16 MiB D2H CUDA 합 약 154 ms
- 직접 기록된 staging-slot backpressure 약 41 ms/step
- 나머지는 hook, `_foreach_copy_` packing, CUDA event/stream enqueue, gradient release,
  backward kernel과 D2H의 memory-bandwidth 경쟁으로 좁혔다.
- `backward-nsys-rootcause-v1`: 2 runs 중 no-stream 1 성공, stream profile 1 실패/중단
- no-stream `profile.nsys-rep`는 생성됐지만 stream 대조 trace가 완성되지 않아
  “나머지 39 ms”를 Nsight로 완전 귀속했다는 주장은 아직 할 수 없다.
- 이 실험을 최종 성능 결론에 사용하지 않는다.

## 12. Qwen2.5-1.5B capacity와 bucket sweep

### 12.1 초기/최종 capacity

- `pa-final-qwen15b-best-v1`: 실패 1
- `pa-capacity-fp32-qwen15b-v1`: 5 runs 중 성공 3, 실패 2
- All-GPU와 CPU Adam no-stream은 OOM, CPU Adam + streaming 3회는 성공했다.
- 성공 streaming stdout peak는 약 8.377 GiB이며 결과 그림은 detail/phase peak 8.44 GiB를 사용했다.

### 12.2 no-stream OOM snapshot

- `pa-capacity-fp32-qwen15b-nostream-oom-snapshot-v1`: 실패(OOM)
- `v2`: 실패(OOM), OOM 메시지 보존 성공
- v2 stdout의 CUDA OOM 메시지에는 PyTorch allocated 11.34 GiB,
  GPU total capacity 11.90 GiB가 기록돼 있다.
- 실패가 capacity 결론의 직접 증거다.

### 12.3 1.5B bucket sweep

- `pa-capacity-fp32-qwen15b-bucket-sweep-v1`: 13/13 성공
- 8, 16, 32, 64, 128 MiB 후보를 포함한다.
- 1.5B 성공 가능성과 bucket에 따른 시간/메모리 trade-off를 확인했다.
- 최종 capacity 그림: `qwen15b_fp32_capacity.*`

## 13. 최종 그림별 채택 데이터

1. All GPU vs phase offload: `pa-repro-fp32-late-optimizer-smoke-v2/summary`
2. GPU AdamW vs CPU AdamW: `phase-best-vs-cpu-adamw-performance-v1` + memory-v1
3. No-stream vs 16 MiB: no-stream direct remeasure + `stream16-pipeline-performance-3x-gpu1-v1`
4. 1.5B capacity: All-GPU OOM + no-stream OOM snapshot-v2 + CPU-BEST r1..r3

Placement Pareto와 2×3 placement matrix는 유효한 중간 진단이지만 핵심 주장
구조에는 사용하지 않아 result figure 목록에서 제외했다.

정확한 파일 경로는 상위 `../README.md`의 “Result figures and source runs”와
`../manifest.csv`에 기록돼 있다.

## 14. 재현 및 데이터 해석 규칙

- 성능 비교는 같은 GPU에서 telemetry/detail/nsys를 끈 반복 run을 우선한다.
- phase memory는 phase 시작 시 peak reset이 검증된 detail run을 사용한다.
- `allocated`, `reserved`, `nvidia-smi device used`를 서로 같은 값처럼 사용하지 않는다.
- OOM run은 평균 성능에 넣지 않고 capacity 실패 증거로만 사용한다.
- `timing_s/update_actor` residual로 optimizer 시간을 만들지 않는다.
- optimizer 비교는 `perf/actor_adam_step_total_wall_s` 직접 metric을 사용한다.
- streaming 최종 성능이라고 부르려면 Adam-H2D overlap 설정을 반드시 확인한다.
- JSON telemetry의 bucket별 기록은 수백 회/step 호출되어 wall time을 오염시킬 수 있다.
- 모든 개별 run의 정확한 설정과 metric은 `all_runs.csv`를 기준으로 확인한다.

## 15. 남은 미완료 분석

- backward +80 ms 중 backpressure 약 41 ms 외 나머지의 Nsight 완전 분해
- 최종 Adam-H2D overlap을 켠 상태의 16–512 MiB 동일조건 bucket sweep
- 1.5B에서 최종 overlap 구현과 기존 one-slot capacity sweep의 동일조건 재비교

이 세 항목은 기존 데이터로 확정한 결과가 아니며 후속 실험 항목이다.

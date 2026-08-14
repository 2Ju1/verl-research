# Final figure data manifest

최종 발표용 그림이 어떤 데이터와 스크립트에서 생성되는지 추적하기 위한 목록이다.
모든 시간은 별도 표기가 없으면 warm-up 2 step을 제외한 평균이며, 메모리는 GiB이다.

전체 시행착오와 95개 실험군·512개 run의 전수 기록은
`experiment-history/README.md`에서 시작한다. 상세 연대기는
`experiment-history/EXPERIMENT_HISTORY.md`, 기계 판독 원장은
`experiment-history/all_runs.csv`와 `experiment-history/all_runs.json`이다.

## 1. Placement Pareto

- Figure: `reports/figures/placement_pareto_05b.png`, `.svg`
- Script/data: `reports/figures/plot_placement_pareto.py`
- Data status: 스크립트의 `rows`에 고정된 최종 집계값
- Columns: placement, step time, throughput, allocated peak, device peak

| placement | step (s) | throughput | allocated peak | device peak |
|---|---:|---:|---:|---:|
| GGG | 4.505 | 147.9 | 10.242 | 11.355 |
| GGC | 5.328 | 125.1 | 10.246 | 11.331 |
| GCG | 4.956 | 134.5 | 8.404 | 9.485 |
| GCC | 5.733 | 116.3 | 8.405 | 9.392 |
| CGG | 5.527 | 120.6 | 10.248 | 11.114 |
| CGC | 6.439 | 103.3 | 10.246 | 11.347 |
| CCG | 5.922 | 112.6 | 8.406 | 9.175 |
| CCC | 6.812 | 98.1 | 8.403 | 9.280 |

## 2. Placement별 6-phase memory

- Figure: `reports/figures/phase_memory_2x3_05b.png`, `.svg`
- Script: `reports/figures/plot_phase_memory_2x3.py`
- Aggregated inputs:
  - `outputs/pa-repro-fp32-v1/summary/phase_configs.csv`
  - `outputs/pa-repro-fp32-v1/summary/actor_subphase_configs.csv`
- Config IDs: `FP32-R-GGG`, `FP32-R-GGC`, `FP32-R-GCG`, `FP32-R-GCC`, `FP32-R-CGG`, `FP32-R-CGC`, `FP32-R-CCG`, `FP32-R-CCC`
- Phase keys: rollout, actor_log_prob, reference_log_prob, actor_forward_end, actor_backward_end, actor_optimizer_end

## 3. All-on-GPU vs phase offload

- Figure: `reports/figures/allgpu_vs_phase_offload_05b.png`, `.svg`
- Script: `reports/figures/plot_allgpu_vs_phase_offload.py`
- Aggregated inputs:
  - `outputs/pa-repro-fp32-late-optimizer-smoke-v2/summary/phase_configs.csv`
  - `outputs/pa-repro-fp32-late-optimizer-smoke-v2/summary/actor_subphase_configs.csv`
- Compared configs: `FP32-LATE-GGG`, `FP32-LATE-CCC`
- 마지막 표시 이름은 `Update`; 원본 tag는 `actor_optimizer_end`

## 4. Phase offload best vs CPU AdamW

- Compact figure: `outputs/phase-best-vs-cpu-adamw-memory-v1/summary/phase_best_vs_cpu_adamw_compact.png`, `.pdf`
- Full figure: `outputs/phase-best-vs-cpu-adamw-memory-v1/summary/phase_best_vs_cpu_adamw_all_phases.png`, `.pdf`
- Script: `outputs/phase-best-vs-cpu-adamw-memory-v1/plot_phase_best_vs_cpu_adamw.py`
- Performance runs: `outputs/phase-best-vs-cpu-adamw-performance-v1/`
  - `PHASE-BEST-GPU-ADAMW-r1..r3`
  - `PHASE-BEST-PLUS-CPU-ADAMW-r1..r3`
- Memory runs: `outputs/phase-best-vs-cpu-adamw-memory-v1/`
  - `PHASE-BEST-GPU-ADAMW-r1`
  - `PHASE-BEST-PLUS-CPU-ADAMW-r1`

| configuration | Update peak | Update time |
|---|---:|---:|
| Phase offload best (GPU AdamW) | 8.408 | 0.129 |
| Phase offload best + CPU AdamW | 2.365 | 3.565 |

## 5. No-stream vs optimized 16 MiB streaming

- Figure: `outputs/pa-repro-fp32-v1/streaming-summary/nostream_vs_16mib_optimized.png`, `.pdf`
- Script: `outputs/pa-repro-fp32-v1/streaming-summary/plot_nostream_vs_16mib_optimized.py`
- No-stream runs: `outputs/nostream-vs-16mib-direct-remeasure-v1/NOSTREAM-DIRECT-REMEASURE-r1..r3`
- Streaming runs: `outputs/stream16-pipeline-performance-3x-gpu1-v1/STREAM16-S3-PIPELINE-PERF-r1..r3`
- Direct metrics: `perf/actor_backward_total_wall_s`, `perf/actor_adam_step_total_wall_s`, `perf/max_memory_allocated_gb`

| configuration | backward peak | backward time | Update time | actor update | step |
|---|---:|---:|---:|---:|---:|
| No-stream | 4.721 | 0.212 | 3.551 | 5.152 | 9.814 |
| 16 MiB, 3 slots, Adam-H2D overlap | 3.402 | 0.292 | 3.020 | 3.964 | 8.629 |

16 MiB 최종 설정의 핵심은 `cpu_grad_accumulation=true`와
`overlap_h2d_with_cpu_update=true`이다.

## 6. Streaming root-cause A/B/C

- Figure: `outputs/rootcause-abc-sequential-gpu1-v1/rootcause_abc_comparison.png`, `.pdf`
- Script: `outputs/rootcause-abc-sequential-gpu1-v1/plot_rootcause.py`
- Machine-readable summary: `outputs/rootcause-abc-sequential-gpu1-v1/rootcause_summary.json`
- Raw sequential runs:
  - `a-nostream/A-NOSTREAM-TELEMETRY-r1`
  - `b-stream16-serial/B-STREAM16-SERIAL-TELEMETRY-r1`
  - `c-stream16-pipeline/C-STREAM16-PIPELINE-TELEMETRY-r1`

이 데이터는 원인 분석용 telemetry run이며 최종 성능 수치로 사용하지 않는다.
최종 성능은 위 5번의 telemetry-off 3회 측정을 사용한다.

## 7. GPU Adam vs CPU Adam six phases

- Figure: `outputs/pa-cpu-adam-all-phase-fp32-v1/summary/cpu_adam_05b_by_phase.png`, `.pdf`
- Script: `outputs/pa-repro-fp32-v1/cpu-adam-summary/plot_cpu_adam_05b_phases.py`
- GPU Adam runs: `outputs/pa-late-gpu-adam-30steps-v1/FP32-LATE-CCC-r*`
- CPU Adam runs: `outputs/pa-cpu-adam-all-phase-fp32-v1/FP32-ALL-PHASE-CPU-ADAM-r*`
- Time source: `trainer-metrics-*.jsonl`
- Memory source: `events/memory-*.jsonl`

## Excluded/stale artifacts

- `nostream_vs_16mib_optimized_legacy_residual.*`: Update를 직접 측정하지 않고
  `timing_s/update_actor` residual로 역산한 과거 그림이므로 최종 결과에서 제외한다.
- `nostream-vs-16mib-direct-remeasure-v1`의 streaming run은 Adam-H2D overlap이 꺼진
  진단 대조군이다. 최종 streaming 성능에는 사용하지 않는다.
- Nsight 및 JSON telemetry run의 wall time은 계측 오버헤드가 있으므로 성능 막대에 사용하지 않는다.

## Reproduction

각 그림은 위에 명시된 Python script를 저장소 루트에서 실행하면 재생성된다.
Matplotlib cache 권한 경고를 피하려면 다음처럼 실행한다.

```bash
MPLCONFIGDIR=/tmp/verl-final-figures \
  envs/verl-titan/bin/python <plot-script>
```

## Slide figures shown in the final deck

아래는 최종 슬라이드에 사용한 네 그림과 실제 실험 결과 디렉터리의 일대일 대응이다.

### A. Phase offload GPU AdamW vs Phase offload + CPU AdamW

- Slide figure:
  `outputs/phase-best-vs-cpu-adamw-memory-v1/summary/phase_best_vs_cpu_adamw.png`
- Current compact figure:
  `outputs/phase-best-vs-cpu-adamw-memory-v1/summary/phase_best_vs_cpu_adamw_compact.png`
- Plot script:
  `outputs/phase-best-vs-cpu-adamw-memory-v1/plot_phase_best_vs_cpu_adamw.py`
- GPU AdamW performance results:
  - `outputs/phase-best-vs-cpu-adamw-performance-v1/PHASE-BEST-GPU-ADAMW-r1`
  - `outputs/phase-best-vs-cpu-adamw-performance-v1/PHASE-BEST-GPU-ADAMW-r2`
  - `outputs/phase-best-vs-cpu-adamw-performance-v1/PHASE-BEST-GPU-ADAMW-r3`
- CPU AdamW performance results:
  - `outputs/phase-best-vs-cpu-adamw-performance-v1/PHASE-BEST-PLUS-CPU-ADAMW-r1`
  - `outputs/phase-best-vs-cpu-adamw-performance-v1/PHASE-BEST-PLUS-CPU-ADAMW-r2`
  - `outputs/phase-best-vs-cpu-adamw-performance-v1/PHASE-BEST-PLUS-CPU-ADAMW-r3`
- GPU AdamW phase-memory result:
  `outputs/phase-best-vs-cpu-adamw-memory-v1/PHASE-BEST-GPU-ADAMW-r1`
- CPU AdamW phase-memory result:
  `outputs/phase-best-vs-cpu-adamw-memory-v1/PHASE-BEST-PLUS-CPU-ADAMW-r1`
- Main values: backward peak 4.72/4.72, Update peak 8.41/2.37,
  Update time 0.13/3.56.

### B. All on GPU vs Phase offload phase memory

- Slide figure: `reports/figures/allgpu_vs_phase_offload_05b.png`
- Vector figure: `reports/figures/allgpu_vs_phase_offload_05b.svg`
- Plot script: `reports/figures/plot_allgpu_vs_phase_offload.py`
- Aggregated result directory:
  `outputs/pa-repro-fp32-late-optimizer-smoke-v2/summary`
- Source aggregate files:
  - `outputs/pa-repro-fp32-late-optimizer-smoke-v2/summary/phase_configs.csv`
  - `outputs/pa-repro-fp32-late-optimizer-smoke-v2/summary/actor_subphase_configs.csv`
- All-on-GPU config ID: `FP32-LATE-GGG`
- Phase-offload config ID: `FP32-LATE-CCC`
- Raw run roots:
  - `outputs/pa-repro-fp32-late-optimizer-smoke-v2/FP32-LATE-GGG-r*`
  - `outputs/pa-repro-fp32-late-optimizer-smoke-v2/FP32-LATE-CCC-r*`
- Main values: backward 10.24→4.72 GiB, Update 10.24→8.40 GiB.

### C. No-stream vs optimized 16 MiB streaming

- Slide figure:
  `outputs/pa-repro-fp32-v1/streaming-summary/nostream_vs_16mib_optimized.png`
- PDF:
  `outputs/pa-repro-fp32-v1/streaming-summary/nostream_vs_16mib_optimized.pdf`
- Plot script:
  `outputs/pa-repro-fp32-v1/streaming-summary/plot_nostream_vs_16mib_optimized.py`
- No-stream results:
  - `outputs/nostream-vs-16mib-direct-remeasure-v1/NOSTREAM-DIRECT-REMEASURE-r1`
  - `outputs/nostream-vs-16mib-direct-remeasure-v1/NOSTREAM-DIRECT-REMEASURE-r2`
  - `outputs/nostream-vs-16mib-direct-remeasure-v1/NOSTREAM-DIRECT-REMEASURE-r3`
- Final 16 MiB overlap results:
  - `outputs/stream16-pipeline-performance-3x-gpu1-v1/STREAM16-S3-PIPELINE-PERF-r1`
  - `outputs/stream16-pipeline-performance-3x-gpu1-v1/STREAM16-S3-PIPELINE-PERF-r2`
  - `outputs/stream16-pipeline-performance-3x-gpu1-v1/STREAM16-S3-PIPELINE-PERF-r3`
- Main values: backward peak 4.72→3.40 GiB, backward time 0.21→0.29 s,
  Update time 3.55→3.02 s.
- Do not use for the final bar:
  `outputs/nostream-vs-16mib-direct-remeasure-v1/STREAM16-S3-DIRECT-REMEASURE-r*`.
  이 streaming 대조군은 Adam-H2D overlap이 꺼져 있다.

### D. Qwen2.5-1.5B FP32 capacity

- Slide figure:
  `outputs/pa-capacity-fp32-qwen15b-bucket-sweep-v1/summary/qwen15b_fp32_capacity.png`
- PDF:
  `outputs/pa-capacity-fp32-qwen15b-bucket-sweep-v1/summary/qwen15b_fp32_capacity.pdf`
- Plot script:
  `outputs/pa-capacity-fp32-qwen15b-bucket-sweep-v1/summary/plot_15b_capacity.py`
- All-GPU OOM run:
  `outputs/pa-capacity-fp32-qwen15b-v1/FP32-FINAL-GPU-OPT-r1`
- CPU Adam no-stream OOM run:
  `outputs/pa-capacity-fp32-qwen15b-nostream-oom-snapshot-v2/FP32-15B-CPU-NOSTREAM-OOM-SNAPSHOT-r1`
- CPU Adam + streaming successful 3-run results:
  - `outputs/pa-capacity-fp32-qwen15b-v1/FP32-FINAL-CPU-BEST-r1`
  - `outputs/pa-capacity-fp32-qwen15b-v1/FP32-FINAL-CPU-BEST-r2`
  - `outputs/pa-capacity-fp32-qwen15b-v1/FP32-FINAL-CPU-BEST-r3`
- Additional bucket sweep results:
  `outputs/pa-capacity-fp32-qwen15b-bucket-sweep-v1/`
- OOM evidence: no-stream stdout의 CUDA OOM 메시지에 PyTorch allocated
  memory가 11.34 GiB로 기록되어 있다.
- Successful run metric: `perf/max_memory_allocated_gb`는 약 8.377 GiB이며,
  그림은 phase/detail peak를 반올림한 8.44 GiB를 표시한다.
- GPU capacity line: CUDA가 보고한 total capacity 11.90 GiB.

### E. Qwen2.5-0.5B bucket-size sweep

- Raw result root: `outputs/pa-optimized-bucket-sweep-v1`
- Matrix: `reports/final-figure-data/collected/05_bucket_size_sweep_05b/optimized_bucket_sweep.json`
- Buckets: 16, 32, 64, 128, 256, 512 MiB
- Repeats: 각 bucket마다 r1, r2, r3
- Run IDs: `STREAM-OPT-B{16,32,64,128,256,512}-r{1,2,3}`
- Collected copy: `reports/final-figure-data/collected/05_bucket_size_sweep_05b`
- Common settings: 3 staging slots, async D2H, early gradient release,
  reusable GPU packing buffers, direct CPU gradient buffers.
- Important limitation: 이 sweep은 `overlap_h2d_with_cpu_update=false`로 수행됐다.
  즉 bucket 크기에 따른 D2H/backward memory trade-off를 비교하는 과거 sweep이며,
  최종 16 MiB Adam-H2D overlap 성능 실험은 아니다.

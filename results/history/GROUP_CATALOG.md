# Exhaustive output-group catalog

자동 생성된 전체 output group 목록이다. 수치는 stdout에서 warm-up을 제외하고 읽은 값이며,
metric이 없거나 실패한 run은 빈 칸으로 남긴다.

## activation-fallback-smoke

- Path: `outputs/activation-fallback-smoke`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `C03_activation-r1` | ok |  |  |  |  |  | 11.066 |
| `C15_all-r1` | ok |  |  |  |  |  | 10.150 |

## activation-fixed-full

- Path: `outputs/activation-fixed-full`
- Runs: 9

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `C03_activation-r1` | ok |  |  |  |  |  | 11.066 |
| `C03_activation-r2` | ok |  |  |  |  |  | 11.066 |
| `C03_activation-r3` | ok |  |  |  |  |  | 11.066 |
| `C11_ref_optimizer_activation-r1` | failed |  |  |  |  |  | 10.151 |
| `C11_ref_optimizer_activation-r2` | ok |  |  |  |  |  | 10.151 |
| `C11_ref_optimizer_activation-r3` | ok |  |  |  |  |  | 10.151 |
| `C15_all-r1` | ok |  |  |  |  |  | 10.150 |
| `C15_all-r2` | ok |  |  |  |  |  | 10.150 |
| `C15_all-r3` | failed |  |  |  |  |  |  |

## actor-phase-async-smoke

- Path: `outputs/actor-phase-async-smoke`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-ASYNC-r1` | ok | Phase actor residency / async D2H CPU Adam | 64 | 2 | 0.931 | 5.124 | 3.235 |

## all-gpu-memory-diagnostic-v1

- Path: `outputs/all-gpu-memory-diagnostic-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `R-GGG-MEMTRACE-r1` | ok | All GPU residency / phase allocator trace |  |  | 0.259 | 0.093 | 10.248 |

## all-gpu-memory-diagnostic-v2

- Path: `outputs/all-gpu-memory-diagnostic-v2`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `R-GGG-MEMTRACE-r1` | ok | All GPU residency / phase allocator trace |  |  | 0.258 | 0.099 | 10.248 |

## all-gpu-memory-diagnostic-v3

- Path: `outputs/all-gpu-memory-diagnostic-v3`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `R-GGG-MEMTRACE-r1` | ok | All GPU residency / phase allocator trace |  |  | 0.257 | 0.094 | 10.248 |

## backward-nsys-rootcause-v1

- Path: `outputs/backward-nsys-rootcause-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `NSYS-NOSTREAM-r1` | ok | No-stream backward Nsight |  |  | 0.225 | 3.592 | 4.721 |
| `NSYS-STREAM16-r1` | failed | 16 MiB streaming backward Nsight | 16 | 3 |  |  |  |

## core-memory-diagnostics-v1

- Path: `outputs/core-memory-diagnostics-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `O-PART-MEMTRACE-r1` | failed | CPU Adam / actor partial residency / allocator trace |  |  |  |  |  |
| `R-GCC-MEMTRACE-r1` | ok | A GPU / Ref phase / Adam swap / allocator trace |  |  | 0.245 | 0.132 | 8.408 |

## cpu-best-foreach-true-fp32-v1

- Path: `outputs/cpu-best-foreach-true-fp32-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-CPU-BEST-FOREACH-r1` | ok | CPU-BEST / torch AdamW foreach=True | 8 | 1 | 0.895 | 3.302 | 3.388 |

## non-fsdp-unified-fp16-smoke-v1

- Path: `outputs/non-fsdp-unified-fp16-smoke-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `O-SYNC-r1` | ok | Whole synchronous CPU offload |  |  | 0.299 | 6.953 | 4.219 |

## non-fsdp-unified-smoke-v1

- Path: `outputs/non-fsdp-unified-smoke-v1`
- Runs: 9

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `O-ASYNC-r1` | ok | Async D2H + early release | 64 | 2 | 1.241 | 5.080 | 3.252 |
| `O-CACC-r1` | ok | Four-microbatch CPU accumulation | 64 | 2 | 3.909 | 5.074 | 2.978 |
| `O-GACC-r1` | ok | Four-microbatch GPU accumulation | 64 |  | 0.608 | 5.964 | 4.784 |
| `O-PART-r1` | ok | CPU Adam / actor partial residency |  |  | 0.438 | 5.732 | 4.221 |
| `O-PIPE-r1` | ok | CPU accumulation + H2D pipeline | 64 | 2 | 3.854 | 4.602 | 2.978 |
| `O-REL-r1` | ok | Synchronous bucket release | 64 |  | 0.441 | 6.092 | 4.220 |
| `O-SYNC-r1` | ok | Whole synchronous CPU offload |  |  | 0.434 | 5.713 | 4.217 |
| `R-CGC-r1` | ok | A phase / Ref GPU / Adam swap |  |  | 0.439 | 0.122 | 10.264 |
| `R-GCC-r1` | ok | A GPU / Ref phase / Adam swap |  |  | 0.429 | 0.125 | 8.411 |

## non-fsdp-unified-v1

- Path: `outputs/non-fsdp-unified-v1`
- Runs: 48

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `O-ASYNC-r1` | ok | Async D2H + early release | 64 | 2 | 0.917 | 4.930 | 3.223 |
| `O-ASYNC-r2` | ok | Async D2H + early release | 64 | 2 | 0.922 | 4.348 | 3.223 |
| `O-ASYNC-r3` | ok | Async D2H + early release | 64 | 2 | 0.915 | 4.988 | 3.223 |
| `O-BKT-r1` | ok | Synchronous bucket D2H | 64 |  | 0.245 | 4.934 | 4.263 |
| `O-BKT-r2` | ok | Synchronous bucket D2H | 64 |  | 0.247 | 5.458 | 4.263 |
| `O-BKT-r3` | ok | Synchronous bucket D2H | 64 |  | 0.244 | 5.551 | 4.263 |
| `O-CACC-r1` | ok | Four-microbatch CPU accumulation | 64 | 2 | 3.578 | 5.080 | 2.978 |
| `O-CACC-r2` | ok | Four-microbatch CPU accumulation | 64 | 2 | 3.531 | 4.208 | 2.978 |
| `O-CACC-r3` | ok | Four-microbatch CPU accumulation | 64 | 2 | 3.577 | 4.885 | 2.978 |
| `O-GACC-r1` | ok | Four-microbatch GPU accumulation | 64 |  | 0.535 | 5.500 | 4.747 |
| `O-GACC-r2` | ok | Four-microbatch GPU accumulation | 64 |  | 0.545 | 4.911 | 4.747 |
| `O-GACC-r3` | ok | Four-microbatch GPU accumulation | 64 |  | 0.546 | 5.026 | 4.747 |
| `O-PART-r1` | ok | CPU Adam / actor partial residency |  |  | 0.244 | 5.286 | 4.222 |
| `O-PART-r2` | ok | CPU Adam / actor partial residency |  |  | 0.245 | 5.368 | 4.222 |
| `O-PART-r3` | ok | CPU Adam / actor partial residency |  |  | 0.246 | 5.316 | 4.222 |
| `O-PIPE-r1` | ok | CPU accumulation + H2D pipeline | 64 | 2 | 3.487 | 4.380 | 2.978 |
| `O-PIPE-r2` | ok | CPU accumulation + H2D pipeline | 64 | 2 | 3.519 | 4.230 | 2.978 |
| `O-PIPE-r3` | ok | CPU accumulation + H2D pipeline | 64 | 2 | 3.669 | 4.474 | 2.978 |
| `O-REL-r1` | ok | Synchronous bucket release | 64 |  | 0.245 | 5.674 | 4.217 |
| `O-REL-r2` | ok | Synchronous bucket release | 64 |  | 0.248 | 5.773 | 4.217 |
| `O-REL-r3` | ok | Synchronous bucket release | 64 |  | 0.245 | 5.769 | 4.217 |
| `O-SYNC-r1` | ok | Whole synchronous CPU offload |  |  | 0.244 | 5.270 | 4.217 |
| `O-SYNC-r2` | ok | Whole synchronous CPU offload |  |  | 0.247 | 5.242 | 4.217 |
| `O-SYNC-r3` | ok | Whole synchronous CPU offload |  |  | 0.247 | 5.383 | 4.217 |
| `R-CCC-r1` | ok | A phase / Ref phase / Adam swap |  |  | 0.251 | 0.129 | 8.410 |
| `R-CCC-r2` | ok | A phase / Ref phase / Adam swap |  |  | 0.255 | 0.128 | 8.410 |
| `R-CCC-r3` | ok | A phase / Ref phase / Adam swap |  |  | 0.253 | 0.129 | 8.410 |
| `R-CCG-r1` | ok | A phase / Ref phase / Adam GPU |  |  | 0.254 | 0.079 | 8.410 |
| `R-CCG-r2` | ok | A phase / Ref phase / Adam GPU |  |  | 0.257 | 0.079 | 8.410 |
| `R-CCG-r3` | ok | A phase / Ref phase / Adam GPU |  |  | 0.256 | 0.079 | 8.410 |
| `R-CGC-r1` | ok | A phase / Ref GPU / Adam swap |  |  | 0.255 | 0.128 | 10.262 |
| `R-CGC-r2` | ok | A phase / Ref GPU / Adam swap |  |  | 0.258 | 0.129 | 10.262 |
| `R-CGC-r3` | ok | A phase / Ref GPU / Adam swap |  |  | 0.261 | 0.129 | 10.262 |
| `R-CGG-r1` | ok | A phase / Ref GPU / Adam GPU |  |  | 0.270 | 0.079 | 10.251 |
| `R-CGG-r2` | ok | A phase / Ref GPU / Adam GPU |  |  | 0.263 | 0.079 | 10.251 |
| `R-CGG-r3` | ok | A phase / Ref GPU / Adam GPU |  |  | 0.267 | 0.079 | 10.251 |
| `R-GCC-r1` | ok | A GPU / Ref phase / Adam swap |  |  | 0.257 | 0.128 | 8.408 |
| `R-GCC-r2` | ok | A GPU / Ref phase / Adam swap |  |  | 0.257 | 0.128 | 8.408 |
| `R-GCC-r3` | ok | A GPU / Ref phase / Adam swap |  |  | 0.262 | 0.128 | 8.408 |
| `R-GCG-r1` | ok | A GPU / Ref phase / Adam GPU |  |  | 0.261 | 0.079 | 8.409 |
| `R-GCG-r2` | ok | A GPU / Ref phase / Adam GPU |  |  | 0.265 | 0.079 | 8.409 |
| `R-GCG-r3` | ok | A GPU / Ref phase / Adam GPU |  |  | 0.261 | 0.079 | 8.409 |
| `R-GGC-r1` | ok | A GPU / Ref GPU / Adam swap |  |  | 0.260 | 0.128 | 10.260 |
| `R-GGC-r2` | ok | A GPU / Ref GPU / Adam swap |  |  | 0.269 | 0.127 | 10.260 |
| `R-GGC-r3` | ok | A GPU / Ref GPU / Adam swap |  |  | 0.268 | 0.129 | 10.260 |
| `R-GGG-r1` | ok | A GPU / Ref GPU / Adam GPU |  |  | 0.242 | 0.079 | 10.246 |
| `R-GGG-r2` | ok | A GPU / Ref GPU / Adam GPU |  |  | 0.256 | 0.079 | 10.246 |
| `R-GGG-r3` | ok | A GPU / Ref GPU / Adam GPU |  |  | 0.252 | 0.079 | 10.246 |

## nostream-vs-16mib-direct-remeasure-v1

- Path: `outputs/nostream-vs-16mib-direct-remeasure-v1`
- Runs: 6

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `NOSTREAM-DIRECT-REMEASURE-r1` | ok | No-stream / direct AdamW timing remeasure |  |  | 0.212 | 3.525 | 4.721 |
| `NOSTREAM-DIRECT-REMEASURE-r2` | ok | No-stream / direct AdamW timing remeasure |  |  | 0.212 | 3.557 | 4.721 |
| `NOSTREAM-DIRECT-REMEASURE-r3` | ok | No-stream / direct AdamW timing remeasure |  |  | 0.213 | 3.571 | 4.721 |
| `STREAM16-S3-DIRECT-REMEASURE-r1` | ok | 16 MiB / 3 slots / direct CPU gradients / lazy buffers | 16 | 3 | 0.289 | 3.678 | 3.402 |
| `STREAM16-S3-DIRECT-REMEASURE-r2` | ok | 16 MiB / 3 slots / direct CPU gradients / lazy buffers | 16 | 3 | 0.291 | 3.668 | 3.401 |
| `STREAM16-S3-DIRECT-REMEASURE-r3` | ok | 16 MiB / 3 slots / direct CPU gradients / lazy buffers | 16 | 3 | 0.295 | 3.694 | 3.401 |

## nostream-vs-16mib-telemetry-once-gpu1-v1

- Path: `outputs/nostream-vs-16mib-telemetry-once-gpu1-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `NOSTREAM-TELEMETRY-ONCE-r1` | ok | No-stream / telemetry isolation |  |  | 0.222 | 3.561 | 4.721 |
| `STREAM16-S3-DIRECT-TELEMETRY-ONCE-r1` | ok | 16 MiB / 3 slots / direct CPU gradients / telemetry isolation | 16 | 3 | 0.580 | 3.709 | 3.401 |

## nostream-vs-16mib-telemetry-once-v1

- Path: `outputs/nostream-vs-16mib-telemetry-once-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `NOSTREAM-TELEMETRY-ONCE-r1` | ok | No-stream / telemetry isolation |  |  | 0.218 | 3.531 | 4.721 |
| `STREAM16-S3-DIRECT-TELEMETRY-ONCE-r1` | ok | 16 MiB / 3 slots / direct CPU gradients / telemetry isolation | 16 | 3 | 0.559 | 3.688 | 3.401 |

## offload-c04-diagnostics-final

- Path: `outputs/offload-c04-diagnostics-final`
- Runs: 27

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `A00_c00_control-r1` | ok |  |  |  |  |  | 10.601 |
| `A00_c00_control-r2` | ok |  |  |  |  |  | 10.601 |
| `A00_c00_control-r3` | ok |  |  |  |  |  | 10.601 |
| `A01_c04_baseline-r1` | failed |  |  |  |  |  |  |
| `A01_c04_baseline-r2` | failed |  |  |  |  |  |  |
| `A01_c04_baseline-r3` | failed |  |  |  |  |  |  |
| `A02_reference_headroom-r1` | ok |  |  |  |  |  | 9.684 |
| `A02_reference_headroom-r2` | ok |  |  |  |  |  | 9.684 |
| `A02_reference_headroom-r3` | ok |  |  |  |  |  | 9.684 |
| `A03_actor_cache_retained-r1` | ok |  |  |  |  |  | 10.602 |
| `A03_actor_cache_retained-r2` | ok |  |  |  |  |  | 10.602 |
| `A03_actor_cache_retained-r3` | ok |  |  |  |  |  | 10.602 |
| `A04_rollout_cache_retained-r1` | failed |  |  |  |  |  |  |
| `A04_rollout_cache_retained-r2` | failed |  |  |  |  |  |  |
| `A04_rollout_cache_retained-r3` | failed |  |  |  |  |  |  |
| `A05_expandable_segments-r1` | ok |  |  |  |  |  | 10.598 |
| `A05_expandable_segments-r2` | ok |  |  |  |  |  | 10.598 |
| `A05_expandable_segments-r3` | ok |  |  |  |  |  | 10.598 |
| `A06_adamw_no_foreach-r1` | failed |  |  |  |  |  |  |
| `A06_adamw_no_foreach-r2` | failed |  |  |  |  |  |  |
| `A06_adamw_no_foreach-r3` | failed |  |  |  |  |  |  |
| `A08_all_cache_retained-r1` | ok |  |  |  |  |  | 10.602 |
| `A08_all_cache_retained-r2` | ok |  |  |  |  |  | 10.602 |
| `A08_all_cache_retained-r3` | ok |  |  |  |  |  | 10.602 |
| `A09_cache_retained_expandable-r1` | ok |  |  |  |  |  | 10.598 |
| `A09_cache_retained_expandable-r2` | ok |  |  |  |  |  | 10.598 |
| `A09_cache_retained_expandable-r3` | ok |  |  |  |  |  | 10.598 |

## offload-c04-diagnostics-v1

- Path: `outputs/offload-c04-diagnostics-v1`
- Runs: 10

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `A00_c00_control-r1` | ok |  |  |  |  |  | 10.601 |
| `A01_c04_baseline-r1` | failed |  |  |  |  |  |  |
| `A02_reference_headroom-r1` | ok |  |  |  |  |  | 9.684 |
| `A03_actor_cache_retained-r1` | ok |  |  |  |  |  | 10.602 |
| `A04_rollout_cache_retained-r1` | failed |  |  |  |  |  |  |
| `A05_expandable_segments-r1` | ok |  |  |  |  |  | 10.598 |
| `A06_adamw_no_foreach-r1` | failed |  |  |  |  |  |  |
| `A07_oom_snapshot-r1` | failed |  |  |  |  |  |  |
| `A08_all_cache_retained-r1` | ok |  |  |  |  |  | 10.602 |
| `A09_cache_retained_expandable-r1` | ok |  |  |  |  |  | 10.598 |

## offload-fullft-v3

- Path: `outputs/offload-fullft-v3`
- Runs: 24

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `C00_none-r1` | ok |  |  |  |  |  | 11.064 |
| `C00_none-r2` | ok |  |  |  |  |  | 11.064 |
| `C00_none-r3` | ok |  |  |  |  |  | 11.064 |
| `C01_ref-r1` | ok |  |  |  |  |  | 10.150 |
| `C01_ref-r2` | ok |  |  |  |  |  | 10.150 |
| `C01_ref-r3` | ok |  |  |  |  |  | 10.150 |
| `C02_optimizer-r1` | ok |  |  |  |  |  | 11.064 |
| `C02_optimizer-r2` | ok |  |  |  |  |  | 11.064 |
| `C02_optimizer-r3` | ok |  |  |  |  |  | 11.064 |
| `C03_activation-r1` | ok |  |  |  |  |  | 11.064 |
| `C03_activation-r2` | ok |  |  |  |  |  | 11.064 |
| `C03_activation-r3` | ok |  |  |  |  |  | 11.064 |
| `C04_actor_param-r1` | failed |  |  |  |  |  |  |
| `C04_actor_param-r2` | failed |  |  |  |  |  |  |
| `C04_actor_param-r3` | failed |  |  |  |  |  |  |
| `C05_ref_optimizer-r1` | ok |  |  |  |  |  | 10.150 |
| `C05_ref_optimizer-r2` | ok |  |  |  |  |  | 10.150 |
| `C05_ref_optimizer-r3` | ok |  |  |  |  |  | 10.150 |
| `C11_ref_optimizer_activation-r1` | ok |  |  |  |  |  | 10.150 |
| `C11_ref_optimizer_activation-r2` | ok |  |  |  |  |  | 10.150 |
| `C11_ref_optimizer_activation-r3` | ok |  |  |  |  |  | 10.150 |
| `C15_all-r1` | ok |  |  |  |  |  | 10.150 |
| `C15_all-r2` | ok |  |  |  |  |  | 10.150 |
| `C15_all-r3` | ok |  |  |  |  |  | 10.150 |

## offload-fullft-v4

- Path: `outputs/offload-fullft-v4`
- Runs: 15

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `C00_none-r1` | ok |  |  |  |  |  | 11.064 |
| `C00_none-r2` | ok |  |  |  |  |  | 11.064 |
| `C00_none-r3` | ok |  |  |  |  |  | 11.064 |
| `C01_ref-r1` | ok |  |  |  |  |  | 10.150 |
| `C01_ref-r2` | ok |  |  |  |  |  | 10.150 |
| `C01_ref-r3` | ok |  |  |  |  |  | 10.150 |
| `C02_optimizer-r1` | ok |  |  |  |  |  | 11.064 |
| `C02_optimizer-r2` | ok |  |  |  |  |  | 11.064 |
| `C02_optimizer-r3` | ok |  |  |  |  |  | 11.064 |
| `C04_actor_param-r1` | failed |  |  |  |  |  |  |
| `C04_actor_param-r2` | failed |  |  |  |  |  |  |
| `C04_actor_param-r3` | failed |  |  |  |  |  |  |
| `C05_ref_optimizer-r1` | ok |  |  |  |  |  | 10.150 |
| `C05_ref_optimizer-r2` | ok |  |  |  |  |  | 10.150 |
| `C05_ref_optimizer-r3` | ok |  |  |  |  |  | 10.150 |

## offload-fullft-v5-detail

- Path: `outputs/offload-fullft-v5-detail`
- Runs: 8

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `C00_none-r1` | ok |  |  |  |  |  | 11.064 |
| `C01_ref-r1` | ok |  |  |  |  |  | 10.149 |
| `C02_optimizer-r1` | ok |  |  |  |  |  | 11.064 |
| `C03_activation-r1` | ok |  |  |  |  |  | 11.066 |
| `C04_actor_param-r1` | failed |  |  |  |  |  |  |
| `C05_ref_optimizer-r1` | ok |  |  |  |  |  | 10.149 |
| `C11_ref_optimizer_activation-r1` | ok |  |  |  |  |  | 10.149 |
| `C15_all-r1` | ok |  |  |  |  |  | 10.150 |

## offload-fullft-v5-performance

- Path: `outputs/offload-fullft-v5-performance`
- Runs: 24

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `C00_none-r1` | ok |  |  |  |  |  | 11.064 |
| `C00_none-r2` | ok |  |  |  |  |  | 11.064 |
| `C00_none-r3` | ok |  |  |  |  |  | 11.064 |
| `C01_ref-r1` | ok |  |  |  |  |  | 10.149 |
| `C01_ref-r2` | ok |  |  |  |  |  | 10.149 |
| `C01_ref-r3` | ok |  |  |  |  |  | 10.149 |
| `C02_optimizer-r1` | ok |  |  |  |  |  | 11.064 |
| `C02_optimizer-r2` | ok |  |  |  |  |  | 11.064 |
| `C02_optimizer-r3` | ok |  |  |  |  |  | 11.064 |
| `C03_activation-r1` | ok |  |  |  |  |  | 11.066 |
| `C03_activation-r2` | ok |  |  |  |  |  | 11.066 |
| `C03_activation-r3` | ok |  |  |  |  |  | 11.066 |
| `C04_actor_param-r1` | failed |  |  |  |  |  |  |
| `C04_actor_param-r2` | failed |  |  |  |  |  |  |
| `C04_actor_param-r3` | failed |  |  |  |  |  |  |
| `C05_ref_optimizer-r1` | ok |  |  |  |  |  | 10.149 |
| `C05_ref_optimizer-r2` | ok |  |  |  |  |  | 10.149 |
| `C05_ref_optimizer-r3` | ok |  |  |  |  |  | 10.149 |
| `C11_ref_optimizer_activation-r1` | ok |  |  |  |  |  | 10.149 |
| `C11_ref_optimizer_activation-r2` | ok |  |  |  |  |  | 10.149 |
| `C11_ref_optimizer_activation-r3` | ok |  |  |  |  |  | 10.149 |
| `C15_all-r1` | ok |  |  |  |  |  | 10.150 |
| `C15_all-r2` | ok |  |  |  |  |  | 10.150 |
| `C15_all-r3` | ok |  |  |  |  |  | 10.150 |

## offload-fullft-v5-smoke

- Path: `outputs/offload-fullft-v5-smoke`
- Runs: 5

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `C00_none-r1` | ok |  |  |  |  |  | 10.601 |
| `C01_ref-r1` | ok |  |  |  |  |  | 9.683 |
| `C03_activation-r1` | ok |  |  |  |  |  | 10.602 |
| `C04_actor_param-r1` | failed |  |  |  |  |  |  |
| `C15_all-r1` | ok |  |  |  |  |  | 9.684 |

## offload-g01-expandable-check

- Path: `outputs/offload-g01-expandable-check`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `G01_reference_resident_expandable-r1` | ok |  |  |  |  |  | 11.058 |

## offload-gpu-adam-residency-v1-performance

- Path: `outputs/offload-gpu-adam-residency-v1-performance`
- Runs: 18

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `G00_actor_resident_base-r1` | ok |  |  |  |  |  | 10.149 |
| `G00_actor_resident_base-r2` | ok |  |  |  |  |  | 10.149 |
| `G00_actor_resident_base-r3` | ok |  |  |  |  |  | 10.149 |
| `G01_reference_resident-r1` | ok |  |  |  |  |  | 11.066 |
| `G01_reference_resident-r2` | ok |  |  |  |  |  | 11.066 |
| `G01_reference_resident-r3` | ok |  |  |  |  |  | 11.066 |
| `G02_optimizer_resident-r1` | ok |  |  |  |  |  | 10.149 |
| `G02_optimizer_resident-r2` | ok |  |  |  |  |  | 10.149 |
| `G02_optimizer_resident-r3` | ok |  |  |  |  |  | 10.149 |
| `G03_activation_gpu-r1` | ok |  |  |  |  |  | 10.149 |
| `G03_activation_gpu-r2` | ok |  |  |  |  |  | 10.149 |
| `G03_activation_gpu-r3` | ok |  |  |  |  |  | 10.149 |
| `G04_reference_only_offload-r1` | ok |  |  |  |  |  | 10.149 |
| `G04_reference_only_offload-r2` | ok |  |  |  |  |  | 10.149 |
| `G04_reference_only_offload-r3` | ok |  |  |  |  |  | 10.149 |
| `G05_all_gpu-r1` | ok |  |  |  |  |  | 11.064 |
| `G05_all_gpu-r2` | ok |  |  |  |  |  | 11.064 |
| `G05_all_gpu-r3` | ok |  |  |  |  |  | 11.064 |

## offload-gpu-adam-residency-v2-noforeach-performance

- Path: `outputs/offload-gpu-adam-residency-v2-noforeach-performance`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `G01_reference_resident-r1` | failed |  |  |  |  |  |  |

## offload-gpu-adam-residency-v2-performance

- Path: `outputs/offload-gpu-adam-residency-v2-performance`
- Runs: 15

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `G00_cpu_resident_base-r1` | ok |  |  |  |  |  | 10.150 |
| `G00_cpu_resident_base-r2` | ok |  |  |  |  |  | 10.150 |
| `G00_cpu_resident_base-r3` | ok |  |  |  |  |  | 10.150 |
| `G01_reference_resident-r1` | failed |  |  |  |  |  |  |
| `G01_reference_resident-r2` | failed |  |  |  |  |  |  |
| `G01_reference_resident-r3` | failed |  |  |  |  |  |  |
| `G02_optimizer_resident-r1` | ok |  |  |  |  |  | 10.150 |
| `G02_optimizer_resident-r2` | ok |  |  |  |  |  | 10.150 |
| `G02_optimizer_resident-r3` | ok |  |  |  |  |  | 10.150 |
| `G03_activation_gpu-r1` | ok |  |  |  |  |  | 10.150 |
| `G03_activation_gpu-r2` | ok |  |  |  |  |  | 10.150 |
| `G03_activation_gpu-r3` | ok |  |  |  |  |  | 10.150 |
| `G04_actor_resident-r1` | ok |  |  |  |  |  | 10.149 |
| `G04_actor_resident-r2` | ok |  |  |  |  |  | 10.149 |
| `G04_actor_resident-r3` | ok |  |  |  |  |  | 10.149 |

## offload-gpu-adam-residency-v2-performance-final

- Path: `outputs/offload-gpu-adam-residency-v2-performance-final`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `G01_reference_resident-r1` | failed |  |  |  |  |  |  |

## offload-gpu-adam-residency-v3-performance

- Path: `outputs/offload-gpu-adam-residency-v3-performance`
- Runs: 15

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `G00_cpu_resident_base-r1` | ok |  |  |  |  |  | 10.138 |
| `G00_cpu_resident_base-r2` | ok |  |  |  |  |  | 10.138 |
| `G00_cpu_resident_base-r3` | ok |  |  |  |  |  | 10.138 |
| `G01_reference_resident-r1` | ok |  |  |  |  |  | 11.058 |
| `G01_reference_resident-r2` | ok |  |  |  |  |  | 11.058 |
| `G01_reference_resident-r3` | ok |  |  |  |  |  | 11.058 |
| `G02_optimizer_resident-r1` | ok |  |  |  |  |  | 10.138 |
| `G02_optimizer_resident-r2` | ok |  |  |  |  |  | 10.138 |
| `G02_optimizer_resident-r3` | ok |  |  |  |  |  | 10.138 |
| `G03_activation_gpu-r1` | ok |  |  |  |  |  | 10.138 |
| `G03_activation_gpu-r2` | ok |  |  |  |  |  | 10.138 |
| `G03_activation_gpu-r3` | ok |  |  |  |  |  | 10.138 |
| `G04_actor_resident-r1` | ok |  |  |  |  |  | 10.138 |
| `G04_actor_resident-r2` | ok |  |  |  |  |  | 10.138 |
| `G04_actor_resident-r3` | ok |  |  |  |  |  | 10.138 |

## offload-measurement-v2

- Path: `outputs/offload-measurement-v2`
- Runs: 5

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `C00_none-r1` | ok |  |  |  |  |  | 11.064 |
| `C01_ref-r1` | ok |  |  |  |  |  | 10.150 |
| `C02_optimizer-r1` | ok |  |  |  |  |  | 11.064 |
| `C04_actor_param-r1` | failed |  |  |  |  |  |  |
| `C15_all-r1` | ok |  |  |  |  |  | 10.150 |

## offload-residency-smoke

- Path: `outputs/offload-residency-smoke`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `M00_phase_min-r1` | ok |  |  |  |  |  | 6.467 |
| `M03_optimizer_resident-r1` | ok |  |  |  |  |  | 9.684 |
| `M05_all_gpu-r1` | ok |  |  |  |  |  | 10.601 |

## offload-residency-v1-detail

- Path: `outputs/offload-residency-v1-detail`
- Runs: 6

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `M00_phase_min-r1` | ok |  |  |  |  |  | 6.469 |
| `M01_actor_resident-r1` | ok |  |  |  |  |  | 6.468 |
| `M02_ref_resident-r1` | ok |  |  |  |  |  | 7.385 |
| `M03_optimizer_resident-r1` | ok |  |  |  |  |  | 10.150 |
| `M04_activation_gpu-r1` | ok |  |  |  |  |  | 6.469 |
| `M05_all_gpu-r1` | ok |  |  |  |  |  | 11.064 |

## offload-residency-v1-performance

- Path: `outputs/offload-residency-v1-performance`
- Runs: 18

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `M00_phase_min-r1` | ok |  |  |  |  |  | 6.469 |
| `M00_phase_min-r2` | ok |  |  |  |  |  | 6.469 |
| `M00_phase_min-r3` | ok |  |  |  |  |  | 6.469 |
| `M01_actor_resident-r1` | ok |  |  |  |  |  | 6.468 |
| `M01_actor_resident-r2` | ok |  |  |  |  |  | 6.468 |
| `M01_actor_resident-r3` | ok |  |  |  |  |  | 6.468 |
| `M02_ref_resident-r1` | ok |  |  |  |  |  | 7.385 |
| `M02_ref_resident-r2` | ok |  |  |  |  |  | 7.385 |
| `M02_ref_resident-r3` | ok |  |  |  |  |  | 7.385 |
| `M03_optimizer_resident-r1` | ok |  |  |  |  |  | 10.150 |
| `M03_optimizer_resident-r2` | ok |  |  |  |  |  | 10.150 |
| `M03_optimizer_resident-r3` | ok |  |  |  |  |  | 10.150 |
| `M04_activation_gpu-r1` | ok |  |  |  |  |  | 6.469 |
| `M04_activation_gpu-r2` | ok |  |  |  |  |  | 6.469 |
| `M04_activation_gpu-r3` | ok |  |  |  |  |  | 6.469 |
| `M05_all_gpu-r1` | ok |  |  |  |  |  | 11.064 |
| `M05_all_gpu-r2` | ok |  |  |  |  |  | 11.064 |
| `M05_all_gpu-r3` | ok |  |  |  |  |  | 11.064 |

## offload-residency-v1-smoke

- Path: `outputs/offload-residency-v1-smoke`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `M00_phase_min-r1` | ok |  |  |  |  |  | 6.469 |
| `M03_optimizer_resident-r1` | ok |  |  |  |  |  | 9.684 |
| `M05_all_gpu-r1` | ok |  |  |  |  |  | 10.601 |

## offload-residency-v2-performance

- Path: `outputs/offload-residency-v2-performance`
- Runs: 9

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `M00_grouped_min-r1` | ok |  |  |  |  |  | 6.469 |
| `M00_grouped_min-r2` | ok |  |  |  |  |  | 6.469 |
| `M00_grouped_min-r3` | ok |  |  |  |  |  | 6.469 |
| `M01_actor_resident_cpu_adam-r1` | ok |  |  |  |  |  | 6.468 |
| `M01_actor_resident_cpu_adam-r2` | ok |  |  |  |  |  | 6.468 |
| `M01_actor_resident_cpu_adam-r3` | ok |  |  |  |  |  | 6.468 |
| `M02_actor_resident_gpu_adam-r1` | ok |  |  |  |  |  | 10.151 |
| `M02_actor_resident_gpu_adam-r2` | ok |  |  |  |  |  | 10.151 |
| `M02_actor_resident_gpu_adam-r3` | ok |  |  |  |  |  | 10.151 |

## offload-residency-v2-smoke

- Path: `outputs/offload-residency-v2-smoke`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `M00_grouped_min-r1` | ok |  |  |  |  |  | 6.469 |
| `M01_actor_resident_cpu_adam-r1` | ok |  |  |  |  |  | 6.468 |
| `M02_actor_resident_gpu_adam-r1` | ok |  |  |  |  |  | 9.684 |

## pa-activation-probe-fp32-v1

- Path: `outputs/pa-activation-probe-fp32-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-LATE-CCC-ACTOFF-r1` | failed | Phase offload / activation offload |  |  |  |  |  |
| `FP32-LATE-CCC-ACTON-r1` | ok | Phase offload / activation resident |  |  | 0.201 | 0.173 | 8.405 |

## pa-allocator-best-v1

- Path: `outputs/pa-allocator-best-v1`
- Runs: 4

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-CUDAASYNC-TRIM-r1` | ok | PA-ASYNC / cudaMallocAsync / forward trim | 64 | 2 | 0.952 | 5.090 | 3.208 |
| `PA-DIRECT1-EXPAND-r1` | ok | PA-ASYNC / 1 MiB groups / expandable segments | 1 | 2 | 1.116 | 4.988 | 3.208 |
| `PA-EXPAND-TRIM-r1` | ok | PA-ASYNC / expandable segments / forward trim | 64 | 2 | 0.880 | 4.424 | 3.208 |
| `PA-EXPAND-r1` | ok | PA-ASYNC / expandable segments | 64 | 2 | 0.926 | 4.435 | 3.208 |

## pa-allocator-native-confirm

- Path: `outputs/pa-allocator-native-confirm`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-NATIVE-r1` | ok | PA-ASYNC / native allocator | 64 | 2 | 0.922 | 4.999 | 3.235 |

## pa-allocator-opt-smoke

- Path: `outputs/pa-allocator-opt-smoke`
- Runs: 7

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-CUDAASYNC-r1` | ok | PA-ASYNC / cudaMallocAsync allocator | 64 | 2 | 0.901 | 4.419 | 3.208 |
| `PA-DIRECT1-EXPAND-r1` | ok | PA-ASYNC / 1 MiB groups / expandable segments | 1 | 2 | 1.309 | 5.222 | 3.208 |
| `PA-DIRECT1-TRIM-r1` | incomplete | PA-ASYNC / 1 MiB groups / forward segment trim | 1 | 2 | 1.226 | 5.237 | 3.236 |
| `PA-DIRECT1-r1` | ok | PA-ASYNC / 1 MiB groups / direct oversized-gradient D2H | 1 | 2 | 1.111 | 4.464 | 3.236 |
| `PA-EXPAND-r1` | ok | PA-ASYNC / expandable segments | 64 | 2 | 0.924 | 4.792 | 3.208 |
| `PA-FWD-TRIM-r1` | ok | PA-ASYNC / release inactive forward segments | 64 | 2 | 0.927 | 4.970 | 3.235 |
| `PA-NATIVE-r1` | ok | PA-ASYNC / native allocator | 64 | 2 | 0.905 | 4.412 | 3.235 |

## pa-allocator-opt-smoke-remain

- Path: `outputs/pa-allocator-opt-smoke-remain`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-DIRECT1-EXPAND-r1` | ok | PA-ASYNC / 1 MiB groups / expandable segments | 1 | 2 | 1.183 | 5.313 | 3.208 |
| `PA-DIRECT1-TRIM-r1` | ok | PA-ASYNC / 1 MiB groups / forward segment trim | 1 | 2 | 1.142 | 5.109 | 3.236 |

## pa-async-fixed2-memory-check

- Path: `outputs/pa-async-fixed2-memory-check`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-ASYNC-FIXED2-r1` | ok | Phase actor residency / two reusable GPU packing buffers | 64 | 2 | 0.802 | 5.004 | 3.360 |

## pa-async-fixed2-quick

- Path: `outputs/pa-async-fixed2-quick`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-ASYNC-FIXED2-r1` | ok | Phase actor residency / two reusable GPU packing buffers | 64 | 2 | 0.718 | 4.999 | 3.360 |

## pa-async-quick

- Path: `outputs/pa-async-quick`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-ASYNC-r1` | ok | Phase actor residency / async D2H CPU Adam | 64 | 2 | 0.930 | 5.021 | 3.235 |

## pa-capacity-fp32-qwen15b-bucket-sweep-v1

- Path: `outputs/pa-capacity-fp32-qwen15b-bucket-sweep-v1`
- Runs: 13

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-15B-CPU-STREAM08-S1-r1` | ok | Qwen 1.5B FP32 CPU Adam / 8 MiB / one slot | 8 | 1 | 3.453 | 12.590 | 8.377 |
| `FP32-15B-CPU-STREAM16-S1-r1` | ok | Qwen 1.5B FP32 CPU Adam / 16 MiB / one slot | 16 | 1 | 3.164 | 12.634 | 8.385 |
| `FP32-15B-CPU-STREAM16-S1-r2` | ok | Qwen 1.5B FP32 CPU Adam / 16 MiB / one slot | 16 | 1 | 3.225 | 12.563 | 8.385 |
| `FP32-15B-CPU-STREAM16-S1-r3` | ok | Qwen 1.5B FP32 CPU Adam / 16 MiB / one slot | 16 | 1 | 3.169 | 12.574 | 8.385 |
| `FP32-15B-CPU-STREAM32-S1-r1` | ok | Qwen 1.5B FP32 CPU Adam / 32 MiB / one slot | 32 | 1 | 3.081 | 12.580 | 8.397 |
| `FP32-15B-CPU-STREAM32-S1-r2` | ok | Qwen 1.5B FP32 CPU Adam / 32 MiB / one slot | 32 | 1 | 3.111 | 12.565 | 8.397 |
| `FP32-15B-CPU-STREAM32-S1-r3` | ok | Qwen 1.5B FP32 CPU Adam / 32 MiB / one slot | 32 | 1 | 3.118 | 12.593 | 8.397 |
| `FP32-15B-CPU-STREAM64-S1-r1` | ok | Qwen 1.5B FP32 CPU Adam / 64 MiB / one slot | 64 | 1 | 2.919 | 12.575 | 8.438 |
| `FP32-15B-CPU-STREAM64-S1-r2` | ok | Qwen 1.5B FP32 CPU Adam / 64 MiB / one slot | 64 | 1 | 2.497 | 12.610 | 8.438 |
| `FP32-15B-CPU-STREAM64-S1-r3` | ok | Qwen 1.5B FP32 CPU Adam / 64 MiB / one slot | 64 | 1 | 2.897 | 12.645 | 8.438 |
| `FP32-FINAL-CPU-STREAM128-S1-r1` | ok | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 | 2.830 | 12.651 | 8.499 |
| `FP32-FINAL-CPU-STREAM128-S1-r2` | ok | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 | 2.762 | 12.580 | 8.499 |
| `FP32-FINAL-CPU-STREAM128-S1-r3` | ok | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 | 2.838 | 12.605 | 8.499 |

## pa-capacity-fp32-qwen15b-nostream-oom-snapshot-v1

- Path: `outputs/pa-capacity-fp32-qwen15b-nostream-oom-snapshot-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-15B-CPU-NOSTREAM-OOM-SNAPSHOT-r1` | failed | Qwen 1.5B FP32 CPU Adam no-stream OOM snapshot |  |  |  |  |  |

## pa-capacity-fp32-qwen15b-nostream-oom-snapshot-v2

- Path: `outputs/pa-capacity-fp32-qwen15b-nostream-oom-snapshot-v2`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-15B-CPU-NOSTREAM-OOM-SNAPSHOT-r1` | failed | Qwen 1.5B FP32 CPU Adam no-stream OOM snapshot |  |  |  |  |  |

## pa-capacity-fp32-qwen15b-v1

- Path: `outputs/pa-capacity-fp32-qwen15b-v1`
- Runs: 5

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-BEST-r1` | ok | Partial Actor residency / CPU Adam / 8 MiB / one slot | 8 | 1 | 3.475 | 12.523 | 8.377 |
| `FP32-FINAL-CPU-BEST-r2` | ok | Partial Actor residency / CPU Adam / 8 MiB / one slot | 8 | 1 | 3.400 | 12.509 | 8.377 |
| `FP32-FINAL-CPU-BEST-r3` | ok | Partial Actor residency / CPU Adam / 8 MiB / one slot | 8 | 1 | 3.404 | 12.530 | 8.377 |
| `FP32-FINAL-CPU-NOSTREAM-r1` | failed | Partial Actor residency / CPU Adam / full GPU gradient |  |  |  |  |  |
| `FP32-FINAL-GPU-OPT-r1` | failed | Partial Actor residency / Reference phase / GPU Adam |  |  |  |  |  |

## pa-cpu-adam-all-phase-fp32-v1

- Path: `outputs/pa-cpu-adam-all-phase-fp32-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-ALL-PHASE-CPU-ADAM-r1` | ok | All common state phase offload / CPU-resident Adam / CPU update |  |  | 0.214 | 4.059 | 4.721 |
| `FP32-ALL-PHASE-GPU-ADAM-r1` | ok | All common state phase offload / Adam swap / GPU update |  |  | 0.217 | 0.128 | 8.403 |

## pa-cpu-adam-isolation-fp32-v1

- Path: `outputs/pa-cpu-adam-isolation-fp32-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-ALL-PHASE-CPU-ADAM-r1` | failed | All common state phase offload / CPU-resident Adam / CPU update |  |  |  |  |  |
| `FP32-ALL-PHASE-GPU-ADAM-r1` | failed | All common state phase offload / Adam swap / GPU update |  |  |  |  |  |

## pa-cpu-adam-isolation-fp32-v2

- Path: `outputs/pa-cpu-adam-isolation-fp32-v2`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-ALL-PHASE-CPU-ADAM-r1` | ok | All common state phase offload / CPU-resident Adam / CPU update |  |  | 0.216 | 3.642 | 4.721 |
| `FP32-ALL-PHASE-GPU-ADAM-r1` | ok | All common state phase offload / Adam swap / GPU update |  |  | 0.218 | 0.128 | 8.403 |

## pa-current-paired-quick-v1

- Path: `outputs/pa-current-paired-quick-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `NOSTREAM-CURRENT-QUICK-r1` | ok | No-stream current-code quick check |  |  | 0.209 | 3.574 | 4.721 |
| `STREAM16-S3-DIRECT-LAZY-r1` | ok | 16 MiB / 3 slots / direct CPU gradients / lazy packing buffers | 16 | 3 | 0.291 | 3.154 | 3.405 |

## pa-final-qwen15b-best-v1

- Path: `outputs/pa-final-qwen15b-best-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FINAL-CPU-BEST-r1` | failed | Phase residency / CPU Adam / 8 MiB async gradient streaming / one slot / expandable segments | 8 | 1 |  |  |  |

## pa-final-v1

- Path: `outputs/pa-final-v1`
- Runs: 15

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FINAL-CPU-BEST-r1` | ok | Phase residency / CPU Adam / 8 MiB async gradient streaming / one slot / expandable segments | 8 | 1 | 0.946 | 4.948 | 3.214 |
| `FINAL-CPU-BEST-r2` | ok | Phase residency / CPU Adam / 8 MiB async gradient streaming / one slot / expandable segments | 8 | 1 | 0.950 | 5.023 | 3.214 |
| `FINAL-CPU-BEST-r3` | ok | Phase residency / CPU Adam / 8 MiB async gradient streaming / one slot / expandable segments | 8 | 1 | 0.945 | 4.966 | 3.214 |
| `FINAL-CPU-NOSTREAM-r1` | ok | Phase residency / CPU Adam / full GPU gradient / expandable segments |  |  | 0.248 | 5.290 | 4.206 |
| `FINAL-CPU-NOSTREAM-r2` | ok | Phase residency / CPU Adam / full GPU gradient / expandable segments |  |  | 0.251 | 4.766 | 4.206 |
| `FINAL-CPU-NOSTREAM-r3` | ok | Phase residency / CPU Adam / full GPU gradient / expandable segments |  |  | 0.254 | 4.865 | 4.206 |
| `FINAL-CPU-PARETO-r1` | ok | Phase residency / CPU Adam / 32 MiB async gradient streaming / one slot / expandable segments | 32 | 1 | 0.901 | 4.549 | 3.231 |
| `FINAL-CPU-PARETO-r2` | ok | Phase residency / CPU Adam / 32 MiB async gradient streaming / one slot / expandable segments | 32 | 1 | 0.884 | 4.940 | 3.231 |
| `FINAL-CPU-PARETO-r3` | ok | Phase residency / CPU Adam / 32 MiB async gradient streaming / one slot / expandable segments | 32 | 1 | 0.874 | 5.008 | 3.231 |
| `FINAL-CPU-STREAM64-r1` | ok | Phase residency / CPU Adam / 64 MiB async gradient streaming / expandable segments | 64 | 2 | 0.710 | 4.531 | 3.332 |
| `FINAL-CPU-STREAM64-r2` | ok | Phase residency / CPU Adam / 64 MiB async gradient streaming / expandable segments | 64 | 2 | 0.719 | 4.365 | 3.332 |
| `FINAL-CPU-STREAM64-r3` | ok | Phase residency / CPU Adam / 64 MiB async gradient streaming / expandable segments | 64 | 2 | 0.722 | 5.030 | 3.332 |
| `FINAL-GPU-OPT-r1` | ok | Phase residency / GPU AdamW / expandable segments |  |  | 0.246 | 0.080 | 8.392 |
| `FINAL-GPU-OPT-r2` | ok | Phase residency / GPU AdamW / expandable segments |  |  | 0.252 | 0.079 | 8.392 |
| `FINAL-GPU-OPT-r3` | ok | Phase residency / GPU AdamW / expandable segments |  |  | 0.264 | 0.079 | 8.392 |

## pa-late-gpu-adam-30steps-v1

- Path: `outputs/pa-late-gpu-adam-30steps-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-LATE-CCC-r1` | ok | Actor phase / Reference phase / Adam just-in-time load |  |  | 0.212 | 0.128 | 8.408 |

## pa-memory-sweep-v1

- Path: `outputs/pa-memory-sweep-v1`
- Runs: 12

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-MEM-B01-S1-r1` | ok | PA memory sweep / 1 MiB / 1 D2H slot(s) | 1 | 1 | 1.170 | 4.787 | 3.208 |
| `PA-MEM-B01-S2-r1` | ok | PA memory sweep / 1 MiB / 2 D2H slot(s) | 1 | 2 | 1.137 | 4.995 | 3.208 |
| `PA-MEM-B04-S1-r1` | ok | PA memory sweep / 4 MiB / 1 D2H slot(s) | 4 | 1 | 1.006 | 5.033 | 3.211 |
| `PA-MEM-B04-S2-r1` | ok | PA memory sweep / 4 MiB / 2 D2H slot(s) | 4 | 2 | 1.039 | 5.124 | 3.214 |
| `PA-MEM-B08-S1-r1` | ok | PA memory sweep / 8 MiB / 1 D2H slot(s) | 8 | 1 | 0.974 | 4.990 | 3.214 |
| `PA-MEM-B08-S2-r1` | ok | PA memory sweep / 8 MiB / 2 D2H slot(s) | 8 | 2 | 0.954 | 5.042 | 3.221 |
| `PA-MEM-B16-S1-r1` | ok | PA memory sweep / 16 MiB / 1 D2H slot(s) | 16 | 1 | 0.928 | 5.015 | 3.214 |
| `PA-MEM-B16-S2-r1` | ok | PA memory sweep / 16 MiB / 2 D2H slot(s) | 16 | 2 | 0.947 | 5.043 | 3.221 |
| `PA-MEM-B32-S1-r1` | ok | PA memory sweep / 32 MiB / 1 D2H slot(s) | 32 | 1 | 0.911 | 4.372 | 3.231 |
| `PA-MEM-B32-S2-r1` | ok | PA memory sweep / 32 MiB / 2 D2H slot(s) | 32 | 2 | 0.822 | 4.431 | 3.254 |
| `PA-MEM-B64-S1-r1` | ok | PA memory sweep / 64 MiB / 1 D2H slot(s) | 64 | 1 | 0.726 | 4.999 | 3.270 |
| `PA-MEM-B64-S2-r1` | ok | PA memory sweep / 64 MiB / 2 D2H slot(s) | 64 | 2 | 0.719 | 4.705 | 3.332 |

## pa-optimized-bucket-sweep-v1

- Path: `outputs/pa-optimized-bucket-sweep-v1`
- Runs: 18

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM-OPT-B128-r1` | ok | Optimized streaming / 128 MiB / 3 slots | 128 | 3 | 0.286 | 3.680 | 3.735 |
| `STREAM-OPT-B128-r2` | ok | Optimized streaming / 128 MiB / 3 slots | 128 | 3 | 0.288 | 3.693 | 3.735 |
| `STREAM-OPT-B128-r3` | ok | Optimized streaming / 128 MiB / 3 slots | 128 | 3 | 0.286 | 3.692 | 3.735 |
| `STREAM-OPT-B16-r1` | ok | Optimized streaming / 16 MiB / 3 slots | 16 | 3 | 0.287 | 3.674 | 3.402 |
| `STREAM-OPT-B16-r2` | ok | Optimized streaming / 16 MiB / 3 slots | 16 | 3 | 0.297 | 3.672 | 3.402 |
| `STREAM-OPT-B16-r3` | ok | Optimized streaming / 16 MiB / 3 slots | 16 | 3 | 0.289 | 3.680 | 3.402 |
| `STREAM-OPT-B256-r1` | ok | Optimized streaming / 256 MiB / 3 slots | 256 | 3 | 0.287 | 3.200 | 4.119 |
| `STREAM-OPT-B256-r2` | ok | Optimized streaming / 256 MiB / 3 slots | 256 | 3 | 0.294 | 3.694 | 4.119 |
| `STREAM-OPT-B256-r3` | ok | Optimized streaming / 256 MiB / 3 slots | 256 | 3 | 0.292 | 3.676 | 4.119 |
| `STREAM-OPT-B32-r1` | ok | Optimized streaming / 32 MiB / 3 slots | 32 | 3 | 0.294 | 3.673 | 3.451 |
| `STREAM-OPT-B32-r2` | ok | Optimized streaming / 32 MiB / 3 slots | 32 | 3 | 0.291 | 3.686 | 3.451 |
| `STREAM-OPT-B32-r3` | ok | Optimized streaming / 32 MiB / 3 slots | 32 | 3 | 0.297 | 3.674 | 3.451 |
| `STREAM-OPT-B512-r1` | ok | Optimized streaming / 512 MiB / 3 slots | 512 | 3 | 0.295 | 3.684 | 5.126 |
| `STREAM-OPT-B512-r2` | ok | Optimized streaming / 512 MiB / 3 slots | 512 | 3 | 0.295 | 3.681 | 5.126 |
| `STREAM-OPT-B512-r3` | ok | Optimized streaming / 512 MiB / 3 slots | 512 | 3 | 0.293 | 3.678 | 5.126 |
| `STREAM-OPT-B64-r1` | ok | Optimized streaming / 64 MiB / 3 slots | 64 | 3 | 0.287 | 3.738 | 3.568 |
| `STREAM-OPT-B64-r2` | ok | Optimized streaming / 64 MiB / 3 slots | 64 | 3 | 0.289 | 3.748 | 3.568 |
| `STREAM-OPT-B64-r3` | ok | Optimized streaming / 64 MiB / 3 slots | 64 | 3 | 0.288 | 3.689 | 3.568 |

## pa-part-expand-control

- Path: `outputs/pa-part-expand-control`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PA-PART-EXPAND-r1` | ok | Phase actor residency / no streaming / expandable segments |  |  | 0.242 | 5.050 | 4.206 |

## pa-phase-local-peak-backward-lazy-v1

- Path: `outputs/pa-phase-local-peak-backward-lazy-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-STREAM64-S1-r1` | ok | Partial Actor residency / CPU Adam / 64 MiB / one slot | 64 | 1 | 0.690 | 3.626 | 2.365 |

## pa-phase-local-peak-lazy-buffer-v1

- Path: `outputs/pa-phase-local-peak-lazy-buffer-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-STREAM64-S1-r1` | ok | Partial Actor residency / CPU Adam / 64 MiB / one slot | 64 | 1 | 0.701 | 3.820 | 2.365 |

## pa-phase-local-peak-nostream-v1

- Path: `outputs/pa-phase-local-peak-nostream-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-NOSTREAM-r1` | ok | Partial Actor residency / CPU Adam / full GPU gradient |  |  | 0.210 | 4.080 | 4.213 |

## pa-phase-local-peak-nostream-v2

- Path: `outputs/pa-phase-local-peak-nostream-v2`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-NOSTREAM-r1` | failed | Partial Actor residency / CPU Adam / full GPU gradient |  |  | 0.212 | 4.151 | 4.721 |

## pa-phase-local-peak-nostream-v3

- Path: `outputs/pa-phase-local-peak-nostream-v3`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-NOSTREAM-r1` | failed | Partial Actor residency / CPU Adam / full GPU gradient |  |  | 0.205 | 4.096 | 3.705 |

## pa-phase-local-peak-nostream-v4

- Path: `outputs/pa-phase-local-peak-nostream-v4`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-NOSTREAM-r1` | ok | Partial Actor residency / CPU Adam / full GPU gradient |  |  | 0.210 | 3.559 | 2.365 |

## pa-phase-local-peak-v1

- Path: `outputs/pa-phase-local-peak-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-STREAM64-S1-r1` | ok | Partial Actor residency / CPU Adam / 64 MiB / one slot | 64 | 1 | 0.714 | 3.800 | 2.428 |

## pa-repro-fp32-bucket-extra-v1

- Path: `outputs/pa-repro-fp32-bucket-extra-v1`
- Runs: 6

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-STREAM128-S1-r1` | ok | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 | 0.664 | 3.713 | 3.499 |
| `FP32-FINAL-CPU-STREAM128-S1-r2` | ok | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 | 0.674 | 3.721 | 3.499 |
| `FP32-FINAL-CPU-STREAM128-S1-r3` | ok | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 | 0.686 | 3.723 | 3.499 |
| `FP32-FINAL-CPU-STREAM16-S1-r1` | ok | Partial Actor residency / CPU Adam / 16 MiB / one slot | 16 | 1 | 0.928 | 3.717 | 3.388 |
| `FP32-FINAL-CPU-STREAM16-S1-r2` | ok | Partial Actor residency / CPU Adam / 16 MiB / one slot | 16 | 1 | 0.928 | 3.688 | 3.388 |
| `FP32-FINAL-CPU-STREAM16-S1-r3` | ok | Partial Actor residency / CPU Adam / 16 MiB / one slot | 16 | 1 | 0.937 | 3.721 | 3.388 |

## pa-repro-fp32-late-optimizer-smoke

- Path: `outputs/pa-repro-fp32-late-optimizer-smoke`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-LATE-CCC-r1` | failed | Actor phase / Reference phase / Adam just-in-time load |  |  |  |  |  |

## pa-repro-fp32-late-optimizer-smoke-v2

- Path: `outputs/pa-repro-fp32-late-optimizer-smoke-v2`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-LATE-CCC-r1` | ok | Actor phase / Reference phase / Adam just-in-time load |  |  | 0.240 | 0.148 | 8.405 |
| `FP32-LATE-GGG-r1` | ok | Actor GPU / Reference GPU / Adam GPU |  |  | 0.238 | 0.109 | 10.245 |

## pa-repro-fp32-v1

- Path: `outputs/pa-repro-fp32-v1`
- Runs: 48

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-BEST-r1` | ok | Partial Actor residency / CPU Adam / 8 MiB / one slot | 8 | 1 | 0.904 | 3.691 | 3.388 |
| `FP32-FINAL-CPU-BEST-r2` | ok | Partial Actor residency / CPU Adam / 8 MiB / one slot | 8 | 1 | 0.931 | 3.681 | 3.388 |
| `FP32-FINAL-CPU-BEST-r3` | ok | Partial Actor residency / CPU Adam / 8 MiB / one slot | 8 | 1 | 0.930 | 3.701 | 3.388 |
| `FP32-FINAL-CPU-NOSTREAM-r1` | ok | Partial Actor residency / CPU Adam / full GPU gradient |  |  | 0.214 | 4.007 | 4.721 |
| `FP32-FINAL-CPU-NOSTREAM-r2` | ok | Partial Actor residency / CPU Adam / full GPU gradient |  |  | 0.212 | 4.074 | 4.721 |
| `FP32-FINAL-CPU-NOSTREAM-r3` | ok | Partial Actor residency / CPU Adam / full GPU gradient |  |  | 0.213 | 4.012 | 4.721 |
| `FP32-FINAL-CPU-PARETO-r1` | ok | Partial Actor residency / CPU Adam / 32 MiB / one slot | 32 | 1 | 0.879 | 3.703 | 3.404 |
| `FP32-FINAL-CPU-PARETO-r2` | ok | Partial Actor residency / CPU Adam / 32 MiB / one slot | 32 | 1 | 0.875 | 3.690 | 3.404 |
| `FP32-FINAL-CPU-PARETO-r3` | ok | Partial Actor residency / CPU Adam / 32 MiB / one slot | 32 | 1 | 0.852 | 3.688 | 3.404 |
| `FP32-FINAL-CPU-STREAM128-S1-r1` | failed | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 |  |  |  |
| `FP32-FINAL-CPU-STREAM128-S1-r2` | failed | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 |  |  |  |
| `FP32-FINAL-CPU-STREAM128-S1-r3` | failed | Partial Actor residency / CPU Adam / 128 MiB / one slot | 128 | 1 |  |  |  |
| `FP32-FINAL-CPU-STREAM16-S1-r1` | failed | Partial Actor residency / CPU Adam / 16 MiB / one slot | 16 | 1 |  |  |  |
| `FP32-FINAL-CPU-STREAM16-S1-r2` | failed | Partial Actor residency / CPU Adam / 16 MiB / one slot | 16 | 1 |  |  |  |
| `FP32-FINAL-CPU-STREAM16-S1-r3` | failed | Partial Actor residency / CPU Adam / 16 MiB / one slot | 16 | 1 |  |  |  |
| `FP32-FINAL-CPU-STREAM64-S1-r1` | ok | Partial Actor residency / CPU Adam / 64 MiB / one slot | 64 | 1 | 0.725 | 3.593 | 3.443 |
| `FP32-FINAL-CPU-STREAM64-S1-r2` | ok | Partial Actor residency / CPU Adam / 64 MiB / one slot | 64 | 1 | 0.731 | 3.707 | 3.443 |
| `FP32-FINAL-CPU-STREAM64-S1-r3` | ok | Partial Actor residency / CPU Adam / 64 MiB / one slot | 64 | 1 | 0.727 | 3.237 | 3.443 |
| `FP32-FINAL-CPU-STREAM64-r1` | ok | Partial Actor residency / CPU Adam / 64 MiB / two slots | 64 | 2 | 0.671 | 3.689 | 3.506 |
| `FP32-FINAL-CPU-STREAM64-r2` | ok | Partial Actor residency / CPU Adam / 64 MiB / two slots | 64 | 2 | 0.675 | 3.692 | 3.506 |
| `FP32-FINAL-CPU-STREAM64-r3` | ok | Partial Actor residency / CPU Adam / 64 MiB / two slots | 64 | 2 | 0.693 | 3.687 | 3.506 |
| `FP32-FINAL-GPU-OPT-r1` | ok | Partial Actor residency / Reference phase / GPU Adam |  |  | 0.225 | 0.079 | 8.406 |
| `FP32-FINAL-GPU-OPT-r2` | ok | Partial Actor residency / Reference phase / GPU Adam |  |  | 0.228 | 0.079 | 8.406 |
| `FP32-FINAL-GPU-OPT-r3` | ok | Partial Actor residency / Reference phase / GPU Adam |  |  | 0.225 | 0.079 | 8.406 |
| `FP32-R-CCC-r1` | ok | Actor phase / Reference phase / Adam swap |  |  | 0.227 | 0.129 | 8.403 |
| `FP32-R-CCC-r2` | ok | Actor phase / Reference phase / Adam swap |  |  | 0.220 | 0.127 | 8.403 |
| `FP32-R-CCC-r3` | ok | Actor phase / Reference phase / Adam swap |  |  | 0.222 | 0.127 | 8.403 |
| `FP32-R-CCG-r1` | ok | Actor phase / Reference phase / Adam GPU |  |  | 0.222 | 0.079 | 8.406 |
| `FP32-R-CCG-r2` | ok | Actor phase / Reference phase / Adam GPU |  |  | 0.218 | 0.079 | 8.406 |
| `FP32-R-CCG-r3` | ok | Actor phase / Reference phase / Adam GPU |  |  | 0.222 | 0.079 | 8.406 |
| `FP32-R-CGC-r1` | ok | Actor phase / Reference GPU / Adam swap |  |  | 0.226 | 0.128 | 10.246 |
| `FP32-R-CGC-r2` | ok | Actor phase / Reference GPU / Adam swap |  |  | 0.235 | 0.127 | 10.246 |
| `FP32-R-CGC-r3` | ok | Actor phase / Reference GPU / Adam swap |  |  | 0.236 | 0.127 | 10.246 |
| `FP32-R-CGG-r1` | ok | Actor phase / Reference GPU / Adam GPU |  |  | 0.230 | 0.079 | 10.248 |
| `FP32-R-CGG-r2` | ok | Actor phase / Reference GPU / Adam GPU |  |  | 0.229 | 0.079 | 10.248 |
| `FP32-R-CGG-r3` | ok | Actor phase / Reference GPU / Adam GPU |  |  | 0.222 | 0.079 | 10.248 |
| `FP32-R-GCC-r1` | ok | Actor GPU / Reference phase / Adam swap |  |  | 0.234 | 0.128 | 8.405 |
| `FP32-R-GCC-r2` | ok | Actor GPU / Reference phase / Adam swap |  |  | 0.238 | 0.128 | 8.405 |
| `FP32-R-GCC-r3` | ok | Actor GPU / Reference phase / Adam swap |  |  | 0.226 | 0.127 | 8.405 |
| `FP32-R-GCG-r1` | ok | Actor GPU / Reference phase / Adam GPU |  |  | 0.223 | 0.079 | 8.404 |
| `FP32-R-GCG-r2` | ok | Actor GPU / Reference phase / Adam GPU |  |  | 0.229 | 0.079 | 8.404 |
| `FP32-R-GCG-r3` | ok | Actor GPU / Reference phase / Adam GPU |  |  | 0.223 | 0.079 | 8.404 |
| `FP32-R-GGC-r1` | ok | Actor GPU / Reference GPU / Adam swap |  |  | 0.228 | 0.128 | 10.246 |
| `FP32-R-GGC-r2` | ok | Actor GPU / Reference GPU / Adam swap |  |  | 0.233 | 0.128 | 10.246 |
| `FP32-R-GGC-r3` | ok | Actor GPU / Reference GPU / Adam swap |  |  | 0.238 | 0.127 | 10.246 |
| `FP32-R-GGG-r1` | ok | Actor GPU / Reference GPU / Adam GPU |  |  | 0.210 | 0.079 | 10.242 |
| `FP32-R-GGG-r2` | ok | Actor GPU / Reference GPU / Adam GPU |  |  | 0.212 | 0.079 | 10.242 |
| `FP32-R-GGG-r3` | ok | Actor GPU / Reference GPU / Adam GPU |  |  | 0.216 | 0.079 | 10.242 |

## pa-saved-activation-probe-fp32-v1

- Path: `outputs/pa-saved-activation-probe-fp32-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-LATE-CCC-ACTON-r1` | ok | Phase offload / activation resident |  |  | 0.206 | 0.173 | 8.405 |

## pa-stream-bucket-sweep-nt-v1

- Path: `outputs/pa-stream-bucket-sweep-nt-v1`
- Runs: 4

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM128-S2-NT-r1` | ok | 128 MiB / 2 slots / no telemetry | 128 | 2 | 0.456 | 3.742 | 3.617 |
| `STREAM16-S2-NT-r1` | ok | 16 MiB / 2 slots / no telemetry | 16 | 2 | 0.534 | 3.758 | 3.394 |
| `STREAM32-S2-NT-r1` | ok | 32 MiB / 2 slots / no telemetry | 32 | 2 | 0.464 | 3.732 | 3.428 |
| `STREAM8-S2-NT-r1` | ok | 8 MiB / 2 slots / no telemetry | 8 | 2 | 0.502 | 3.696 | 3.394 |

## pa-stream-direct-final-sweep-v1

- Path: `outputs/pa-stream-direct-final-sweep-v1`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM128-S3-DIRECT-NT-r1` | ok | 128 MiB / 3 slots / direct CPU gradients | 128 | 3 | 0.278 | 3.459 | 3.735 |
| `STREAM128-S4-DIRECT-NT-r1` | ok | 128 MiB / 4 slots / direct CPU gradients | 128 | 4 | 0.270 | 3.678 | 3.853 |
| `STREAM256-S2-DIRECT-NT-r1` | ok | 256 MiB / 2 slots / direct CPU gradients | 256 | 2 | 0.306 | 3.684 | 3.873 |

## pa-stream-foreach-test-v1

- Path: `outputs/pa-stream-foreach-test-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM128-S3-DIRECT-NT-r1` | ok | 128 MiB / 3 slots / direct CPU gradients | 128 | 3 | 0.275 | 3.695 | 3.735 |

## pa-stream-optimized-final-v1

- Path: `outputs/pa-stream-optimized-final-v1`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM128-S3-DIRECT-NT-r1` | ok | 128 MiB / 3 slots / direct CPU gradients | 128 | 3 | 0.288 | 3.422 | 3.735 |
| `STREAM128-S3-DIRECT-NT-r2` | ok | 128 MiB / 3 slots / direct CPU gradients | 128 | 3 | 0.284 | 3.704 | 3.735 |
| `STREAM128-S3-DIRECT-NT-r3` | ok | 128 MiB / 3 slots / direct CPU gradients | 128 | 3 | 0.285 | 3.693 | 3.735 |

## pa-stream-optimized-phase-v1

- Path: `outputs/pa-stream-optimized-phase-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM128-S3-DIRECT-DIAG-r1` | ok | 128 MiB / 3 slots / direct CPU gradients / diagnostic | 128 | 3 | 0.356 | 3.710 | 3.735 |

## pa-stream128-direct-nt-v1

- Path: `outputs/pa-stream128-direct-nt-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM128-S2-DIRECT-NT-r1` | ok | 128 MiB / 2 slots / direct CPU gradients | 128 | 2 | 0.287 | 3.692 | 3.617 |

## pa-stream16-lazy-final-v1

- Path: `outputs/pa-stream16-lazy-final-v1`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM16-S3-DIRECT-LAZY-r1` | ok | 16 MiB / 3 slots / direct CPU gradients / lazy packing buffers | 16 | 3 | 0.284 | 3.180 | 3.402 |
| `STREAM16-S3-DIRECT-LAZY-r2` | ok | 16 MiB / 3 slots / direct CPU gradients / lazy packing buffers | 16 | 3 | 0.290 | 3.137 | 3.402 |
| `STREAM16-S3-DIRECT-LAZY-r3` | ok | 16 MiB / 3 slots / direct CPU gradients / lazy packing buffers | 16 | 3 | 0.290 | 3.689 | 3.401 |

## pa-stream16-lazy-phase-v1

- Path: `outputs/pa-stream16-lazy-phase-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM16-S3-DIRECT-LAZY-DIAG-r1` | ok | 16 MiB / 3 slots / direct CPU gradients / lazy diagnostic | 16 | 3 | 0.549 | 3.748 | 3.401 |

## pa-stream64-2slot-no-telemetry-quick-v1

- Path: `outputs/pa-stream64-2slot-no-telemetry-quick-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-STREAM64-S2-NOTELEMETRY-r1` | ok | 64 MiB two-slot streaming without detailed telemetry | 64 | 2 | 0.470 | 3.730 | 3.506 |

## pa-stream64-no-telemetry-quick-v1

- Path: `outputs/pa-stream64-no-telemetry-quick-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `FP32-FINAL-CPU-STREAM64-S1-NOTELEMETRY-r1` | ok | 64 MiB one-slot streaming without detailed telemetry | 64 | 1 | 0.490 | 3.746 | 3.443 |

## phase-best-vs-cpu-adamw-memory-v1

- Path: `outputs/phase-best-vs-cpu-adamw-memory-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PHASE-BEST-GPU-ADAMW-r1` | ok | Phase offload best / late-load GPU AdamW |  |  | 0.213 | 0.128 | 8.408 |
| `PHASE-BEST-PLUS-CPU-ADAMW-r1` | ok | Phase offload best / CPU AdamW |  |  | 0.214 | 3.557 | 2.365 |

## phase-best-vs-cpu-adamw-performance-v1

- Path: `outputs/phase-best-vs-cpu-adamw-performance-v1`
- Runs: 6

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `PHASE-BEST-GPU-ADAMW-r1` | ok | Phase offload best / late-load GPU AdamW |  |  | 0.212 | 0.130 | 8.408 |
| `PHASE-BEST-GPU-ADAMW-r2` | ok | Phase offload best / late-load GPU AdamW |  |  | 0.213 | 0.129 | 8.408 |
| `PHASE-BEST-GPU-ADAMW-r3` | ok | Phase offload best / late-load GPU AdamW |  |  | 0.217 | 0.129 | 8.408 |
| `PHASE-BEST-PLUS-CPU-ADAMW-r1` | ok | Phase offload best / CPU AdamW |  |  | 0.220 | 3.533 | 4.721 |
| `PHASE-BEST-PLUS-CPU-ADAMW-r2` | ok | Phase offload best / CPU AdamW |  |  | 0.215 | 3.591 | 4.721 |
| `PHASE-BEST-PLUS-CPU-ADAMW-r3` | ok | Phase offload best / CPU AdamW |  |  | 0.219 | 3.570 | 4.721 |

## rootcause-abc-sequential-gpu1-v1

- Path: `outputs/rootcause-abc-sequential-gpu1-v1`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `A-NOSTREAM-TELEMETRY-r1` | ok | A no-stream serial Adam and H2D |  |  | 0.224 | 3.579 | 4.721 |
| `B-STREAM16-SERIAL-TELEMETRY-r1` | ok | B stream D2H, serial Adam and H2D | 16 | 3 | 0.576 | 3.716 | 3.401 |
| `C-STREAM16-PIPELINE-TELEMETRY-r1` | ok | C stream D2H, bucket Adam plus H2D overlap | 16 | 3 | 0.574 | 3.407 | 3.401 |

## rootcause-abc-v1

- Path: `outputs/rootcause-abc-v1`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `A-NOSTREAM-TELEMETRY-r1` | ok | A no-stream serial Adam and H2D |  |  | 0.228 | 3.907 | 4.721 |
| `B-STREAM16-SERIAL-TELEMETRY-r1` | ok | B stream D2H, serial Adam and H2D | 16 | 3 | 0.753 | 4.090 | 3.401 |
| `C-STREAM16-PIPELINE-TELEMETRY-r1` | ok | C stream D2H, bucket Adam plus H2D overlap | 16 | 3 | 0.681 | 3.792 | 3.401 |

## stream16-pipeline-performance-3x-gpu1-v1

- Path: `outputs/stream16-pipeline-performance-3x-gpu1-v1`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM16-S3-PIPELINE-PERF-r1` | ok | 16 MiB streaming with Adam-H2D overlap | 16 | 3 | 0.289 | 3.016 | 3.401 |
| `STREAM16-S3-PIPELINE-PERF-r2` | ok | 16 MiB streaming with Adam-H2D overlap | 16 | 3 | 0.290 | 3.028 | 3.402 |
| `STREAM16-S3-PIPELINE-PERF-r3` | ok | 16 MiB streaming with Adam-H2D overlap | 16 | 3 | 0.298 | 3.017 | 3.403 |

## stream16-pipeline-performance-once-gpu1-v1

- Path: `outputs/stream16-pipeline-performance-once-gpu1-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM16-S3-PIPELINE-PERF-r1` | ok | 16 MiB streaming with Adam-H2D overlap | 16 | 3 | 0.300 | 3.028 | 3.402 |

## stream16-s3-fresh-clone-telemetry-once-gpu1-v1

- Path: `outputs/stream16-s3-fresh-clone-telemetry-once-gpu1-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM16-S3-FRESH-CLONE-TELEMETRY-ONCE-r1` | ok | 16 MiB / 3 slots / fresh pageable gradient clone / telemetry | 16 | 3 | 0.572 | 3.731 | 3.401 |

## stream16-s3-materialized-telemetry-once-gpu1-v1

- Path: `outputs/stream16-s3-materialized-telemetry-once-gpu1-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM16-S3-MATERIALIZED-TELEMETRY-ONCE-r1` | ok | 16 MiB / 3 slots / persistent pageable gradients / telemetry | 16 | 3 | 0.578 | 3.709 | 3.401 |

## stream16-s3-materialized-telemetry-once-v1

- Path: `outputs/stream16-s3-materialized-telemetry-once-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM16-S3-MATERIALIZED-TELEMETRY-ONCE-r1` | failed | 16 MiB / 3 slots / persistent pageable gradients / telemetry | 16 | 3 |  |  |  |

## stream16-s3-pageable-telemetry-once-v1

- Path: `outputs/stream16-s3-pageable-telemetry-once-v1`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `STREAM16-S3-PAGEABLE-TELEMETRY-ONCE-r1` | ok | 16 MiB / 3 slots / pageable master gradients / telemetry | 16 | 3 | 0.889 | 3.716 | 3.401 |

## zero-offload-z00-smoke

- Path: `outputs/zero-offload-z00-smoke`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `Z00-r1` | failed |  |  |  |  |  |  |

## zero-offload-z01-z03-smoke

- Path: `outputs/zero-offload-z01-z03-smoke`
- Runs: 3

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `Z01-r1` | ok |  |  |  |  |  |  |
| `Z02-r1` | ok |  |  |  |  |  |  |
| `Z03-r1` | ok |  | 64 |  |  |  |  |

## zero-offload-z01-z03-smoke-v2

- Path: `outputs/zero-offload-z01-z03-smoke-v2`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `Z02-r1` | ok |  |  |  |  |  |  |

## zero-offload-z03m-z04-precision-v1

- Path: `outputs/zero-offload-z03m-z04-precision-v1`
- Runs: 2

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `Z03M-r1` | ok |  | 64 |  | 0.432 | 6.216 | 5.185 |
| `Z04-r1` | ok |  | 64 | 2 | 1.237 | 4.982 | 4.171 |

## zero-offload-z04-z06-smoke

- Path: `outputs/zero-offload-z04-z06-smoke`
- Runs: 4

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `Z03M-r1` | ok |  | 64 |  |  |  |  |
| `Z04-r1` | ok |  | 64 | 2 |  |  |  |
| `Z05-r1` | ok |  | 64 | 2 |  |  |  |
| `Z06-r1` | ok |  | 64 | 2 |  |  |  |

## zero-offload-z06-stability-smoke

- Path: `outputs/zero-offload-z06-stability-smoke`
- Runs: 1

| run | status | label | bucket | slots | backward s | update s | peak GiB |
|---|---|---|---:|---:|---:|---:|---:|
| `Z06-r1` | ok |  | 64 | 2 | 2.491 | 3.561 | 3.899 |

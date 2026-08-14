# Experiment Design and Reproduction Guide

This document explains how the study was constructed, how intermediate
results should be interpreted, and how to reproduce the main comparisons. It
is intentionally more operational than the root README. For the complete
chronology of all 95 experiment groups and 512 runs, see the
[full experiment history](../reports/result-data/experiment-history/EXPERIMENT_HISTORY.md).

## 1. Research objectives

The study separates four questions that are easy to conflate:

1. Which tensors or states cause each GRPO phase peak?
2. How much memory is removed by phase-exclusive Actor/Reference residency?
3. What changes when AdamW execution, not only Adam state residency, moves to
   CPU?
4. Can gradient D2H, CPU Adam, and updated-parameter H2D be pipelined enough to
   recover the serial CPU optimizer penalty?

Each question uses a matched control. Results from different instrumentation
levels or optimizer boundaries are not mixed.

## 2. Experimental phases and memory ownership

| Phase | GPU work | Expected dominant live state |
|---|---|---|
| Rollout | Actor generation | Actor parameters, KV/rollout workspace |
| Actor log-prob | Actor scoring | Actor parameters and forward workspace |
| Reference log-prob | Reference scoring | Reference parameters and workspace |
| Actor forward | PPO loss forward | Actor parameters and saved activations |
| Actor backward | autograd | Actor parameters, activations, gradients |
| Update | AdamW and parameter reload | optimizer state or CPU/H2D pipeline |

The phase-offload schedule removes the inactive model before loading the model
needed by the next phase. The CPU path retains FP32 master parameters and
Adam states on host memory.

## 3. Implementation progression

### 3.1 Placement ablation

The initial C/G/M matrices measured Actor, Reference, optimizer-state, and
activation placement. These runs established two points:

- Reference phase offload is a predictable memory/time trade-off.
- Merely storing optimizer state on CPU does not remove the Update peak when it
  must be loaded back for GPU AdamW.

The older matrix analysis remains traceable in the complete experiment-history
inventory, but it is not presented as a primary study result.

### 3.2 Phase-local measurement correction

Early plots accidentally mixed cumulative CUDA peaks with phase-local peaks.
The benchmark was changed to reset the peak at phase boundaries. Gradients were
also released before measuring the optimizer-only Update peak. Failed v2/v3
probes were superseded by the successful no-stream v4 measurement.

Consequences:

- old Update bars that still contained gradients are not optimizer-only peaks;
- phase-local `allocated` values must not be replaced with allocator `reserved`
  or `nvidia-smi` device-used values;
- result figures use the corrected phase boundary.

### 3.3 CPU AdamW isolation

The matched experiment keeps phase placement fixed and changes only the update
implementation:

- late-load GPU AdamW: GPU optimizer state exists only near Update;
- CPU AdamW: FP32 master parameter, gradient, and Adam state remain on CPU.

This produces the isolated 8.408 -> 2.365 GiB Update-memory reduction, while
showing the serial CPU cost of 3.565 seconds.

### 3.4 Bounded gradient streaming

Backward hooks observe when parameter gradients become ready. The engine:

1. packs ready gradients into a bounded GPU bucket;
2. enqueues asynchronous D2H on a dedicated CUDA stream;
3. retains the staging slot until its CUDA event completes;
4. releases source GPU gradients early;
5. accumulates into direct CPU gradient buffers.

Lazy packing-buffer allocation was added after early streaming runs increased
Rollout memory even though streaming should only affect backward. This keeps
streaming buffers out of earlier phases.

### 3.5 Slot and bucket tuning

Quick 64/128 MiB tests were followed by direct-buffer and slot experiments.
Three slots were selected for the 16 MiB pipeline because they balance
producer progress and bounded pinned memory under that bucket layout. It is not
a universal optimum independent of bucket size or hardware.

The available 0.5B sweep covers 16, 32, 64, 128, 256, and 512 MiB with three
runs each. It is useful for backward/memory trends, but its
`overlap_h2d_with_cpu_update` setting is **false**. It cannot establish the
optimal bucket for the overlap-enabled pipeline.

### 3.6 Adam–H2D overlap root cause

An apparently contradictory result showed streaming Update slower than
no-stream. The direct timer was correct for that run: the matrix had gradient
streaming enabled but Adam-to-parameter H2D overlap disabled.

The controlled A/B/C sequence used:

- A: no-stream, serial CPU Adam/H2D;
- B: 16 MiB gradient stream, serial CPU Adam/H2D;
- C: 16 MiB stream, bucket CPU Adam plus H2D overlap.

On the same GPU with diagnostic telemetry, Update was 3.579, 3.716, and 3.407
seconds respectively. Telemetry-off repetitions reduced the optimized
16 MiB Update to 3.020 seconds.

### 3.7 Backward overhead

The optimized stream increases backward from 0.212 to 0.292 seconds. Recorded
evidence includes approximately 41 ms/step of staging-slot backpressure and
about 154 ms of total D2H CUDA activity that overlaps other work. The remainder
is consistent with hook dispatch, packing, event/stream enqueue, early release,
and memory-bandwidth contention.

The no-stream Nsight trace completed, but the matching streaming trace did not.
Therefore, this study does not claim a complete kernel-level attribution of
the remaining overhead.

## 4. Measurement protocol

### Performance runs

- disable `--detail`, JSON transfer telemetry, forced synchronization, and
  Nsight;
- use the same GPU and run competing configurations sequentially;
- exclude warm-up steps;
- use direct `perf/actor_backward_total_wall_s` and
  `perf/actor_adam_step_total_wall_s` metrics;
- repeat the reported comparison three times.

### Memory runs

- enable phase-local instrumentation separately from performance runs;
- reset the CUDA peak at the beginning of each phase;
- report PyTorch peak allocated memory as the tensor-lifecycle metric;
- report reserved/device-used only as separately labeled allocator/capacity
  metrics.

### Failures and OOM

Failed runs never enter performance means. An OOM remains valid evidence when
the research question is whether a configuration fits. The 1.5B no-stream OOM
stdout records 11.34 GiB allocated on a CUDA-reported 11.90 GiB device.

## 5. Environment setup

```bash
git clone --recurse-submodules https://github.com/2Ju1/verl-research.git
cd verl-research

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r constraints-cu118.txt \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-lock.txt
pip install -e src/verl

python src/verl/examples/data_preprocess/gsm8k.py \
  --local_save_dir "$PWD/data/gsm8k"
```

For the original workstation layout, replace `.venv/bin/python` below with
`envs/verl-titan/bin/python`. Set `CUDA_VISIBLE_DEVICES` explicitly when more
than one GPU is visible.

## 6. Reproduction commands

### 6.1 GPU AdamW versus CPU AdamW performance

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  src/verl/benchmarks/offload/run_matrix.py \
  --matrix benchmarks/offload/configs/phase_best_vs_cpu_adamw.json \
  --output outputs/phase-best-vs-cpu-adamw-performance-v1 \
  --repeats 3 \
  --warmup-steps 2
```

Collect phase-local memory separately:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  src/verl-phase-peak/benchmarks/offload/run_matrix.py \
  --matrix benchmarks/offload/configs/phase_best_vs_cpu_adamw.json \
  --output outputs/phase-best-vs-cpu-adamw-memory-v1 \
  --repeats 1 \
  --warmup-steps 2 \
  --detail
```

`src/verl-phase-peak` was the measurement worktree used for corrected
phase-local probes and is intentionally not tracked by the parent repository.
For a fresh clone, reproduce those probes from the corresponding phase-peak
instrumentation commit/worktree or use the curated memory results. This is a
known reproducibility limitation, not an interchangeable path alias.

### 6.2 Historical bucket sweep

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  src/verl/benchmarks/offload/run_matrix.py \
  --matrix reports/result-data/collected/05_bucket_size_sweep_05b/optimized_bucket_sweep.json \
  --output outputs/pa-optimized-bucket-sweep-v1 \
  --repeats 3 \
  --warmup-steps 2
```

This matrix tests 16/32/64/128/256/512 MiB, three staging slots, asynchronous
D2H, early release, reusable packing buffers, and direct CPU gradient buffers.
Adam–H2D overlap is off.

### 6.3 Regenerate the full local inventory

After new runs are added under `outputs/`:

```bash
.venv/bin/python \
  reports/result-data/build_experiment_inventory.py
```

This updates the run/group machine-readable indexes. It does not overwrite the
human-authored experiment history.

### 6.4 Regenerate tracked figures

```bash
MPLCONFIGDIR=/tmp/verl-result-figures \
  .venv/bin/python reports/figures/plot_allgpu_vs_phase_offload.py

MPLCONFIGDIR=/tmp/verl-result-figures \
  .venv/bin/python reports/figures/plot_results.py
```

Figure-specific raw inputs and plotting scripts are listed in the
[result-data manifest](../reports/result-data/README.md).

## 7. Evidence map

| Claim | Primary tracked evidence |
|---|---|
| Six-phase placement memory | `reports/result-data/collected/02_allgpu_vs_phase_offload/` |
| GPU versus CPU AdamW | `reports/result-data/collected/01_phase_offload_vs_cpu_adamw/` |
| No-stream versus optimized 16 MiB | `reports/result-data/collected/03_nostream_vs_stream16/` |
| 1.5B OOM/success boundary | `reports/result-data/collected/04_qwen15b_capacity/` |
| 0.5B bucket sweep | `reports/result-data/collected/05_bucket_size_sweep_05b/` |
| Figure provenance | `reports/result-data/manifest.csv` |
| Every local run | `reports/result-data/experiment-history/all_runs.csv` |
| Trial-and-error chronology | `reports/result-data/experiment-history/EXPERIMENT_HISTORY.md` |

## 8. Known limitations

- The primary study uses one GPU model and small batch/sequence settings.
- The 16--512 MiB sweep with Adam–H2D overlap enabled has not been run.
- The exact remaining backward overhead lacks a completed paired Nsight trace.
- Some early reports use FP16 or older allocator configurations; their results
  are historical diagnostics, not replacements for the reported FP32 comparison.
- The corrected memory-only runner lived in a separate local worktree; its
  curated results are tracked, but that worktree path is not portable.

These limitations are also recorded in the experiment history so that an
incomplete diagnostic cannot silently become a reported performance claim.

#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/sda/juwon/verl-research
DEST="$ROOT/reports/result-data/collected"

mkdir -p \
  "$DEST/01_phase_offload_vs_cpu_adamw/performance" \
  "$DEST/01_phase_offload_vs_cpu_adamw/memory" \
  "$DEST/02_allgpu_vs_phase_offload" \
  "$DEST/03_nostream_vs_stream16/nostream" \
  "$DEST/03_nostream_vs_stream16/stream16_overlap" \
  "$DEST/04_qwen15b_capacity/all_gpu_oom" \
  "$DEST/04_qwen15b_capacity/cpu_adam_nostream_oom" \
  "$DEST/04_qwen15b_capacity/cpu_adam_stream_success" \
  "$DEST/04_qwen15b_capacity/bucket_sweep" \
  "$DEST/05_bucket_size_sweep_05b/runs"

copy_run_files() {
  local source=$1
  local destination=$2
  mkdir -p "$destination"
  for name in run.json result.json stdout.log; do
    if [[ -f "$source/$name" ]]; then
      cp -p "$source/$name" "$destination/$name"
    fi
  done
}

for source in "$ROOT"/outputs/phase-best-vs-cpu-adamw-performance-v1/PHASE-BEST-*-r*; do
  copy_run_files "$source" "$DEST/01_phase_offload_vs_cpu_adamw/performance/$(basename "$source")"
done
for source in "$ROOT"/outputs/phase-best-vs-cpu-adamw-memory-v1/PHASE-BEST-*-r*; do
  copy_run_files "$source" "$DEST/01_phase_offload_vs_cpu_adamw/memory/$(basename "$source")"
done

cp -p \
  "$ROOT/outputs/pa-repro-fp32-late-optimizer-smoke-v2/summary/phase_configs.csv" \
  "$ROOT/outputs/pa-repro-fp32-late-optimizer-smoke-v2/summary/actor_subphase_configs.csv" \
  "$DEST/02_allgpu_vs_phase_offload/"
for source in "$ROOT"/outputs/pa-repro-fp32-late-optimizer-smoke-v2/FP32-LATE-GGG-r* \
              "$ROOT"/outputs/pa-repro-fp32-late-optimizer-smoke-v2/FP32-LATE-CCC-r*; do
  if [[ -d "$source" ]]; then
    copy_run_files "$source" "$DEST/02_allgpu_vs_phase_offload/$(basename "$source")"
  fi
done

for source in "$ROOT"/outputs/nostream-vs-16mib-direct-remeasure-v1/NOSTREAM-DIRECT-REMEASURE-r*; do
  copy_run_files "$source" "$DEST/03_nostream_vs_stream16/nostream/$(basename "$source")"
done
for source in "$ROOT"/outputs/stream16-pipeline-performance-3x-gpu1-v1/STREAM16-S3-PIPELINE-PERF-r*; do
  copy_run_files "$source" "$DEST/03_nostream_vs_stream16/stream16_overlap/$(basename "$source")"
done

copy_run_files \
  "$ROOT/outputs/pa-capacity-fp32-qwen15b-v1/FP32-FINAL-GPU-OPT-r1" \
  "$DEST/04_qwen15b_capacity/all_gpu_oom/FP32-FINAL-GPU-OPT-r1"
copy_run_files \
  "$ROOT/outputs/pa-capacity-fp32-qwen15b-nostream-oom-snapshot-v2/FP32-15B-CPU-NOSTREAM-OOM-SNAPSHOT-r1" \
  "$DEST/04_qwen15b_capacity/cpu_adam_nostream_oom/FP32-15B-CPU-NOSTREAM-OOM-SNAPSHOT-r1"
for source in "$ROOT"/outputs/pa-capacity-fp32-qwen15b-v1/FP32-FINAL-CPU-BEST-r*; do
  copy_run_files "$source" "$DEST/04_qwen15b_capacity/cpu_adam_stream_success/$(basename "$source")"
done
cp -p "$ROOT"/outputs/pa-capacity-fp32-qwen15b-v1/summary/*.csv \
  "$DEST/04_qwen15b_capacity/cpu_adam_stream_success/"
for source in "$ROOT"/outputs/pa-capacity-fp32-qwen15b-bucket-sweep-v1/FP32-*-r*; do
  copy_run_files "$source" "$DEST/04_qwen15b_capacity/bucket_sweep/$(basename "$source")"
done

for source in "$ROOT"/outputs/pa-optimized-bucket-sweep-v1/STREAM-OPT-B*-r*; do
  copy_run_files "$source" "$DEST/05_bucket_size_sweep_05b/runs/$(basename "$source")"
done
cp -p "$ROOT/tmp/optimized_bucket_sweep.json" \
  "$DEST/05_bucket_size_sweep_05b/optimized_bucket_sweep.json"

cp -p "$ROOT/reports/result-data/manifest.csv" "$DEST/manifest.csv"

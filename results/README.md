# Results and data

현재 연구에서 채택한 결과 그림, 원본 측정값, 재생성 코드를 한곳에 정리한다.
중간 진단과 폐기된 시행착오 데이터는 포함하지 않는다.

## Directory layout

```text
results/
├── figures/    핵심 결과 그림 4개 (PNG)
├── plotting/   그림 재생성 코드
├── data/       채택 결과의 설정, 측정값과 로그
└── manifest.csv
```

## 1. All-on-GPU vs phase offload

- Figure: `figures/allgpu_vs_phase_offload.png`
- Script: `plotting/plot_allgpu_vs_phase_offload.py`
- Data: `data/02_allgpu_vs_phase_offload/`
- Main result: Actor backward 10.24→4.72 GiB, Update 10.24→8.40 GiB

## 2. GPU AdamW vs CPU AdamW

- Figure: `figures/cpu_adamw.png`
- Script: `plotting/plot_results.py`
- Data: `data/01_phase_offload_vs_cpu_adamw/`

| configuration | Update peak | Update time |
|---|---:|---:|
| Phase offload + GPU AdamW | 8.408 GiB | 0.129 s |
| Phase offload + CPU AdamW | 2.365 GiB | 3.565 s |

## 3. No-stream vs optimized 16 MiB streaming

- Figure: `figures/gradient_streaming.png`
- Script: `plotting/plot_results.py`
- Data: `data/03_nostream_vs_stream16/`

| configuration | backward peak | backward time | Update time | step time |
|---|---:|---:|---:|---:|
| No-stream | 4.721 GiB | 0.212 s | 3.551 s | 9.814 s |
| 16 MiB, 3 slots, Adam–H2D overlap | 3.402 GiB | 0.292 s | 3.020 s | 8.629 s |

## 4. Qwen2.5-1.5B capacity

- Figure: `figures/qwen15b_capacity.png`
- Script: `plotting/plot_results.py`
- Data: `data/04_qwen15b_capacity/`
- All-GPU: OOM
- CPU AdamW without streaming: OOM during backward gradient accumulation
- CPU AdamW with streaming: success, 8.44 GiB phase-local peak

## 5. Bucket-size sweep

- Matrix and data: `data/05_bucket_size_sweep_05b/`
- Buckets: 16, 32, 64, 128, 256, 512 MiB; three runs each
- Limitation: `overlap_h2d_with_cpu_update=false`; retained for bucket and
  backward-memory analysis, not as the optimized timing comparison.

## Regenerate figures

```bash
MPLCONFIGDIR=/tmp/verl-result-figures \
  envs/verl-titan/bin/python results/plotting/plot_allgpu_vs_phase_offload.py

MPLCONFIGDIR=/tmp/verl-result-figures \
  envs/verl-titan/bin/python results/plotting/plot_results.py
```

The exact figure-to-data mapping is in `manifest.csv`.

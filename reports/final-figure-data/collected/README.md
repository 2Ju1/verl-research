# Collected final experiment data

최종 슬라이드의 네 실험에 사용된 최소 데이터 사본이다.

- `01_phase_offload_vs_cpu_adamw`: GPU AdamW/CPU AdamW 성능 3회 및 memory run
- `02_allgpu_vs_phase_offload`: GGG/CCC raw run과 phase 집계 CSV
- `03_nostream_vs_stream16`: no-stream 3회와 최종 16 MiB overlap 3회
- `04_qwen15b_capacity`: All-GPU OOM, CPU Adam no-stream OOM, streaming 성공 3회, bucket sweep
- `05_bucket_size_sweep_05b`: 0.5B 16/32/64/128/256/512 MiB 각 3회

각 raw run에는 그래프와 수치 검증에 필요한 `run.json`, `result.json`,
`stdout.log`만 포함한다. 대용량 `nvidia-smi.csv`, Nsight trace, 임시 Ray 로그는
의도적으로 제외했다. 원본 위치와 의미는 상위 `../README.md`와 `manifest.csv`를
참조한다.

주의: `05_bucket_size_sweep_05b`는 당시 최적화된 D2H streaming sweep이지만
`overlap_h2d_with_cpu_update=false`인 실험이다. 따라서 최종 16 MiB
Adam-H2D overlap 결과와 Update 시간을 직접 섞어 비교하면 안 된다.

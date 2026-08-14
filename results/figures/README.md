# Primary result figures

현재 연구 결과를 설명하는 핵심 그림만 PNG 형식으로 모은 디렉터리다.

| Figure | Comparison |
|---|---|
| `allgpu_vs_phase_offload.png` | All-on-GPU와 phase offload의 phase별 GPU memory |
| `cpu_adamw.png` | GPU AdamW와 CPU AdamW의 Update memory/time |
| `gradient_streaming.png` | No-stream과 최적화된 16 MiB gradient streaming |
| `qwen15b_capacity.png` | Qwen2.5-1.5B의 OOM 및 학습 성공 경계 |

그림의 데이터 출처는 [`../README.md`](../README.md), 생성 코드는
[`../plotting/`](../plotting/)에서 확인할 수 있다.

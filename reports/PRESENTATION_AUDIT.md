# Final Presentation Consistency Audit

검토 기준은 `outputs/0814.pdf` 25 pages와 최종 발표 대본이다. PDF는 raw output
디렉터리에 있어 Git이 추적하지 않지만, 해당 발표의 흐름과 최종 raw run을
GitHub 문서의 기준으로 사용했다.

교정된 25-page speaker notes는 `FINAL_PRESENTATION_SCRIPT_KO.md`에 보존한다.

## 수정한 충돌

| 항목 | 기존 상태 | 처리 |
|---|---|---|
| 용어 | 일부 문서가 학습 내부 구간을 단계/Stage/Actor update로 혼용 | 학습 내부 구간은 phase, 전체 반복 단위만 training step으로 통일 |
| 실험 흐름 | C/G/M residency ablation이 메인 보고서 | 최종 발표의 Phase offload -> CPU AdamW -> gradient streaming -> 1.5B 흐름으로 교체 |
| Rollout backend | 오래된 overview가 vLLM server 표시 | 최종 실험의 Hugging Face eager backend와 충돌해 그림 제거 |
| Precision | allocator 보고서가 FP16 smoke를 대표 결과처럼 설명 | 최종 FP32 발표와 분리하기 위해 중간 보고서 제거 |
| Protocol | 구 보고서는 30 steps/warm-up 5, 발표는 32/2 | 최종 run별 실제 metric count를 보고서에 명시 |
| Final figure 표시 | placement Pareto와 2x3 matrix가 final manifest에 포함 | 최종 발표에 쓰인 네 비교만 final로 유지 |
| Streaming 결론 | slide 22 subtitle은 1.7% overhead | raw data에 맞춰 9.814 -> 8.629 s, 12.1% 단축으로 수정 |
| 1.5B OOM | slide 24는 gradient 79% 누적 주장 | raw event에서 직접 재현되지 않아 backward OOM과 11.34 GiB/54 MiB만 주장 |

## 보존한 역사 자료

실패 및 과거 실험 자체는 삭제하지 않았다. 다음 원장에 역사 자료로 보존한다.

- `reports/final-figure-data/experiment-history/EXPERIMENT_HISTORY.md`
- `reports/final-figure-data/experiment-history/GROUP_CATALOG.md`
- `reports/final-figure-data/experiment-history/all_runs.csv`
- `reports/final-figure-data/experiment-history/all_runs.json`

이 자료의 `폐기/대체`, `진단`, `실패` 표시는 최종 발표 수치와 구분하기 위한
것이다.

## 발표 자료 자체에서 고쳐야 할 두 문구

1. Slide 22 subtitle
   - 기존: `Streaming lowers peak memory with only 1.7% end-to-end overhead`
   - 권장: `Streaming lowers backward memory and shortens the training step by 12.1%`
2. Slide 12 performance protocol
   - 기존: 모든 결과가 `32 steps, exclude 2 warm-up steps`인 것처럼 표시
   - 권장: `3 repeats; 2 warm-up steps excluded (28 or 30 measured steps depending on experiment)`

Slide 24의 `79%`는 산출 근거를 발표 speaker note에 추가하지 못하면
`OOM during backward as gradients accumulate`로 완화하는 것이 정확하다.

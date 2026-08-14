# 실험 이력 아카이브 사용 안내

이 디렉터리는 `outputs/**/run.json` 전수를 기준으로 만든 offload 연구 실험 이력이다.
2026-07-27부터 2026-08-13까지의 **95개 output group, 512개 run**을 포함한다.
성공 run뿐 아니라 실패, OOM, 중단 및 후속 실험으로 대체된 결과도 보존한다.

## 먼저 읽을 파일

1. `EXPERIMENT_HISTORY.md`
   - 시행착오의 시간적·논리적 흐름
   - 각 실험의 목적, 결과, 잘못된 가설, 측정 오류, 수정 내용
   - 채택 데이터와 폐기/대체 데이터의 구분
2. `GROUP_CATALOG.md`
   - 95개 output group과 그 아래 512개 run의 전수 목록
   - 각 run의 상태, 핵심 설정, 시간 및 메모리 metric
3. `all_runs.csv`
   - 필터링·정렬·통계 처리를 위한 run 단위 원장
4. `output_groups.csv`
   - 실험군 단위 실행 기간과 성공/실패/미완료 집계
5. `all_runs.json`
   - CSV보다 구조를 보존한 기계 판독용 원장

## 정확성 원칙

- 원본 `outputs`는 복사하거나 수정하지 않고 읽기만 했다.
- 값이 존재하지 않는 run은 추정해 채우지 않고 빈 값으로 남겼다.
- 실패/OOM은 성능 평균에서 제외하되 실패 자체가 결론인 경우 증거로 기록했다.
- detail/telemetry/Nsight 계측값과 telemetry-off 성능값을 구분했다.
- phase-local allocated peak, reserved peak, device-used memory를 혼합하지 않았다.
- residual로 역산한 과거 Optimize 시간은 보고 수치에서 제외했다.
- 16–512 MiB 과거 sweep은 Adam–H2D overlap이 꺼진 상태임을 명시했다.
- 아직 끝나지 않은 Nsight 원인 분해는 완료된 결론처럼 기록하지 않았다.

## 원장 재생성

저장소 루트에서 아래 명령을 실행한다.

```bash
envs/verl-titan/bin/python \
  results/build_experiment_inventory.py
```

재생성기는 `outputs/**/run.json`, `result.json`, stdout 및 aggregate 결과를 읽어
run/group 원장을 갱신한다. 사람이 해석한 연대기인 `EXPERIMENT_HISTORY.md`는
자동으로 덮어쓰지 않는다.

## 관련 결과 데이터 묶음

- 결과 그림별 출처: `../README.md`
- 결과 그림 데이터 manifest: `../manifest.csv`
- 필요한 원본만 모은 사본: `../data/`

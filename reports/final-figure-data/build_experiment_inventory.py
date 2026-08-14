#!/usr/bin/env python3
"""Build an evidence-first inventory of every benchmark run under outputs/."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/sda/juwon/verl-research")
OUTPUTS = ROOT / "outputs"
DEST = ROOT / "reports/final-figure-data/experiment-history"
DEST.mkdir(parents=True, exist_ok=True)

METRICS = [
    "perf/actor_forward_total_wall_s",
    "perf/actor_backward_total_wall_s",
    "perf/actor_adam_step_total_wall_s",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "timing_s/update_actor",
    "timing_s/step",
    "perf/throughput",
]

FIELDS = [
    "output_group", "run_id", "path", "label", "status", "returncode",
    "started_at", "finished_at", "wall_s", "model_path", "repeat", "warmup_steps",
    "strategy", "compute_dtype", "rollout_dtype", "actor_param", "ref_param",
    "actor_optimizer", "cpu_optimizer", "activation", "retain_between_rollout_log_prob",
    "bucket_layout", "bucket_mb", "num_staging_buffers", "async_d2h",
    "early_grad_release", "reuse_gpu_packing_buffers", "direct_cpu_grad_buffers",
    "cpu_grad_accumulation", "overlap_h2d_with_cpu_update", "telemetry",
    "detail", "sync", "nsys", "metric_steps",
] + [key.replace("/", "_") + "_mean" for key in METRICS]


def scalar(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def metric_rows(path: Path, warmup: int):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        step = re.search(r"\bstep:(\d+)\s+-", line)
        if not step or int(step.group(1)) <= warmup:
            continue
        rows.append({
            key: float(value)
            for key, value in re.findall(
                r"([A-Za-z0-9_./-]+):(-?[0-9]+(?:\.[0-9]+)?)", line
            )
        })
    return rows


records = []
for run_path in sorted(OUTPUTS.glob("**/run.json")):
    run_dir = run_path.parent
    relative = run_dir.relative_to(ROOT)
    parts = relative.parts
    output_group = parts[1] if len(parts) > 1 else ""
    try:
        run = json.loads(run_path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    result_path = run_dir / "result.json"
    try:
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        result = {}
    config = run.get("config", {})
    engine = config.get("offload_engine", {}) or {}
    warmup = int(run.get("warmup_steps", 0) or 0)
    rows = metric_rows(run_dir / "stdout.log", warmup)
    returncode = result.get("returncode")
    if returncode == 0:
        status = "ok"
    elif returncode is None:
        status = "incomplete"
    else:
        status = "failed"
    record = {
        "output_group": output_group,
        "run_id": run.get("run_id", run_dir.name),
        "path": str(relative),
        "label": config.get("label", result.get("label", "")),
        "status": status,
        "returncode": returncode,
        "started_at": run.get("started_at"),
        "finished_at": result.get("finished_at"),
        "wall_s": result.get("wall_s"),
        "model_path": run.get("model_path"),
        "repeat": run.get("repeat"),
        "warmup_steps": warmup,
        "strategy": config.get("strategy"),
        "compute_dtype": config.get("compute_dtype"),
        "rollout_dtype": config.get("rollout_dtype"),
        "actor_param": config.get("actor_param"),
        "ref_param": config.get("ref_param"),
        "actor_optimizer": config.get("actor_optimizer"),
        "cpu_optimizer": config.get("cpu_optimizer"),
        "activation": config.get("activation"),
        "retain_between_rollout_log_prob": config.get("retain_between_rollout_log_prob"),
        "bucket_layout": engine.get("bucket_layout"),
        "bucket_mb": engine.get("bucket_mb"),
        "num_staging_buffers": engine.get("num_staging_buffers"),
        "async_d2h": engine.get("async_d2h"),
        "early_grad_release": engine.get("early_grad_release"),
        "reuse_gpu_packing_buffers": engine.get("reuse_gpu_packing_buffers"),
        "direct_cpu_grad_buffers": engine.get("direct_cpu_grad_buffers"),
        "cpu_grad_accumulation": engine.get("cpu_grad_accumulation"),
        "overlap_h2d_with_cpu_update": engine.get("overlap_h2d_with_cpu_update"),
        "telemetry": engine.get("telemetry"),
        "detail": run.get("detail"),
        "sync": run.get("sync"),
        "nsys": run.get("nsys"),
        "metric_steps": len(rows),
    }
    for key in METRICS:
        values = [row[key] for row in rows if key in row]
        record[key.replace("/", "_") + "_mean"] = statistics.mean(values) if values else None
    records.append({key: scalar(record.get(key)) for key in FIELDS})

with (DEST / "all_runs.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(records)

(DEST / "all_runs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))

groups = defaultdict(list)
for record in records:
    groups[record["output_group"]].append(record)

with (DEST / "output_groups.csv").open("w", newline="", encoding="utf-8") as stream:
    fields = ["output_group", "runs", "ok", "failed", "incomplete", "first_started_at", "last_finished_at", "path"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name, items in sorted(groups.items()):
        starts = [float(item["started_at"]) for item in items if item["started_at"] not in (None, "")]
        ends = [float(item["finished_at"]) for item in items if item["finished_at"] not in (None, "")]
        writer.writerow({
            "output_group": name,
            "runs": len(items),
            "ok": sum(item["status"] == "ok" for item in items),
            "failed": sum(item["status"] == "failed" for item in items),
            "incomplete": sum(item["status"] == "incomplete" for item in items),
            "first_started_at": datetime.fromtimestamp(min(starts)).isoformat() if starts else "",
            "last_finished_at": datetime.fromtimestamp(max(ends)).isoformat() if ends else "",
            "path": f"outputs/{name}",
        })

catalog = [
    "# Exhaustive output-group catalog",
    "",
    "자동 생성된 전체 output group 목록이다. 수치는 stdout에서 warm-up을 제외하고 읽은 값이며,",
    "metric이 없거나 실패한 run은 빈 칸으로 남긴다.",
    "",
]
for name, items in sorted(groups.items()):
    catalog += [f"## {name}", "", f"- Path: `outputs/{name}`", f"- Runs: {len(items)}", ""]
    catalog.append("| run | status | label | bucket | slots | backward s | update s | peak GiB |")
    catalog.append("|---|---|---|---:|---:|---:|---:|---:|")
    for item in sorted(items, key=lambda value: str(value["run_id"])):
        def shown(key, digits=3):
            value = item.get(key)
            if value in (None, ""):
                return ""
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return str(value)
        label = str(item.get("label") or "").replace("|", "/")
        catalog.append(
            f"| `{item['run_id']}` | {item['status']} | {label} | "
            f"{shown('bucket_mb', 0)} | {shown('num_staging_buffers', 0)} | "
            f"{shown('perf_actor_backward_total_wall_s_mean')} | "
            f"{shown('perf_actor_adam_step_total_wall_s_mean')} | "
            f"{shown('perf_max_memory_allocated_gb_mean')} |"
        )
    catalog.append("")
(DEST / "GROUP_CATALOG.md").write_text("\n".join(catalog), encoding="utf-8")

print(f"wrote {len(records)} runs in {len(groups)} output groups to {DEST}")

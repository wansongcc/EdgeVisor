#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRIC_COLUMNS = [
    "episode_completion_time",
    "episode_delay_over_baseline",
    "recovery_latency",
    "cumulative_stall_time",
    "max_token_stall",
    "p99_tpot",
]

SUMMARY_COLUMNS = [
    "variant",
    "episode_completion_time_mean",
    "episode_completion_time_std",
    "episode_delay_mean",
    "episode_delay_std",
    "recovery_latency_mean",
    "recovery_latency_std",
    "cumulative_stall_mean",
    "cumulative_stall_std",
    "max_token_stall_mean",
    "max_token_stall_std",
    "p99_tpot_mean",
    "p99_tpot_std",
]


def iter_trace_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("trace.json")


def load_row(path: Path, root: Path) -> Dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(trace.get("agent_metrics", {}))
    rel = path.relative_to(root)
    parts = rel.parts
    variant = parts[0] if len(parts) > 0 else "unknown"
    fluctuation = parts[1] if len(parts) > 1 else "unknown"
    metrics.update(
        {
            "variant": variant,
            "fluctuation": fluctuation,
            "trace_path": str(path),
            "task_success": bool(metrics.get("task_success")),
        }
    )
    return metrics


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    variants = sorted({str(row.get("variant", "unknown")) for row in rows})
    out: List[Dict[str, Any]] = []
    for variant in variants:
        group = [row for row in rows if row.get("variant") == variant]
        values = {col: [float(row.get(col, 0.0) or 0.0) for row in group] for col in METRIC_COLUMNS}
        out.append(
            {
                "variant": variant,
                "episode_completion_time_mean": mean(values["episode_completion_time"]),
                "episode_completion_time_std": stdev(values["episode_completion_time"]),
                "episode_delay_mean": mean(values["episode_delay_over_baseline"]),
                "episode_delay_std": stdev(values["episode_delay_over_baseline"]),
                "recovery_latency_mean": mean(values["recovery_latency"]),
                "recovery_latency_std": stdev(values["recovery_latency"]),
                "cumulative_stall_mean": mean(values["cumulative_stall_time"]),
                "cumulative_stall_std": stdev(values["cumulative_stall_time"]),
                "max_token_stall_mean": mean(values["max_token_stall"]),
                "max_token_stall_std": stdev(values["max_token_stall"]),
                "p99_tpot_mean": mean(values["p99_tpot"]),
                "p99_tpot_std": stdev(values["p99_tpot"]),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_markdown(path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| Variant | Episode Completion Time | Episode Delay | Recovery Latency | Cumulative Stall | Max Token Stall | P99 TPOT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {episode_completion_time_mean:.3f} | {episode_delay_mean:.3f} | "
            "{recovery_latency_mean:.3f} | {cumulative_stall_mean:.3f} | "
            "{max_token_stall_mean:.3f} | {p99_tpot_mean:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_ablation_results.py <result_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    traces = sorted(iter_trace_files(root))
    rows = [load_row(path, root) for path in traces]
    if not rows:
        raise SystemExit(f"no trace.json files found under {root}")

    summary = summarize(rows)
    write_csv(root / "ablation_summary.csv", summary, SUMMARY_COLUMNS)
    write_markdown(root / "ablation_summary.md", "EdgeVisor Agentic Ablation Summary", summary)

    groups = {
        "shadow_kv_ablation_summary.md": ["full", "shadow_disabled_transfer", "shadow_disabled_recompute"],
        "pointer_swizzling_ablation_summary.md": ["full", "pointer_operator_rebuild", "pointer_weight_rematerialize"],
        "jit_ablation_summary.md": ["full", "jit_static", "jit_greedy"],
        "vg_ablation_summary.md": ["full", "vg_flat", "vg_random"],
    }
    for filename, variants in groups.items():
        subset = [row for row in summary if row["variant"] in variants]
        write_markdown(root / filename, filename.replace("_", " ").replace(".md", "").title(), subset)

    write_csv(
        root / "ablation_runs.csv",
        rows,
        [
            "variant",
            "fluctuation",
            "episode_id",
            "episode_length",
            "generation_count",
            "tool_call_count",
            "episode_completion_time",
            "episode_delay_over_baseline",
            "recovery_latency",
            "cumulative_stall_time",
            "max_token_stall",
            "p99_tpot",
            "task_success",
            "trace_path",
        ],
    )
    print(json.dumps({"root": str(root), "runs": len(rows), "summary": str(root / "ablation_summary.csv")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

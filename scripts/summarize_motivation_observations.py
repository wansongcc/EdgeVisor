#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


RUN_COLUMNS = [
    "experiment",
    "variant",
    "fluctuation",
    "prefill_tokens",
    "repeat",
    "task_success",
    "episode_completion_time",
    "episode_delay_over_baseline",
    "normalized_episode_time",
    "affected_generation_latency",
    "total_generation_time_ms",
    "cumulative_stall_time",
    "recovery_latency",
    "max_token_stall",
    "p99_tpot",
    "generation_count",
    "tool_call_count",
    "plan_emit_count",
    "plan_apply_count",
    "dynamic_plan_event_count",
    "t_decision_total_ms",
    "t_state_prepare_total_ms",
    "t_bind_total_ms",
    "t_command_total_ms",
    "t_apply_total_ms",
    "t_recover_total_ms",
    "state_transfer_bytes_total",
    "recompute_tokens_or_layers_total",
    "binding_update_count_total",
    "materialized_bytes_total",
    "state_transfer_mb_total",
    "transfer_bytes_per_token",
    "stall_per_prefill_token",
    "trace_path",
    "run_root",
]

METRICS = [
    "episode_completion_time",
    "affected_generation_latency",
    "total_generation_time_ms",
    "cumulative_stall_time",
    "recovery_latency",
    "max_token_stall",
    "p99_tpot",
    "t_decision_total_ms",
    "t_state_prepare_total_ms",
    "t_bind_total_ms",
    "t_command_total_ms",
    "t_apply_total_ms",
    "t_recover_total_ms",
    "state_transfer_bytes_total",
    "recompute_tokens_or_layers_total",
    "binding_update_count_total",
    "materialized_bytes_total",
]

SUMMARY_COLUMNS = ["experiment", "variant", "fluctuation", "prefill_tokens", "runs", "success_count", "success_rate"]
for metric in METRICS:
    SUMMARY_COLUMNS.extend([f"{metric}_mean", f"{metric}_std"])

OBS1_COLUMNS = [
    "variant",
    "fluctuation",
    "runs",
    "success_rate",
    "episode_completion_time_mean",
    "episode_completion_time_std",
    "normalized_episode_time",
    "affected_generation_latency_mean",
    "cumulative_stall_time_mean",
    "max_token_stall_mean",
    "p99_tpot_mean",
    "boundary_delay_factor",
]

OBS2_COLUMNS = [
    "prefill_tokens",
    "variant",
    "runs",
    "success_rate",
    "episode_completion_time_mean",
    "episode_completion_time_std",
    "recovery_latency_mean",
    "recovery_latency_std",
    "cumulative_stall_time_mean",
    "t_state_prepare_total_ms_mean",
    "t_bind_total_ms_mean",
    "t_apply_total_ms_mean",
    "t_recover_total_ms_mean",
    "state_transfer_bytes_total_mean",
    "state_transfer_mb_total_mean",
    "recompute_tokens_or_layers_total_mean",
    "transfer_stall_factor",
    "net_episode_overhead",
    "transfer_bytes_per_token_mean",
    "stall_per_prefill_token_mean",
]

EVENT_COLUMNS = [
    "trace_path",
    "variant",
    "fluctuation",
    "prefill_tokens",
    "repeat",
    "generation_id",
    "event_id",
    "fallback_reason",
    "apply_success",
    "transfer_bytes",
    "state_transfer_bytes",
    "recompute_tokens_or_layers",
    "t_state_prepare_ms",
    "t_bind_ms",
    "t_apply_ms",
    "t_recover_ms",
    "binding_update_count",
    "materialized_bytes",
]


def safe_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        out = float(value)
        return 0.0 if math.isnan(out) or math.isinf(out) else out
    except Exception:
        return 0.0


def safe_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def all_manifest_items(root: Path) -> Dict[str, Dict[str, Any]]:
    by_run: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("manifest.jsonl")):
        for item in load_jsonl(path):
            run_root = item.get("run_root")
            if run_root:
                by_run[str(Path(str(run_root)).resolve())] = item
    return by_run


def nearest_manifest(trace_path: Path, root: Path, manifest: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for parent in [trace_path.parent, *trace_path.parents]:
        key = str(parent.resolve())
        if key in manifest:
            return manifest[key]
        if parent.resolve() == root:
            break
    return {}


def experiment_label(path: Path, manifest_item: Dict[str, Any]) -> str:
    run_root = str(manifest_item.get("run_root", ""))
    text = f"{path} {run_root}"
    if "obs1" in text:
        return "observation1"
    if "obs2" in text:
        return "observation2"
    if "smoke" in text:
        return "smoke"
    return "unknown"


def load_rows(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifest = all_manifest_items(root)
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    for trace_path in sorted(root.rglob("trace.json")):
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        item = nearest_manifest(trace_path, root, manifest)
        metrics = trace.get("agent_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        dynamic_plan = item.get("dynamic_plan", {}) if isinstance(item.get("dynamic_plan"), dict) else {}
        variant = str(item.get("variant") or dynamic_plan.get("experiment_variant") or "unknown")
        fluctuation = str(item.get("fluctuation") or dynamic_plan.get("fluctuation_type") or "unknown")
        prefill_tokens = safe_int(dynamic_plan.get("prefill_tokens"))
        if prefill_tokens == 0:
            run_root = Path(str(item.get("run_root", "")))
            config = run_root / "ablation_config.json"
            if config.exists():
                try:
                    prefill_tokens = safe_int(json.loads(config.read_text(encoding="utf-8")).get("prefill_tokens"))
                except Exception:
                    pass
        row = {col: "" for col in RUN_COLUMNS}
        row.update(metrics)
        row.update(
            {
                "experiment": experiment_label(trace_path, item),
                "variant": variant,
                "fluctuation": fluctuation,
                "prefill_tokens": prefill_tokens,
                "repeat": item.get("repeat", dynamic_plan.get("repeat", "")),
                "task_success": bool(metrics.get("task_success", False)),
                "trace_path": str(trace_path),
                "run_root": item.get("run_root", str(trace_path.parent)),
            }
        )
        row["state_transfer_mb_total"] = safe_float(row.get("state_transfer_bytes_total")) / (1024.0 * 1024.0)
        row["transfer_bytes_per_token"] = safe_float(row.get("state_transfer_bytes_total")) / max(prefill_tokens, 1)
        row["stall_per_prefill_token"] = safe_float(row.get("recovery_latency")) / max(prefill_tokens, 1)
        rows.append(row)

        llm_events = trace.get("llm_events") if isinstance(trace.get("llm_events"), list) else trace.get("events", [])
        if isinstance(llm_events, list):
            for ev in llm_events:
                if not isinstance(ev, dict):
                    continue
                ev_metrics = ev.get("metrics", {})
                if not isinstance(ev_metrics, dict):
                    continue
                for rec in ev_metrics.get("ablation_events", []) or []:
                    if not isinstance(rec, dict):
                        continue
                    events.append(
                        {
                            "trace_path": str(trace_path),
                            "variant": variant,
                            "fluctuation": fluctuation,
                            "prefill_tokens": prefill_tokens,
                            "repeat": row["repeat"],
                            "generation_id": ev.get("generation_id", ev.get("id", "")),
                            "event_id": rec.get("event_id", ""),
                            "fallback_reason": rec.get("fallback_reason", rec.get("fallbackReason", "")),
                            "apply_success": rec.get("apply_success", ""),
                            "transfer_bytes": rec.get("transfer_bytes", ""),
                            "state_transfer_bytes": rec.get("state_transfer_bytes", ""),
                            "recompute_tokens_or_layers": rec.get("recompute_tokens_or_layers", ""),
                            "t_state_prepare_ms": rec.get("t_state_prepare_ms", ""),
                            "t_bind_ms": rec.get("t_bind_ms", ""),
                            "t_apply_ms": rec.get("t_apply_ms", ""),
                            "t_recover_ms": rec.get("t_recover_ms", ""),
                            "binding_update_count": rec.get("binding_update_count", ""),
                            "materialized_bytes": rec.get("materialized_bytes", ""),
                        }
                    )
    return rows, events


def add_baseline_derivatives(rows: List[Dict[str, Any]]) -> None:
    stable = [
        safe_float(r.get("episode_completion_time"))
        for r in rows
        if r.get("experiment") == "observation1" and r.get("variant") == "stable" and bool(r.get("task_success"))
    ]
    baseline = mean(stable)
    for row in rows:
        episode = safe_float(row.get("episode_completion_time"))
        row["episode_delay_over_baseline"] = episode - baseline if baseline > 0 else 0.0
        row["normalized_episode_time"] = episode / baseline if baseline > 0 else 0.0


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted({(str(r.get("experiment")), str(r.get("variant")), str(r.get("fluctuation")), safe_int(r.get("prefill_tokens"))) for r in rows})
    out: List[Dict[str, Any]] = []
    for experiment, variant, fluctuation, prefill in keys:
        group = [
            r for r in rows
            if str(r.get("experiment")) == experiment
            and str(r.get("variant")) == variant
            and str(r.get("fluctuation")) == fluctuation
            and safe_int(r.get("prefill_tokens")) == prefill
        ]
        row: Dict[str, Any] = {
            "experiment": experiment,
            "variant": variant,
            "fluctuation": fluctuation,
            "prefill_tokens": prefill,
            "runs": len(group),
            "success_count": sum(1 for r in group if bool(r.get("task_success"))),
        }
        row["success_rate"] = row["success_count"] / len(group) if group else 0.0
        for metric in METRICS:
            values = [safe_float(r.get(metric)) for r in group]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values)
        out.append(row)
    return out


def observation1(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [r for r in summary if r.get("experiment") == "observation1"]
    baseline = next((r for r in rows if r.get("variant") == "stable"), None)
    live = next((r for r in rows if r.get("variant") == "full"), None)
    baseline_mean = safe_float(baseline.get("episode_completion_time_mean")) if baseline else 0.0
    live_mean = safe_float(live.get("episode_completion_time_mean")) if live else 0.0
    out: List[Dict[str, Any]] = []
    for r in rows:
        episode_mean = safe_float(r.get("episode_completion_time_mean"))
        out.append(
            {
                "variant": r.get("variant"),
                "fluctuation": r.get("fluctuation"),
                "runs": r.get("runs"),
                "success_rate": r.get("success_rate"),
                "episode_completion_time_mean": episode_mean,
                "episode_completion_time_std": r.get("episode_completion_time_std"),
                "normalized_episode_time": episode_mean / baseline_mean if baseline_mean > 0 else 0.0,
                "affected_generation_latency_mean": r.get("affected_generation_latency_mean"),
                "cumulative_stall_time_mean": r.get("cumulative_stall_time_mean"),
                "max_token_stall_mean": r.get("max_token_stall_mean"),
                "p99_tpot_mean": r.get("p99_tpot_mean"),
                "boundary_delay_factor": (
                    episode_mean / live_mean
                    if r.get("variant") == "boundary_only" and live_mean > 0
                    else ""
                ),
            }
        )
    order = {"stable": 0, "boundary_only": 1, "full": 2}
    return sorted(out, key=lambda r: order.get(str(r.get("variant")), 99))


def observation2(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [r for r in summary if r.get("experiment") == "observation2"]
    by_prefill_full: Dict[int, Dict[str, Any]] = {
        safe_int(r.get("prefill_tokens")): r for r in rows if r.get("variant") == "full"
    }
    out: List[Dict[str, Any]] = []
    for r in rows:
        prefill = safe_int(r.get("prefill_tokens"))
        full = by_prefill_full.get(prefill)
        full_recovery = safe_float(full.get("recovery_latency_mean")) if full else 0.0
        full_episode = safe_float(full.get("episode_completion_time_mean")) if full else 0.0
        recovery = safe_float(r.get("recovery_latency_mean"))
        episode = safe_float(r.get("episode_completion_time_mean"))
        transfer_bytes = safe_float(r.get("state_transfer_bytes_total_mean"))
        out.append(
            {
                "prefill_tokens": prefill,
                "variant": r.get("variant"),
                "runs": r.get("runs"),
                "success_rate": r.get("success_rate"),
                "episode_completion_time_mean": episode,
                "episode_completion_time_std": r.get("episode_completion_time_std"),
                "recovery_latency_mean": recovery,
                "recovery_latency_std": r.get("recovery_latency_std"),
                "cumulative_stall_time_mean": r.get("cumulative_stall_time_mean"),
                "t_state_prepare_total_ms_mean": r.get("t_state_prepare_total_ms_mean"),
                "t_bind_total_ms_mean": r.get("t_bind_total_ms_mean"),
                "t_apply_total_ms_mean": r.get("t_apply_total_ms_mean"),
                "t_recover_total_ms_mean": r.get("t_recover_total_ms_mean"),
                "state_transfer_bytes_total_mean": transfer_bytes,
                "state_transfer_mb_total_mean": transfer_bytes / (1024.0 * 1024.0),
                "recompute_tokens_or_layers_total_mean": r.get("recompute_tokens_or_layers_total_mean"),
                "transfer_stall_factor": recovery / full_recovery if full_recovery > 0 and r.get("variant") != "full" else "",
                "net_episode_overhead": episode - full_episode if full_episode > 0 and r.get("variant") != "full" else "",
                "transfer_bytes_per_token_mean": transfer_bytes / max(prefill, 1),
                "stall_per_prefill_token_mean": recovery / max(prefill, 1),
            }
        )
    return sorted(out, key=lambda r: (safe_int(r.get("prefill_tokens")), str(r.get("variant"))))


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def matrix(rows: List[Dict[str, Any]], columns: Sequence[str]) -> List[List[Any]]:
    return [list(columns)] + [[row.get(col, "") for col in columns] for row in rows]


def col_name(idx: int) -> str:
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def xlsx_cell(value: Any, row_idx: int, col_idx: int) -> str:
    ref = f"{col_name(col_idx)}{row_idx}"
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{safe_float(value):.12g}</v></c>'
    text = html.escape("" if value is None else str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def xlsx_sheet(rows: Sequence[Sequence[Any]]) -> str:
    body = []
    for r_idx, row in enumerate(rows, 1):
        cells = "".join(xlsx_cell(value, r_idx, c_idx) for c_idx, value in enumerate(row, 1))
        body.append(f'<row r="{r_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>' + "".join(body) + "</sheetData></worksheet>"
    )


def write_xlsx(path: Path, sheets: Dict[str, Sequence[Sequence[Any]]]) -> None:
    names = list(sheets)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        overrides = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        ]
        for idx in range(1, len(names) + 1):
            overrides.append(f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        overrides.append("</Types>")
        zf.writestr("[Content_Types].xml", "".join(overrides))
        zf.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        sheet_defs = []
        rels = []
        for idx, name in enumerate(names, 1):
            sheet_defs.append(f'<sheet name="{html.escape(name[:31])}" sheetId="{idx}" r:id="rId{idx}"/>')
            rels.append(f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>')
        rels.append(f'<Relationship Id="rId{len(names)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
        zf.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(sheet_defs) + "</sheets></workbook>")
        zf.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>")
        zf.writestr("xl/styles.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs></styleSheet>')
        for idx, name in enumerate(names, 1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", xlsx_sheet(sheets[name]))


def markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> List[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_markdown(root: Path, obs1: List[Dict[str, Any]], obs2: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    env_path = root / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    failures = [r for r in rows if not bool(r.get("task_success"))]
    boundary = next((r for r in obs1 if r.get("variant") == "boundary_only"), {})
    live = next((r for r in obs1 if r.get("variant") == "full"), {})
    max_prefill = max([safe_int(r.get("prefill_tokens")) for r in obs2] or [0])
    transfer_max = next((r for r in obs2 if r.get("variant") == "shadow_transfer" and safe_int(r.get("prefill_tokens")) == max_prefill), {})
    full_max = next((r for r in obs2 if r.get("variant") == "full" and safe_int(r.get("prefill_tokens")) == max_prefill), {})
    lines = [
        "# EdgeVisor Motivation Experiment Summary",
        "",
        "## 1. Environment",
        "",
        f"- git commit: `{env.get('git_commit', '')}`",
        f"- hostname: `{env.get('hostname', '')}`",
        f"- date/time: `{env.get('created_at', '')}`",
        f"- CUDA_VISIBLE_DEVICES: `{env.get('cuda_visible_devices', '')}`",
        f"- GPU model names: `{env.get('gpu_inventory', '')}`",
        f"- model path: `{env.get('model_path', '')}`",
        f"- tokenizer path: `{env.get('tokenizer_path', '')}`",
        "- EdgeVisor build command: `CUDA_VISIBLE_DEVICES=0,1,2 bash build.sh`",
        f"- trigger mode: `{env.get('plan_mode', 'next_barrier')}`",
        f"- shadow scope: `{env.get('shadow_scope', 'inter_stage_layers')}`",
        f"- network proxy throttle TTL: `{env.get('network_proxy_throttle_ttl_s', '')}` s",
        f"- network proxy rate: inter-stage `{env.get('network_inter_mib_s', '')}` MiB/s, intra-stage `{env.get('network_intra_mib_s', '')}` MiB/s",
        "",
        "## 2. Observation 1 Table",
        "",
    ]
    lines.extend(markdown_table(obs1, OBS1_COLUMNS))
    lines.extend(["", "## 3. Observation 2 Table", ""])
    lines.extend(markdown_table(obs2, OBS2_COLUMNS))
    lines.extend(["", "## 4. Headline Findings", ""])
    bdf = boundary.get("boundary_delay_factor", "")
    if bdf != "":
        bdf_value = safe_float(bdf)
        if bdf_value >= 1.0:
            lines.append(f"- Boundary/no-live continuation increased episode completion time by {bdf_value:.2f}x over prepared live continuation under mixed_bw degradation.")
        else:
            lines.append(f"- Boundary/no-live continuation completed in {bdf_value:.2f}x the prepared-live episode time under this 64 MiB/s, 15 s windowed mixed_bw setting; prepared live continuation was slower in total episode time because its migration/redundancy overhead dominated the short degradation window.")
    tsf = transfer_max.get("transfer_stall_factor", "")
    if tsf != "":
        lines.append(f"- On-demand transfer increased recovery latency by {safe_float(tsf):.2f}x over prepared continuation at {max_prefill} prefill tokens.")
    lines.append(f"- On-demand transfer moved {safe_float(transfer_max.get('state_transfer_mb_total_mean')):.3f} MB of state on average at {max_prefill} prefill tokens.")
    lines.append(
        "- Recovery latency at max prefix: "
        f"full={safe_float(full_max.get('recovery_latency_mean')):.3f} ms, "
        f"shadow_transfer={safe_float(transfer_max.get('recovery_latency_mean')):.3f} ms."
    )
    lines.extend(
        [
            "",
            "## 5. Notes and Caveats",
            "",
            f"- Failed runs: {len(failures)}. See `motivation_runs.csv` for per-run status.",
            "- `stable` disables dynamic migration and uses `none` fluctuation.",
            "- `boundary_only` disables dynamic migration but uses a bounded real perturbation window; network proxy starts throttled and automatically clears after the configured TTL.",
            "- `full` and `shadow_transfer` activate the TCP proxy around dynamic-plan command injection and automatically clear it after the configured TTL.",
            f"- Experiment B used `{env.get('shadow_scope', 'inter_stage_layers')}`.",
            "- No `simulated_stall_ms` or `simulated_recovery_latency_ms` values are generated by the Motivation runner.",
            "",
            "## 6. XLSX Interpretation",
            "",
            "- `Runs`: one row per `trace.json`, with raw agent metrics and derived normalized/stall-per-token fields.",
            "- `Summary`: mean/std grouped by experiment, variant, fluctuation, and prefill length.",
            "- `Observation1`: paper-facing comparison for stable, boundary/no-live, and prepared live continuation.",
            "- `Observation2`: paper-facing prefix sweep comparing prepared continuation and on-demand transfer.",
            "- `Events`: extracted ablation JSONL records embedded in traces for migration/recovery breakdowns.",
        ]
    )
    (root / "motivation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize EdgeVisor Motivation Observation experiments.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-base", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    rows, events = load_rows(root)
    if not rows:
        raise SystemExit(f"no trace.json files found under {root}")
    add_baseline_derivatives(rows)
    summary = summarize(rows)
    obs1 = observation1(summary)
    obs2 = observation2(summary)

    write_csv(root / "motivation_runs.csv", rows, RUN_COLUMNS)
    write_csv(root / "motivation_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(root / "motivation_observation1.csv", obs1, OBS1_COLUMNS)
    write_csv(root / "motivation_observation2.csv", obs2, OBS2_COLUMNS)
    write_csv(root / "motivation_event_breakdown.csv", events, EVENT_COLUMNS)
    write_markdown(root, obs1, obs2, rows)
    write_xlsx(
        root / "motivation_results.xlsx",
        {
            "Observation1": matrix(obs1, OBS1_COLUMNS),
            "Observation2": matrix(obs2, OBS2_COLUMNS),
            "Summary": matrix(summary, SUMMARY_COLUMNS),
            "Runs": matrix(rows, RUN_COLUMNS),
            "Events": matrix(events, EVENT_COLUMNS),
        },
    )
    print(json.dumps({"root": str(root), "runs": len(rows), "events": len(events), "xlsx": str(root / "motivation_results.xlsx")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

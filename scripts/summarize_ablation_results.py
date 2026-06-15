#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


RUN_COLUMNS = [
    "variant",
    "fluctuation",
    "repeat",
    "episode_id",
    "episode_length",
    "generation_count",
    "tool_call_count",
    "task_success",
    "episode_completion_time",
    "total_generation_time_ms",
    "total_tool_time_ms",
    "avg_generation_time_ms",
    "generations_after_cache_warmup_ms",
    "p99_tpot",
    "migration_count",
    "plan_emit_count",
    "plan_apply_count",
    "pp_migration_apply_count",
    "pp_migration_recover_count",
    "head_migration_recover_count",
    "binding_refresh_count",
    "cumulative_stall_time",
    "recovery_latency",
    "max_token_stall",
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
    "rejected_moves_total",
    "fallback_count_total",
    "candidate_count_max",
    "ablation_event_count",
    "dynamic_plan_event_count",
    "persistent_cache_hit_rate",
    "tool_error_count",
    "extra_tool_rejected_count",
    "trace_path",
]

METRIC_COLUMNS = [
    "episode_completion_time",
    "total_generation_time_ms",
    "total_tool_time_ms",
    "avg_generation_time_ms",
    "generations_after_cache_warmup_ms",
    "p99_tpot",
    "migration_count",
    "cumulative_stall_time",
    "recovery_latency",
    "max_token_stall",
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
    "rejected_moves_total",
    "fallback_count_total",
    "persistent_cache_hit_rate",
]

SUMMARY_COLUMNS = [
    "variant",
    "fluctuation",
    "runs",
    "success_count",
    "success_rate",
]
for _metric in METRIC_COLUMNS:
    SUMMARY_COLUMNS.extend([f"{_metric}_mean", f"{_metric}_std"])


def iter_trace_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("trace.json")


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return 0.0
        return out
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_manifest(root: Path) -> Dict[str, Dict[str, Any]]:
    manifest = root / "manifest.jsonl"
    by_run_root: Dict[str, Dict[str, Any]] = {}
    if not manifest.exists():
        return by_run_root
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        run_root = item.get("run_root")
        if run_root:
            by_run_root[str(Path(run_root).resolve())] = item
    return by_run_root


def load_manifest_results(root: Path) -> List[Dict[str, Any]]:
    path = root / "manifest_results.jsonl"
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


def _fallback_labels(path: Path, root: Path) -> Dict[str, Any]:
    rel = path.relative_to(root)
    parts = rel.parts
    variant = parts[0] if len(parts) > 0 else "unknown"
    fluctuation = parts[1] if len(parts) > 1 else "unknown"
    repeat = ""
    for part in parts:
        if part.startswith("rep_"):
            repeat = part.removeprefix("rep_")
            break
    return {"variant": variant, "fluctuation": fluctuation, "repeat": repeat}


def load_row(path: Path, root: Path, manifest: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(trace.get("agent_metrics", {}))
    labels = _fallback_labels(path, root)
    for parent in [path.parent, *path.parents]:
        item = manifest.get(str(parent.resolve()))
        if item:
            labels.update(
                {
                    "variant": item.get("variant", labels["variant"]),
                    "fluctuation": item.get("fluctuation", labels["fluctuation"]),
                    "repeat": item.get("repeat", labels["repeat"]),
                }
            )
            break
        if parent == root:
            break
    row: Dict[str, Any] = {col: "" for col in RUN_COLUMNS}
    row.update(metrics)
    row.update(labels)
    row["trace_path"] = str(path)
    row["task_success"] = bool(metrics.get("task_success"))
    row["episode_id"] = metrics.get("episode_id", trace.get("episode_id", ""))
    return row


def failed_row(result: Dict[str, Any], manifest: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    run_root = str(Path(str(result.get("run_root", ""))).resolve())
    item = manifest.get(run_root, {})
    row: Dict[str, Any] = {col: "" for col in RUN_COLUMNS}
    row.update(
        {
            "variant": item.get("variant", result.get("variant", "unknown")),
            "fluctuation": item.get("fluctuation", result.get("fluctuation", "unknown")),
            "repeat": item.get("repeat", result.get("repeat", "")),
            "task_success": False,
            "episode_completion_time": _safe_float(result.get("elapsed_s")) * 1000.0,
            "trace_path": "",
        }
    )
    return row


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted({(str(row.get("variant", "unknown")), str(row.get("fluctuation", "unknown"))) for row in rows})
    out: List[Dict[str, Any]] = []
    for variant, fluctuation in keys:
        group = [row for row in rows if str(row.get("variant")) == variant and str(row.get("fluctuation")) == fluctuation]
        summary: Dict[str, Any] = {
            "variant": variant,
            "fluctuation": fluctuation,
            "runs": len(group),
            "success_count": sum(1 for row in group if bool(row.get("task_success"))),
        }
        summary["success_rate"] = summary["success_count"] / len(group) if group else 0.0
        for metric in METRIC_COLUMNS:
            values = [_safe_float(row.get(metric)) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values)
        out.append(summary)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_markdown(path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| Variant | Fluctuation | Runs | Success | Episode s mean | Episode s std | Gen ms mean | Stall ms mean | Recover ms mean | State prep ms mean | Bind ms mean | Transfer bytes mean | Recompute mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {fluctuation} | {runs} | {success_rate:.2%} | "
            "{episode_s_mean:.3f} | {episode_s_std:.3f} | {gen_ms_mean:.3f} | "
            "{stall_mean:.3f} | {recover_mean:.3f} | {state_prepare_mean:.3f} | "
            "{bind_mean:.3f} | {transfer_mean:.0f} | {recompute_mean:.3f} |".format(
                variant=row.get("variant", ""),
                fluctuation=row.get("fluctuation", ""),
                runs=_safe_int(row.get("runs")),
                success_rate=_safe_float(row.get("success_rate")),
                episode_s_mean=_safe_float(row.get("episode_completion_time_mean")) / 1000.0,
                episode_s_std=_safe_float(row.get("episode_completion_time_std")) / 1000.0,
                gen_ms_mean=_safe_float(row.get("total_generation_time_ms_mean")),
                stall_mean=_safe_float(row.get("cumulative_stall_time_mean")),
                recover_mean=_safe_float(row.get("recovery_latency_mean")),
                state_prepare_mean=_safe_float(row.get("t_state_prepare_total_ms_mean")),
                bind_mean=_safe_float(row.get("t_bind_total_ms_mean")),
                transfer_mean=_safe_float(row.get("state_transfer_bytes_total_mean")),
                recompute_mean=_safe_float(row.get("recompute_tokens_or_layers_total_mean")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _xlsx_col_name(idx: int) -> str:
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _xlsx_cell(value: Any, row_idx: int, col_idx: int) -> str:
    ref = f"{_xlsx_col_name(col_idx)}{row_idx}"
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = _safe_float(value)
        return f'<c r="{ref}"><v>{num:.12g}</v></c>'
    text = html.escape("" if value is None else str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _xlsx_sheet(rows: Sequence[Sequence[Any]]) -> str:
    body = []
    for r_idx, row in enumerate(rows, 1):
        cells = "".join(_xlsx_cell(value, r_idx, c_idx) for c_idx, value in enumerate(row, 1))
        body.append(f'<row r="{r_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>'
        + "".join(body)
        + "</sheetData></worksheet>"
    )


def write_xlsx(path: Path, sheets: Dict[str, Sequence[Sequence[Any]]]) -> None:
    sheet_names = list(sheets)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")
    workbook_sheets = []
    workbook_rels = []
    for idx, name in enumerate(sheet_names, 1):
        safe = html.escape(name[:31])
        workbook_sheets.append(f'<sheet name="{safe}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
    styles_rid = len(sheet_names) + 1
    workbook_rels.append(
        f'<Relationship Id="rId{styles_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(workbook_sheets)
            + "</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(workbook_rels)
            + "</Relationships>",
        )
        zf.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            "</styleSheet>",
        )
        for idx, name in enumerate(sheet_names, 1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _xlsx_sheet(sheets[name]))


def rows_to_matrix(rows: List[Dict[str, Any]], columns: Sequence[str]) -> List[List[Any]]:
    return [list(columns)] + [[row.get(col, "") for col in columns] for row in rows]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_ablation_results.py <result_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    traces = sorted(iter_trace_files(root))
    manifest = load_manifest(root)
    manifest_results = load_manifest_results(root)
    rows = [load_row(path, root, manifest) for path in traces]
    traced_run_roots = set()
    for row in rows:
        trace_path = str(row.get("trace_path", ""))
        if trace_path:
            for parent in [Path(trace_path).parent, *Path(trace_path).parents]:
                if str(parent.resolve()) in manifest:
                    traced_run_roots.add(str(parent.resolve()))
                    break
                if parent.resolve() == root:
                    break
    for result in manifest_results:
        run_root = str(Path(str(result.get("run_root", ""))).resolve())
        if run_root and run_root not in traced_run_roots and int(result.get("rc", 1) or 0) != 0:
            rows.append(failed_row(result, manifest))
    if not rows:
        raise SystemExit(f"no trace.json files found under {root}")

    summary = summarize(rows)
    write_csv(root / "ablation_runs.csv", rows, RUN_COLUMNS)
    write_csv(root / "ablation_summary.csv", summary, SUMMARY_COLUMNS)
    write_markdown(root / "ablation_summary.md", "EdgeVisor Agentic Ablation Summary", summary)

    groups = {
        "shadow_kv_ablation_summary.md": {"full", "shadow_transfer", "shadow_recompute"},
        "pointer_swizzling_ablation_summary.md": {"full", "pointer_rebuild", "pointer_rematerialize"},
        "jit_ablation_summary.md": {"full", "jit_static", "jit_greedy"},
        "vg_ablation_summary.md": {"full", "vg_flat", "vg_pure_pp"},
    }
    for filename, variants in groups.items():
        subset = [row for row in summary if str(row.get("variant")) in variants]
        write_markdown(root / filename, filename.replace("_", " ").replace(".md", "").title(), subset)

    write_xlsx(
        root / "ablation_results.xlsx",
        {
            "Summary": rows_to_matrix(summary, SUMMARY_COLUMNS),
            "Runs": rows_to_matrix(rows, RUN_COLUMNS),
        },
    )
    print(
        json.dumps(
            {
                "root": str(root),
                "runs": len(rows),
                "summary": str(root / "ablation_summary.csv"),
                "xlsx": str(root / "ablation_results.xlsx"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

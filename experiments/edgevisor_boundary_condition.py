#!/usr/bin/env python3
"""Boundary-condition experiment: EdgeVisor is not a universal win.

The experiment keeps the same 8-container 3/3/2 Docker topology used by the
main ablation. It targets the regime where Shadow KV has little to save:
short context, high/stable bandwidth, and a small Stage-1 head migration.

Variants:
  - with_shadow_kv: kv-redundancy=2, no KV transfer before migration.
  - without_shadow_transfer: kv-redundancy=0, measured Docker TCP KV-state
    transfer before issuing the same UDS migration.

The current runtime does not implement semantic KV copy for head migration, so
`without_shadow_transfer` uses kv-redundancy=0 to disable extra KV computation,
performs a real state-size TCP transfer, then lets the UDS migration apply. The
output quality is intentionally not evaluated, matching the earlier ablations.
"""

import csv
import json
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import edgevisor_ablation_suite as suite


RUN_ID = time.strftime("boundary_%Y%m%d_%H%M%S")
suite.RUN_ID = RUN_ID
suite.OUT = suite.BASE / "edgevisor_boundary_results" / suite.RUN_ID
suite.SOCK_DIR = suite.OUT / "s"
OUT = suite.OUT
SOCK_DIR = suite.SOCK_DIR
EXP = "edgevisor_boundary_condition"

HIGH_STABLE_BW_MBPS = int(os.environ.get("B01_BOUNDARY_BW_MBPS", "1000"))
SHADOW_KV_REDUNDANCY = int(os.environ.get("B01_BOUNDARY_SHADOW_KV", "2"))
NO_SHADOW_KV_REDUNDANCY = int(os.environ.get("B01_BOUNDARY_NO_SHADOW_KV", "0"))
LIVE_STEPS = int(os.environ.get("B01_BOUNDARY_STEPS", "40"))
LIVE_CONTEXTS = [int(x) for x in os.environ.get("B01_BOUNDARY_CONTEXTS", "128,512,2048").split(",") if x]
MICRO_CONTEXTS = [int(x) for x in os.environ.get("B01_BOUNDARY_MICRO_CONTEXTS", "128,256,512,1024,2048,4096,8192,16384").split(",") if x]
MICRO_BWS = [int(x) for x in os.environ.get("B01_BOUNDARY_MICRO_BWS", "1000,100").split(",") if x]
MICRO_REPEATS = int(os.environ.get("B01_BOUNDARY_MICRO_REPEATS", "2"))
SMOKE = os.environ.get("B01_BOUNDARY_SMOKE") == "1"

if SMOKE:
    LIVE_CONTEXTS = [128]
    MICRO_CONTEXTS = [128, 512]
    MICRO_BWS = [HIGH_STABLE_BW_MBPS]
    MICRO_REPEATS = 1
    LIVE_STEPS = 24

MODELS = ["3B"] if SMOKE else ["3B", "8B"]
suite.BASE_NET_MBPS = [HIGH_STABLE_BW_MBPS] * 8


@dataclass
class LiveRow:
    run_id: str
    model: str
    variant: str
    context_len: int
    bandwidth_mbps: int
    kv_redundancy: int
    steps: int
    max_seq_len: int
    state_bytes: int
    t_state_ms: float
    transfer_mbps: float
    t_cmd_ms: float
    t_recover_ms: float
    stall_time_ms: float
    apply_seen: bool
    apply_line: str
    root_exit: int
    wall_ms: float
    eval_tokens_s: float
    pred_tokens_s: float
    post_recovery_tokens_s: float
    avg_pred_ms_after_apply: float
    reject_count: int
    output_dir: str


@dataclass
class TransferRow:
    run_id: str
    model: str
    context_len: int
    bandwidth_mbps: int
    repeat: int
    state_bytes: int
    t_transfer_ms: float
    transfer_mbps: float
    ok: bool
    output_dir: str


live_rows = []
transfer_rows = []


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    SOCK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    design = {
        "run_id": RUN_ID,
        "purpose": "Find boundary conditions where Shadow KV/EdgeVisor redundancy is not a universal win.",
        "topology": "8 Docker containers, active 3/3/2 VG layout, ratios=1:1:1*1:1:1*1:1",
        "high_stable_bandwidth_mbps": HIGH_STABLE_BW_MBPS,
        "live_contexts": LIVE_CONTEXTS,
        "micro_contexts": MICRO_CONTEXTS,
        "micro_bandwidths_mbps": MICRO_BWS,
        "shadow_kv_redundancy": SHADOW_KV_REDUNDANCY,
        "no_shadow_kv_redundancy": NO_SHADOW_KV_REDUNDANCY,
        "migration": "Stage1 Node4 -> Node3, move 1 KV/GQA head via UDS next_barrier; no perturbation is injected.",
        "models": suite.MODELS,
        "notes": [
            "with_shadow_kv avoids state transfer but keeps redundant KV compute enabled.",
            "without_shadow_transfer disables extra KV compute with kv-redundancy=0, performs real TCP transfer, then issues the same UDS migration.",
            "static_no_edgevisor disables the dynamic EdgeVisor path and measures the stable high-bandwidth baseline.",
            "Model text quality is not evaluated; apply/run completion and timing are the metrics.",
        ],
    }
    (OUT / "design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")


def write_dataclass_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(asdict(rows[0]).keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def compute_comparisons():
    by_key = {}
    for row in live_rows:
        by_key[(row.model, row.context_len, row.bandwidth_mbps, row.variant)] = row
    rows = []
    for model in MODELS:
        for ctx in LIVE_CONTEXTS:
            w = by_key.get((model, ctx, HIGH_STABLE_BW_MBPS, "with_shadow_kv"))
            n = by_key.get((model, ctx, HIGH_STABLE_BW_MBPS, "without_shadow_transfer"))
            srow = by_key.get((model, ctx, HIGH_STABLE_BW_MBPS, "static_no_edgevisor"))
            if not w or not n:
                continue
            no_shadow_compute_wall = max(0.0, n.wall_ms - n.t_state_ms)
            redundancy_overhead = w.wall_ms - no_shadow_compute_wall
            net_delta = w.wall_ms - n.wall_ms
            static_delta = (w.wall_ms - srow.wall_ms) if srow else ""
            w_ts = w.pred_tokens_s if w.pred_tokens_s > 0 else w.post_recovery_tokens_s
            n_ts = n.pred_tokens_s if n.pred_tokens_s > 0 else n.post_recovery_tokens_s
            s_ts = (srow.pred_tokens_s if srow and srow.pred_tokens_s > 0 else (srow.post_recovery_tokens_s if srow else 0.0))
            threshold_ctx = ""
            if n.t_state_ms > 0 and redundancy_overhead > 0:
                threshold_ctx = redundancy_overhead * ctx / n.t_state_ms
            rows.append({
                "model": model,
                "context_len": ctx,
                "bandwidth_mbps": HIGH_STABLE_BW_MBPS,
                "with_shadow_wall_ms": round(w.wall_ms, 3),
                "without_shadow_transfer_wall_ms": round(n.wall_ms, 3),
                "without_shadow_state_ms": round(n.t_state_ms, 3),
                "net_delta_ms_positive_means_edgevisor_slower": round(net_delta, 3),
                "redundancy_overhead_ms_est": round(redundancy_overhead, 3),
                "threshold_context_tokens_est": round(threshold_ctx, 1) if threshold_ctx != "" else "",
                "with_shadow_effective_tokens_s": round(w_ts, 4),
                "without_shadow_effective_tokens_s": round(n_ts, 4),
                "throughput_delta_pct": round(((w_ts - n_ts) / n_ts) * 100.0, 3) if n_ts else "",
                "static_no_edgevisor_wall_ms": round(srow.wall_ms, 3) if srow else "",
                "edgevisor_vs_static_delta_ms": round(static_delta, 3) if static_delta != "" else "",
                "static_effective_tokens_s": round(s_ts, 4) if srow else "",
                "edgevisor_vs_static_throughput_delta_pct": round(((w_ts - s_ts) / s_ts) * 100.0, 3) if srow and s_ts else "",
                "verdict": "EdgeVisor loses vs no-shadow transfer" if net_delta > 0 else ("EdgeVisor loses vs static baseline" if static_delta != "" and static_delta > 0 else "EdgeVisor wins/ties"),
                "with_apply_seen": w.apply_seen,
                "without_apply_seen": n.apply_seen,
                "with_root_exit": w.root_exit,
                "without_root_exit": n.root_exit,
            })
    return rows


def write_csvs():
    write_dataclass_csv(OUT / "live_runs.csv", live_rows)
    write_dataclass_csv(OUT / "transfer_microbench.csv", transfer_rows)
    comparisons = compute_comparisons()
    if comparisons:
        fields = list(comparisons[0].keys())
        with (OUT / "boundary_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparisons)
    else:
        (OUT / "boundary_summary.csv").write_text("", encoding="utf-8")
    summary = {
        "run_id": RUN_ID,
        "output": str(OUT),
        "live_rows": len(live_rows),
        "transfer_rows": len(transfer_rows),
        "comparisons": len(comparisons),
        "failed_live_rows": [asdict(r) for r in live_rows if r.root_exit != 0 or not r.apply_seen],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def latest_apply_pos(line):
    m = re.search(r"pos=([0-9]+)", line or "")
    return int(m.group(1)) if m else None


def run_live_case(model_name, variant, context_len):
    kv_red = SHADOW_KV_REDUNDANCY if variant == "with_shadow_kv" else NO_SHADOW_KV_REDUNDANCY
    label = f"{variant}_ctx{context_len}_bw{HIGH_STABLE_BW_MBPS}"
    max_seq_len = 2048
    if variant == "static_no_edgevisor":
        extra = f"--kv-redundancy {kv_red}"
    else:
        extra = (
            "--enable-plan-barrier "
            "--enable-stage-full-weights "
            "--enable-kv-redundancy-during-migration 1 "
            f"--kv-redundancy {kv_red}"
        )
    case_start = time.time()
    meta = suite.start_cluster(model_name, label, EXP, steps=LIVE_STEPS, max_seq_len=max_seq_len, extra_args=extra)
    test_dir = Path(meta["test_dir"])
    sock = Path(meta["sock_host"])
    state_bytes = suite.kv_state_bytes(suite.MODELS[model_name], context_len, moved_kv_heads=1)
    t_state = 0.0
    transfer_mbps = 0.0
    t_cmd = 0.0
    t_recover = 0.0
    apply_seen = False
    apply_line = ""
    root_exit = 999
    eval_ts = pred_ts = post_ts = avg_post = 0.0
    rejects = 0
    try:
        if variant == "static_no_edgevisor":
            wait = suite.run(["docker", "wait", meta["root"]], check=False, timeout=1200 if model_name == "8B" else 900)
            root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
            root_log = suite.read_logs(meta["root"])
            eval_ts, pred_ts, post_ts, avg_post, rejects = suite.parse_root_metrics(root_log, None)
            apply_seen = True
            apply_line = "static baseline: no UDS migration"
            return
        if not suite.wait_for_uds(sock, timeout_s=300):
            raise RuntimeError(f"UDS not ready: {sock}")
        if not suite.wait_for_position(meta["root"], 5, timeout_s=600 if model_name == "8B" else 420):
            raise RuntimeError("root did not reach pos 5 before boundary event")
        event_start = time.time()
        state_log = ""
        if variant == "without_shadow_transfer":
            t_state, ok, state_log = suite.transfer_bytes(meta["all_names"][4], meta["all_names"][3], state_bytes, port=27000 + (context_len % 1000))
            transfer_mbps = (state_bytes * 8 / 1e6) / (t_state / 1000.0) if t_state > 0 else 0.0
            if not ok:
                state_log += "\nTRANSFER_FAILED\n"
        _, t_cmd, cmd_out = suite.uds_set_plan(sock, 1, 4, 3, stage=1, head=1, ffn=0, mode="next_barrier")
        apply_seen, t_recover, apply_line = suite.poll_apply_any(meta["all_names"], 1, event_start, timeout_s=180)
        (test_dir / "state_transfer.log").write_text(state_log, encoding="utf-8", errors="ignore")
        (test_dir / "cmd.json").write_text(cmd_out, encoding="utf-8", errors="ignore")
        wait = suite.run(["docker", "wait", meta["root"]], check=False, timeout=1200 if model_name == "8B" else 900)
        root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
        root_log = suite.read_logs(meta["root"])
        eval_ts, pred_ts, post_ts, avg_post, rejects = suite.parse_root_metrics(root_log, latest_apply_pos(apply_line))
    finally:
        wall_ms = (time.time() - case_start) * 1000.0
        try:
            suite.stop_cluster(meta)
        finally:
            live_rows.append(LiveRow(
                RUN_ID, model_name, variant, context_len, HIGH_STABLE_BW_MBPS, kv_red,
                LIVE_STEPS, max_seq_len, state_bytes, float(t_state), float(transfer_mbps),
                float(t_cmd), float(t_recover), float(t_state + t_cmd), bool(apply_seen),
                apply_line, int(root_exit), float(wall_ms), float(eval_ts), float(pred_ts),
                float(post_ts), float(avg_post), int(rejects), str(test_dir)
            ))
            write_csvs()


def start_net_only(prefix):
    suite.cleanup_prefix(prefix)
    net = f"{prefix}_net"
    suite.run(["docker", "network", "create", net])
    names = []
    for i in range(8):
        name = f"{prefix}_n{i}"
        suite.run([
            "docker", "run", "-d", "--name", name, "--network", net,
            "--cap-add", "NET_ADMIN", "--cpus", str(suite.CPU_LIMITS[i]),
            "-v", f"{suite.PROJ}:/workspace/EdgeVisor",
            "-w", "/workspace/EdgeVisor",
            suite.IMAGE,
            "bash", "-lc", "sleep infinity",
        ])
        names.append(name)
    return {"net": net, "names": names}


def stop_net_only(meta, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in meta["names"]:
        (out_dir / f"{name}.log").write_text(suite.read_logs(name), encoding="utf-8", errors="ignore")
    suite.run(["docker", "rm", "-f"] + meta["names"], check=False)
    suite.run(["docker", "network", "rm", meta["net"]], check=False)


def run_transfer_microbench(model_name):
    prefix = f"b01ab_boundary_xfer_{model_name.lower()}_{int(time.time())}"
    out_dir = OUT / "transfer_microbench" / model_name
    meta = start_net_only(prefix)
    try:
        for bw in MICRO_BWS:
            for name in meta["names"]:
                suite.apply_tc(name, bw)
            time.sleep(0.2)
            for ctx in MICRO_CONTEXTS:
                nbytes = suite.kv_state_bytes(suite.MODELS[model_name], ctx, moved_kv_heads=1)
                for rep in range(1, MICRO_REPEATS + 1):
                    port = 28000 + (ctx % 1000) + rep + (0 if model_name == "3B" else 100)
                    t_ms, ok, log = suite.transfer_bytes(meta["names"][4], meta["names"][3], nbytes, port=port)
                    mbps = (nbytes * 8 / 1e6) / (t_ms / 1000.0) if t_ms > 0 else 0.0
                    transfer_rows.append(TransferRow(RUN_ID, model_name, ctx, bw, rep, nbytes, float(t_ms), float(mbps), bool(ok), str(out_dir)))
                    (out_dir / f"xfer_{model_name}_{bw}mbps_ctx{ctx}_rep{rep}.log").parent.mkdir(parents=True, exist_ok=True)
                    (out_dir / f"xfer_{model_name}_{bw}mbps_ctx{ctx}_rep{rep}.log").write_text(log, encoding="utf-8", errors="ignore")
                    write_csvs()
    finally:
        stop_net_only(meta, out_dir)
        write_csvs()


def read_csv(path):
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return []
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col_name(idx):
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def sheet_xml(matrix):
    rows = []
    for r_idx, row in enumerate(matrix, start=1):
        cells = []
        for c_idx, val in enumerate(row):
            if val is None:
                continue
            ref = f"{col_name(c_idx)}{r_idx}"
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                cells.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(val))}</t></is></c>')
        rows.append(f'<row r="{r_idx}">' + "".join(cells) + "</row>")
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(rows) + '</sheetData></worksheet>'


def csv_matrix(path):
    rows = read_csv(path)
    if not rows:
        return [["empty"]]
    headers = list(rows[0].keys())
    matrix = [headers]
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            try:
                if v != "" and re.fullmatch(r"-?\d+", str(v)):
                    vals.append(int(v))
                elif v != "" and re.fullmatch(r"-?\d+\.\d+(e[+-]?\d+)?", str(v), re.I):
                    vals.append(float(v))
                else:
                    vals.append(v)
            except Exception:
                vals.append(v)
        matrix.append(vals)
    return matrix


def write_xlsx(path, sheets):
    safe_sheets = []
    used = set()
    for name, matrix in sheets:
        safe = re.sub(r"[\\/?*:\[\]]", "_", name)[:31] or "Sheet"
        base = safe
        i = 2
        while safe in used:
            suffix = f"_{i}"
            safe = (base[:31 - len(suffix)] + suffix)[:31]
            i += 1
        used.add(safe)
        safe_sheets.append((safe, matrix))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(safe_sheets) + 1)) + '</Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        workbook_sheets = "".join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(safe_sheets, start=1))
        z.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + workbook_sheets + '</sheets></workbook>')
        rels = "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(safe_sheets) + 1))
        z.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + rels + '</Relationships>')
        for i, (_, matrix) in enumerate(safe_sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(matrix))


def avg(vals):
    vals = [float(v) for v in vals if v not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def make_report():
    write_csvs()
    live = read_csv(OUT / "live_runs.csv")
    summary = read_csv(OUT / "boundary_summary.csv")
    xfer = read_csv(OUT / "transfer_microbench.csv")
    xfer_summary = []
    grouped = {}
    for row in xfer:
        key = (row["model"], int(row["context_len"]), int(row["bandwidth_mbps"]))
        grouped.setdefault(key, []).append(row)
    for (model, ctx, bw), rows in sorted(grouped.items()):
        xfer_summary.append({
            "model": model,
            "context_len": ctx,
            "bandwidth_mbps": bw,
            "state_MB": round(float(rows[0]["state_bytes"]) / 1024 / 1024, 3),
            "avg_transfer_ms": round(avg([r["t_transfer_ms"] for r in rows]), 3),
            "avg_transfer_mbps": round(avg([r["transfer_mbps"] for r in rows]), 3),
            "repeats": len(rows),
        })
    if xfer_summary:
        fields = list(xfer_summary[0].keys())
        with (OUT / "transfer_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(xfer_summary)

    readme = [
        ["item", "value"],
        ["run_id", RUN_ID],
        ["remote_output", str(OUT)],
        ["topology", "8 Docker containers, 3/3/2 VG, ratios=1:1:1*1:1:1*1:1"],
        ["live_condition", f"stable high bandwidth {HIGH_STABLE_BW_MBPS} Mbps, no injected straggler/wave"],
        ["migration", "Stage1 Node4 -> Node3, move 1 KV/GQA head"],
        ["shadow_variant", f"kv-redundancy={SHADOW_KV_REDUNDANCY}"],
        ["no_shadow_variant", f"kv-redundancy={NO_SHADOW_KV_REDUNDANCY} plus real TCP transfer"],
        ["live_rows", len(live)],
        ["transfer_rows", len(xfer)],
    ]
    sheets = [
        ("README", readme),
        ("Boundary Summary", csv_matrix(OUT / "boundary_summary.csv")),
        ("Live Runs", csv_matrix(OUT / "live_runs.csv")),
        ("Transfer Summary", csv_matrix(OUT / "transfer_summary.csv")),
        ("Transfer Raw", csv_matrix(OUT / "transfer_microbench.csv")),
        ("Design", [["json"], [json.dumps(json.loads((OUT / "design.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2)]]),
    ]
    xlsx = OUT / "edgevisor_boundary_condition_results.xlsx"
    write_xlsx(xlsx, sheets)

    md = []
    md.append("# EdgeVisor Boundary Condition Experiment\n")
    md.append(f"- Run: `{RUN_ID}`")
    md.append(f"- Remote output: `{OUT}`")
    md.append("- Setup: 8 Docker containers, active `3/3/2` VG layout, `ratios=1:1:1*1:1:1*1:1`.")
    md.append(f"- Condition: stable high bandwidth `{HIGH_STABLE_BW_MBPS} Mbps`, short-context state sizes, no injected network/CPU perturbation.")
    md.append("- Migration: Stage 1 `Node4 -> Node3`, move 1 KV/GQA head via UDS `next_barrier`.")
    md.append(f"- Variants: `with_shadow_kv` uses `kv-redundancy={SHADOW_KV_REDUNDANCY}`; `without_shadow_transfer` uses `kv-redundancy={NO_SHADOW_KV_REDUNDANCY}` and performs real Docker TCP KV-state transfer before migration; `static_no_edgevisor` disables plan barrier/full-weight residency/Shadow KV under the same stable 3/3/2 topology.")
    md.append("\n## Boundary Summary\n")
    if summary:
        headers = ["model", "context_len", "net_delta_ms_positive_means_edgevisor_slower", "edgevisor_vs_static_delta_ms", "redundancy_overhead_ms_est", "threshold_context_tokens_est", "throughput_delta_pct", "edgevisor_vs_static_throughput_delta_pct", "verdict"]
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in summary:
            md.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    else:
        md.append("No complete with/without pairs were recorded.")
    md.append("\n## Interpretation\n")
    losses = [r for r in summary if "loses" in str(r.get("verdict"))]
    if losses:
        md.append("In the short-context/high-bandwidth rows above, `net_delta_ms_positive_means_edgevisor_slower > 0` means the Shadow-KV configuration finished slower than the no-shadow transfer path; `edgevisor_vs_static_delta_ms > 0` means it is slower than the stable no-EdgeVisor baseline. This is the desired non-universal-win boundary: the one-time KV transfer is cheap enough that continuous redundant KV computation/loading dominates.")
    else:
        md.append("No live row showed a strict EdgeVisor loss under the selected comparison. Use the transfer summary and threshold estimates to locate the crossover, or rerun with a higher `B01_BOUNDARY_SHADOW_KV` value to stress redundant KV maintenance.")
    md.append("\nThe `threshold_context_tokens_est` column estimates where the avoided transfer cost would equal the measured redundancy overhead. Contexts below that estimate are the region where no-shadow transfer can be competitive or faster under this stable high-bandwidth setup.")
    md.append("\n## Files\n")
    md.append("- `edgevisor_boundary_condition_results.xlsx`: Excel-compatible workbook with summary, live runs, transfer summary, raw transfer rows, and design JSON.")
    md.append("- `live_runs.csv`: live distributed inference timing rows.")
    md.append("- `transfer_microbench.csv`: repeated real TCP state transfer measurements.")
    md.append("- `boundary_summary.csv`: paired with/without comparison and estimated boundary.")
    (OUT / "edgevisor_boundary_condition_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"REPORT {xlsx}")


def main():
    ensure_dirs()
    suite.cleanup_prefix("b01ab_")
    print(f"OUT={OUT}", flush=True)
    build = suite.run([
        "docker", "run", "--rm", "--name", f"b01ab_boundary_build_{int(time.time())}",
        "-v", f"{suite.PROJ}:/workspace/EdgeVisor",
        "-w", "/workspace/EdgeVisor",
        suite.IMAGE,
        "bash", "-lc", "make -j$(nproc) dllama",
    ], check=False, timeout=900)
    (OUT / "build.log").write_text(build.stdout or "", encoding="utf-8", errors="ignore")
    if build.returncode != 0:
        print("BUILD_FAILED", file=sys.stderr)
        write_csvs()
        sys.exit(build.returncode)
    print("BUILD_OK", flush=True)
    try:
        for model in MODELS:
            for ctx in LIVE_CONTEXTS:
                for variant in ["with_shadow_kv", "without_shadow_transfer", "static_no_edgevisor"]:
                    print(f"RUN live {model} ctx={ctx} {variant}", flush=True)
                    run_live_case(model, variant, ctx)
        for model in MODELS:
            print(f"RUN transfer microbench {model}", flush=True)
            run_transfer_microbench(model)
    finally:
        suite.cleanup_prefix("b01ab_")
        write_csvs()
        make_report()
    print(f"DONE OUT={OUT}", flush=True)


if __name__ == "__main__":
    main()

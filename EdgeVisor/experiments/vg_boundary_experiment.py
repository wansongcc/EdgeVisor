#!/usr/bin/env python3
"""VG boundary validation experiment.

Goal: validate the claim that devices should form a VG only inside a boundary:
when intra-group TP gains exceed communication/straggler costs. Outside that
boundary, exposing devices as inter-VG PP stages can be better.

Persistent outputs stay under /home/cc/yhbian/B01.
"""

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import edgevisor_ablation_suite as suite

RUN_ID = time.strftime("vg_boundary_%Y%m%d_%H%M%S")
suite.RUN_ID = RUN_ID
suite.OUT = suite.BASE / "vg_boundary_results" / RUN_ID
suite.SOCK_DIR = suite.OUT / "s"
OUT = suite.OUT

# Keep this intentionally small enough to run as a validation experiment.
MODEL = os.environ.get("B01_VG_BOUNDARY_MODEL", "3B")
STEPS = int(os.environ.get("B01_VG_BOUNDARY_STEPS", "20"))
MAX_SEQ_LEN = int(os.environ.get("B01_VG_BOUNDARY_MAX_SEQ", "1024"))
PROMPT = os.environ.get("B01_VG_BOUNDARY_PROMPT", "The capital of France is")
ROOT_THREADS = float(os.environ.get("B01_VG_BOUNDARY_ROOT_THREADS", "2"))
WORKER_THREADS = float(os.environ.get("B01_VG_BOUNDARY_WORKER_THREADS", "2"))

# 3B has 28 layers. Pure PP variants use 8 single-node stages.
VG_332_RATIOS = "1:1:1*1:1:1*1:1"
PP_BALANCED_3B = "1@4*1@4*1@4*1@4*1@3*1@3*1@3*1@3"
# Node4 is the weak node in the high-skew scenario, so give it only one layer.
PP_NODE4_LIGHT_3B = "1@4*1@4*1@4*1@4*1@1*1@4*1@4*1@3"

BALANCED_CPUS = [2.0] * 8
NODE4_SKEW_CPUS = [2.0, 2.0, 2.0, 2.0, 0.30, 2.0, 2.0, 2.0]

SCENARIOS = [
    {
        "scenario": "high_bw_low_skew",
        "bandwidth_mbps": 1000,
        "cpu_limits": BALANCED_CPUS,
        "pp_ratios": PP_BALANCED_3B,
        "hypothesis": "VG should be competitive or better: TP communication is cheap and skew is small.",
    },
    {
        "scenario": "low_bw_low_skew",
        "bandwidth_mbps": 10,
        "cpu_limits": BALANCED_CPUS,
        "pp_ratios": PP_BALANCED_3B,
        "hypothesis": "PP should become competitive/better: TP has more intra-stage communication pressure.",
    },
    {
        "scenario": "high_bw_high_skew",
        "bandwidth_mbps": 1000,
        "cpu_limits": NODE4_SKEW_CPUS,
        "pp_ratios": PP_NODE4_LIGHT_3B,
        "hypothesis": "PP should be better: a very slow node inside a VG becomes a TP straggler; PP can isolate it with fewer layers.",
    },
]


@dataclass
class RunRow:
    run_id: str
    scenario: str
    topology: str
    model: str
    ratios: str
    bandwidth_mbps: int
    cpu_limits: str
    steps: int
    max_seq_len: int
    root_exit: int
    wall_ms: float
    eval_tokens_s: float
    pred_tokens_s: float
    avg_pred_ms: float
    reject_count: int
    output_dir: str
    hypothesis: str


rows = []


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    suite.SOCK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "design.json").write_text(json.dumps({
        "run_id": RUN_ID,
        "purpose": "Validate VG boundary: 3/3/2 VG vs pure/inter-VG PP under bandwidth and compute-skew changes.",
        "model": MODEL,
        "steps": STEPS,
        "max_seq_len": MAX_SEQ_LEN,
        "topologies": {
            "vg_332": VG_332_RATIOS,
            "pp_balanced_3b": PP_BALANCED_3B,
            "pp_node4_light_3b": PP_NODE4_LIGHT_3B,
        },
        "scenarios": SCENARIOS,
        "notes": [
            "No dynamic migration is used here; this isolates the topology decision: form VG for intra-stage TP or degrade to inter-VG PP.",
            "All runs use the same 8 Docker containers and the same 3B model; only ratios, bandwidth, and CPU limits differ.",
            "The goal is trend validation, not a final paper-quality sweep.",
        ],
    }, indent=2), encoding="utf-8")


def write_csvs():
    if rows:
        fields = list(asdict(rows[0]).keys())
        with (OUT / "runs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow(asdict(row))
    else:
        (OUT / "runs.csv").write_text("", encoding="utf-8")

    comparisons = []
    by = {(r.scenario, r.topology): r for r in rows}
    for sc in SCENARIOS:
        s = sc["scenario"]
        vg = by.get((s, "vg_332"))
        pp = by.get((s, "pp_fallback"))
        if not vg or not pp:
            continue
        vg_ts = vg.pred_tokens_s or (1000.0 / vg.avg_pred_ms if vg.avg_pred_ms else 0.0)
        pp_ts = pp.pred_tokens_s or (1000.0 / pp.avg_pred_ms if pp.avg_pred_ms else 0.0)
        wall_delta = vg.wall_ms - pp.wall_ms
        ts_delta_pct = ((vg_ts - pp_ts) / pp_ts * 100.0) if pp_ts else 0.0
        if vg.root_exit != 0 or pp.root_exit != 0:
            verdict = "invalid_or_failed_run"
        elif wall_delta < 0 and ts_delta_pct >= -5:
            verdict = "VG better/competitive"
        elif wall_delta > 0 and ts_delta_pct < 0:
            verdict = "PP better"
        elif wall_delta > 0:
            verdict = "PP lower wall time"
        else:
            verdict = "mixed"
        comparisons.append({
            "scenario": s,
            "bandwidth_mbps": sc["bandwidth_mbps"],
            "cpu_limits": ",".join(str(x) for x in sc["cpu_limits"]),
            "vg_wall_ms": round(vg.wall_ms, 3),
            "pp_wall_ms": round(pp.wall_ms, 3),
            "vg_minus_pp_wall_ms": round(wall_delta, 3),
            "vg_pred_tokens_s": round(vg_ts, 5),
            "pp_pred_tokens_s": round(pp_ts, 5),
            "vg_vs_pp_tokens_s_delta_pct": round(ts_delta_pct, 3),
            "verdict": verdict,
            "hypothesis": sc["hypothesis"],
        })
    if comparisons:
        fields = list(comparisons[0].keys())
        with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(comparisons)
    else:
        (OUT / "summary.csv").write_text("", encoding="utf-8")

    (OUT / "summary.json").write_text(json.dumps({
        "run_id": RUN_ID,
        "output": str(OUT),
        "rows": len(rows),
        "comparisons": comparisons,
        "failed_rows": [asdict(r) for r in rows if r.root_exit != 0],
    }, indent=2), encoding="utf-8")

    lines = [
        "# VG Boundary Validation Experiment",
        "",
        f"- Run: `{RUN_ID}`",
        f"- Output: `{OUT}`",
        f"- Model: `{MODEL}`",
        f"- Steps: `{STEPS}`",
        "- Question: when should devices form a VG for intra-stage TP, and when is inter-VG PP better?",
        "",
        "## Summary",
        "",
    ]
    if comparisons:
        lines.append("| scenario | bandwidth Mbps | VG wall ms | PP wall ms | VG-PP wall ms | VG tok/s | PP tok/s | tok/s delta % | verdict |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for c in comparisons:
            lines.append(
                f"| {c['scenario']} | {c['bandwidth_mbps']} | {c['vg_wall_ms']} | {c['pp_wall_ms']} | {c['vg_minus_pp_wall_ms']} | "
                f"{c['vg_pred_tokens_s']} | {c['pp_pred_tokens_s']} | {c['vg_vs_pp_tokens_s_delta_pct']} | {c['verdict']} |"
            )
    else:
        lines.append("No complete comparisons yet.")
    lines += [
        "",
        "## Interpretation Guide",
        "",
        "- If VG wins at high bandwidth/low skew, that supports VG as useful inside a bandwidth/compute boundary.",
        "- If PP wins at low bandwidth or high compute skew, that supports the fallback-to-PP boundary condition.",
        "- This is a validation sweep, not a final exhaustive paper experiment.",
    ]
    (OUT / "vg_boundary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(scenario, topology, ratios):
    suite.RATIOS = ratios
    suite.BASE_NET_MBPS = [int(scenario["bandwidth_mbps"])] * 8
    suite.CPU_LIMITS = list(scenario["cpu_limits"])
    suite.ROOT_THREADS = int(ROOT_THREADS)
    suite.WORKER_THREADS = int(WORKER_THREADS)
    variant = f"{scenario['scenario']}_{topology}"
    meta = suite.start_cluster(
        MODEL,
        variant,
        "vg_boundary",
        steps=STEPS,
        max_seq_len=MAX_SEQ_LEN,
        extra_args="",
        prompt=PROMPT,
    )
    test_dir = Path(meta["test_dir"])
    t0 = time.time()
    root_exit = 999
    try:
        wait = suite.run(["docker", "wait", meta["root"]], check=False, timeout=900)
        root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
        wall_ms = (time.time() - t0) * 1000.0
        root_log = suite.read_logs(meta["root"])
        eval_ts, pred_ts, _post_ts, avg_pred, rejects = suite.parse_root_metrics(root_log)
        rows.append(RunRow(
            run_id=RUN_ID,
            scenario=scenario["scenario"],
            topology=topology,
            model=MODEL,
            ratios=ratios,
            bandwidth_mbps=int(scenario["bandwidth_mbps"]),
            cpu_limits=",".join(str(x) for x in scenario["cpu_limits"]),
            steps=STEPS,
            max_seq_len=MAX_SEQ_LEN,
            root_exit=root_exit,
            wall_ms=wall_ms,
            eval_tokens_s=eval_ts,
            pred_tokens_s=pred_ts,
            avg_pred_ms=avg_pred,
            reject_count=rejects,
            output_dir=str(test_dir),
            hypothesis=scenario["hypothesis"],
        ))
        write_csvs()
    finally:
        suite.stop_cluster(meta)
        write_csvs()


def main():
    ensure_dirs()
    suite.cleanup_prefix("b01ab_")
    print(f"OUT={OUT}", flush=True)
    # Reuse the existing built dllama binary; build is intentionally skipped to keep this validation quick.
    try:
        for scenario in SCENARIOS:
            print(f"RUN {scenario['scenario']} vg_332", flush=True)
            run_one(scenario, "vg_332", VG_332_RATIOS)
            print(f"RUN {scenario['scenario']} pp_fallback", flush=True)
            run_one(scenario, "pp_fallback", scenario["pp_ratios"])
    finally:
        suite.cleanup_prefix("b01ab_")
        write_csvs()
    print(f"DONE OUT={OUT}", flush=True)


if __name__ == "__main__":
    main()

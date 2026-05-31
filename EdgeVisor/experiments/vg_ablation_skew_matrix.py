#!/usr/bin/env python3
"""VG ablation skew matrix.

Validates VG boundary with three stressors:
- compute skew: a slow device inside VG becomes a TP straggler.
- bandwidth skew: weak links to/from that device punish VG-internal TP.
- combined skew: slow compute + weak network.

Each scenario compares 3/3/2 VG against a PP fallback that gives the weak node
only one layer. Persistent outputs stay under /home/cc/yhbian/B01.
"""

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import edgevisor_ablation_suite as suite

RUN_ID = time.strftime("vg_ablation_skew_%Y%m%d_%H%M%S")
suite.RUN_ID = RUN_ID
suite.OUT = suite.BASE / "vg_boundary_results" / RUN_ID
suite.SOCK_DIR = suite.OUT / "s"
OUT = suite.OUT

MODEL = "3B"
STEPS = 8
MAX_SEQ_LEN = 1024
PROMPT = "The capital of France is"
WEAK_NODE = 4
HIGH_BW_MBPS = 1000
WEAK_BW_MBPS = 2
WEAK_DELAY_MS = 80

VG_332 = "1:1:1*1:1:1*1:1"
PP_BALANCED = "1@4*1@4*1@4*1@4*1@3*1@3*1@3*1@3"
PP_WEAK_NODE_LIGHT = "1@4*1@4*1@4*1@4*1@1*1@4*1@4*1@3"

BALANCED_CPUS = [2.0] * 8
COMPUTE_SKEW_CPUS = [2.0, 2.0, 2.0, 2.0, 0.30, 2.0, 2.0, 2.0]

SCENARIOS = [
    {
        "scenario": "no_skew_reference",
        "cpu_limits": BALANCED_CPUS,
        "weak_network": False,
        "pp_ratios": PP_BALANCED,
        "hypothesis": "VG should win/compete when bandwidth is high and compute skew is small.",
    },
    {
        "scenario": "compute_skew",
        "cpu_limits": COMPUTE_SKEW_CPUS,
        "weak_network": False,
        "pp_ratios": PP_WEAK_NODE_LIGHT,
        "hypothesis": "PP fallback should improve when a very slow device becomes a VG-internal TP straggler.",
    },
    {
        "scenario": "bandwidth_skew",
        "cpu_limits": BALANCED_CPUS,
        "weak_network": True,
        "pp_ratios": PP_WEAK_NODE_LIGHT,
        "hypothesis": "PP fallback should improve if traffic to/from a VG-internal device is weak and PP can reduce that device's work.",
    },
    {
        "scenario": "compute_and_bandwidth_skew",
        "cpu_limits": COMPUTE_SKEW_CPUS,
        "weak_network": True,
        "pp_ratios": PP_WEAK_NODE_LIGHT,
        "hypothesis": "PP fallback should be best when compute and bandwidth skews hit the same VG-internal device.",
    },
]

@dataclass
class Row:
    run_id: str
    scenario: str
    topology: str
    model: str
    ratios: str
    weak_node: int
    weak_network: bool
    weak_bw_mbps: int
    weak_delay_ms: int
    cpu_limits: str
    steps: int
    root_exit: int
    wall_ms: float
    avg_pred_ms: float
    effective_tokens_s: float
    reject_count: int
    output_dir: str
    hypothesis: str

rows = []


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    suite.SOCK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "design.json").write_text(json.dumps({
        "run_id": RUN_ID,
        "purpose": "VG ablation: no skew, compute skew, bandwidth skew, and combined skew.",
        "model": MODEL,
        "steps": STEPS,
        "max_seq_len": MAX_SEQ_LEN,
        "vg_topology": VG_332,
        "pp_balanced": PP_BALANCED,
        "pp_weak_node_light": PP_WEAK_NODE_LIGHT,
        "weak_node": WEAK_NODE,
        "high_bw_mbps": HIGH_BW_MBPS,
        "weak_bw_mbps": WEAK_BW_MBPS,
        "weak_delay_ms": WEAK_DELAY_MS,
        "scenarios": SCENARIOS,
        "network_note": "Weak-network scenarios use per-destination tc: all other containers shape egress traffic whose destination is weak-node IP; weak node shapes all egress.",
    }, indent=2), encoding="utf-8")


def docker_ip(name):
    p = suite.run(["docker", "inspect", "-f", "{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name], check=False)
    return (p.stdout or "").strip()


def apply_high_bw_all(meta):
    for name in meta["all_names"]:
        suite.apply_tc(name, HIGH_BW_MBPS)


def apply_weak_node_network(meta, weak_node=WEAK_NODE, weak_bw=WEAK_BW_MBPS, delay_ms=WEAK_DELAY_MS):
    names = meta["all_names"]
    weak_name = names[weak_node]
    weak_ip = docker_ip(weak_name)
    if not weak_ip:
        raise RuntimeError(f"cannot get docker IP for {weak_name}")
    tc_logs = []
    for idx, name in enumerate(names):
        if idx == weak_node:
            cmd = (
                "tc qdisc del dev eth0 root 2>/dev/null || true; "
                f"tc qdisc add dev eth0 root netem delay {delay_ms}ms rate {weak_bw}mbit; "
                "tc qdisc show dev eth0"
            )
        else:
            # Only traffic sent to the weak node is shaped. Other egress traffic is left on the default band.
            cmd = (
                "tc qdisc del dev eth0 root 2>/dev/null || true; "
                "tc qdisc add dev eth0 root handle 1: prio bands 3; "
                f"tc qdisc add dev eth0 parent 1:3 handle 30: netem delay {delay_ms}ms rate {weak_bw}mbit; "
                f"tc filter add dev eth0 protocol ip parent 1:0 prio 3 u32 match ip dst {weak_ip}/32 flowid 1:3; "
                "tc qdisc show dev eth0; tc filter show dev eth0 parent 1:0"
            )
        res = suite.docker_exec(name, cmd, check=False)
        tc_logs.append(f"=== {idx}:{name} ===\n{res.stdout}\nrc={res.returncode}\n")
    return weak_ip, "\n".join(tc_logs)


def write_outputs():
    if rows:
        fields = list(asdict(rows[0]).keys())
        with (OUT / "runs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
    else:
        (OUT / "runs.csv").write_text("", encoding="utf-8")

    by = {(r.scenario, r.topology): r for r in rows}
    comparisons = []
    for sc in SCENARIOS:
        s = sc["scenario"]
        vg = by.get((s, "vg_332"))
        pp = by.get((s, "pp_fallback"))
        if not vg or not pp:
            continue
        wall_delta = vg.wall_ms - pp.wall_ms
        tok_delta_pct = ((vg.effective_tokens_s - pp.effective_tokens_s) / pp.effective_tokens_s * 100.0) if pp.effective_tokens_s else 0.0
        if vg.root_exit != 0 or pp.root_exit != 0:
            verdict = "invalid_or_failed_run"
        elif wall_delta > 0 and tok_delta_pct < 0:
            verdict = "PP better"
        elif wall_delta < 0 and tok_delta_pct > 0:
            verdict = "VG better"
        elif wall_delta > 0:
            verdict = "PP lower wall"
        else:
            verdict = "mixed_or_close"
        comparisons.append({
            "scenario": s,
            "weak_network": sc["weak_network"],
            "cpu_limits": ",".join(str(x) for x in sc["cpu_limits"]),
            "vg_wall_ms": round(vg.wall_ms, 3),
            "pp_wall_ms": round(pp.wall_ms, 3),
            "vg_minus_pp_wall_ms": round(wall_delta, 3),
            "vg_avg_pred_ms": round(vg.avg_pred_ms, 3),
            "pp_avg_pred_ms": round(pp.avg_pred_ms, 3),
            "vg_tokens_s": round(vg.effective_tokens_s, 5),
            "pp_tokens_s": round(pp.effective_tokens_s, 5),
            "vg_vs_pp_tokens_s_delta_pct": round(tok_delta_pct, 3),
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
        "rows": [asdict(r) for r in rows],
        "comparisons": comparisons,
        "failed_rows": [asdict(r) for r in rows if r.root_exit != 0],
    }, indent=2), encoding="utf-8")

    lines = [
        "# VG Ablation Skew Matrix",
        "",
        f"- Run: `{RUN_ID}`",
        f"- Output: `{OUT}`",
        f"- Model: `{MODEL}`",
        f"- Steps: `{STEPS}`",
        "- Comparison: `3/3/2 VG` vs `PP fallback` with weak node assigned only one layer.",
        "",
        "## Summary",
        "",
    ]
    if comparisons:
        lines.append("| scenario | VG wall ms | PP wall ms | VG-PP wall ms | VG tok/s | PP tok/s | delta % | verdict |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for c in comparisons:
            lines.append(f"| {c['scenario']} | {c['vg_wall_ms']} | {c['pp_wall_ms']} | {c['vg_minus_pp_wall_ms']} | {c['vg_tokens_s']} | {c['pp_tokens_s']} | {c['vg_vs_pp_tokens_s_delta_pct']} | {c['verdict']} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Compute skew is injected by limiting Node4 to 0.30 CPU.",
        "- Bandwidth skew is injected with per-destination tc: traffic to/from Node4 is shaped to 2 Mbps + 80 ms delay.",
        "- This is a validation sweep for VG boundary, not a final exhaustive paper experiment.",
    ]
    (OUT / "vg_ablation_skew_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(sc, topology, ratios):
    suite.RATIOS = ratios
    suite.ACTIVE_WORKER_COUNT = 7
    suite.BASE_NET_MBPS = [HIGH_BW_MBPS] * 8
    suite.CPU_LIMITS = list(sc["cpu_limits"])
    suite.ROOT_THREADS = 2
    suite.WORKER_THREADS = 2
    variant = f"{sc['scenario']}_{topology}"
    meta = suite.start_cluster(MODEL, variant, "vg_ablation_skew", steps=STEPS, max_seq_len=MAX_SEQ_LEN, extra_args="", prompt=PROMPT)
    test_dir = Path(meta["test_dir"])
    t0 = time.time()
    root_exit = 999
    try:
        apply_high_bw_all(meta)
        if sc["weak_network"]:
            weak_ip, tc_log = apply_weak_node_network(meta)
            (test_dir / "weak_network_tc.log").write_text(f"weak_node={WEAK_NODE} weak_ip={weak_ip}\n" + tc_log, encoding="utf-8")
        wait = suite.run(["docker", "wait", meta["root"]], check=False, timeout=1200)
        root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
        wall_ms = (time.time() - t0) * 1000.0
        root_log = suite.read_logs(meta["root"])
        _eval_ts, pred_ts, _post_ts, avg_pred, rejects = suite.parse_root_metrics(root_log)
        effective = pred_ts if pred_ts else (1000.0 / avg_pred if avg_pred else 0.0)
        rows.append(Row(
            run_id=RUN_ID,
            scenario=sc["scenario"],
            topology=topology,
            model=MODEL,
            ratios=ratios,
            weak_node=WEAK_NODE,
            weak_network=bool(sc["weak_network"]),
            weak_bw_mbps=WEAK_BW_MBPS if sc["weak_network"] else HIGH_BW_MBPS,
            weak_delay_ms=WEAK_DELAY_MS if sc["weak_network"] else 0,
            cpu_limits=",".join(str(x) for x in sc["cpu_limits"]),
            steps=STEPS,
            root_exit=root_exit,
            wall_ms=wall_ms,
            avg_pred_ms=avg_pred,
            effective_tokens_s=effective,
            reject_count=rejects,
            output_dir=str(test_dir),
            hypothesis=sc["hypothesis"],
        ))
        write_outputs()
    finally:
        suite.stop_cluster(meta)
        write_outputs()


def main():
    ensure_dirs()
    suite.cleanup_prefix("b01ab_")
    print(f"OUT={OUT}", flush=True)
    try:
        for sc in SCENARIOS:
            print(f"RUN {sc['scenario']} vg_332", flush=True)
            run_one(sc, "vg_332", VG_332)
            print(f"RUN {sc['scenario']} pp_fallback", flush=True)
            run_one(sc, "pp_fallback", sc["pp_ratios"])
    finally:
        suite.cleanup_prefix("b01ab_")
        write_outputs()
    print(f"DONE OUT={OUT}", flush=True)

if __name__ == "__main__":
    main()

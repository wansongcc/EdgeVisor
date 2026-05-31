#!/usr/bin/env python3
"""Pure VG absorption ablation.

Question: if there is no VG to absorb a fluctuation locally, what happens?

Compare, under the same wave injected on Node4:
- with_vg_absorb: 3/3/2 VG, Node4 is inside Stage1 VG, absorb by UDS head migration 4->3.
- no_vg_pp_no_absorb: no VG, singleton PP stages, Node4 owns a PP stage and no local VG head migration is possible.

Runs compute, bandwidth, and combined skew. Outputs stay under /home/cc/yhbian/B01.
"""

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import edgevisor_ablation_suite as suite

RUN_ID = time.strftime("vg_absorb_%Y%m%d_%H%M%S")
suite.RUN_ID = RUN_ID
suite.OUT = suite.BASE / "vg_boundary_results" / RUN_ID
suite.SOCK_DIR = suite.OUT / "s"
OUT = suite.OUT

MODEL = "3B"
STEPS = 24
MAX_SEQ_LEN = 1024
PROMPT = "The capital of France is"
WEAK_NODE = 4
EVENT_POS = 8
HIGH_BW_MBPS = 1000
WEAK_BW_MBPS = 2
WEAK_DELAY_MS = 80
COMPUTE_SLOW_CPU = "0.30"

VG_332 = "1:1:1*1:1:1*1:1"
NO_VG_PP = "1@4*1@4*1@4*1@4*1@3*1@3*1@3*1@3"

SCENARIOS = [
    {"scenario": "compute_skew", "compute": True, "bandwidth": False},
    {"scenario": "bandwidth_skew", "compute": False, "bandwidth": True},
    {"scenario": "compute_and_bandwidth_skew", "compute": True, "bandwidth": True},
]

@dataclass
class Row:
    run_id: str
    scenario: str
    variant: str
    topology: str
    ratios: str
    root_exit: int
    steps: int
    event_pos: int
    weak_node: int
    compute_skew: bool
    bandwidth_skew: bool
    t_cmd_ms: float
    t_recover_ms: float
    apply_seen: bool
    wall_ms: float
    pre_wave_avg_ms: float
    post_wave_avg_ms: float
    effective_post_tokens_s: float
    reject_count: int
    apply_line: str
    output_dir: str

rows = []


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    suite.SOCK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "design.json").write_text(json.dumps({
        "run_id": RUN_ID,
        "purpose": "Pure VG ablation: with VG local absorption vs no-VG PP under identical waves.",
        "model": MODEL,
        "steps": STEPS,
        "event_pos": EVENT_POS,
        "weak_node": WEAK_NODE,
        "vg_ratios": VG_332,
        "no_vg_pp_ratios": NO_VG_PP,
        "scenarios": SCENARIOS,
        "with_vg_absorb": "3/3/2 VG; after wave on Node4, issue set_plan stage=1 from Node4 to Node3, moving 1 head.",
        "no_vg_pp_no_absorb": "No VG; singleton PP stages; same wave remains on Node4 stage, no local head migration.",
        "bandwidth_skew": f"per-destination tc to/from Node4: {WEAK_BW_MBPS} Mbps + {WEAK_DELAY_MS} ms delay",
    }, indent=2), encoding="utf-8")


def docker_ip(name):
    p = suite.run(["docker", "inspect", "-f", "{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name], check=False)
    return (p.stdout or "").strip()


def apply_high_bw_all(meta):
    for name in meta["all_names"]:
        suite.apply_tc(name, HIGH_BW_MBPS)


def apply_weak_node_network(meta):
    names = meta["all_names"]
    weak_name = names[WEAK_NODE]
    weak_ip = docker_ip(weak_name)
    logs = []
    for idx, name in enumerate(names):
        if idx == WEAK_NODE:
            cmd = (
                "tc qdisc del dev eth0 root 2>/dev/null || true; "
                f"tc qdisc add dev eth0 root netem delay {WEAK_DELAY_MS}ms rate {WEAK_BW_MBPS}mbit; "
                "tc qdisc show dev eth0"
            )
        else:
            cmd = (
                "tc qdisc del dev eth0 root 2>/dev/null || true; "
                "tc qdisc add dev eth0 root handle 1: prio bands 3; "
                f"tc qdisc add dev eth0 parent 1:3 handle 30: netem delay {WEAK_DELAY_MS}ms rate {WEAK_BW_MBPS}mbit; "
                f"tc filter add dev eth0 protocol ip parent 1:0 prio 3 u32 match ip dst {weak_ip}/32 flowid 1:3; "
                "tc qdisc show dev eth0; tc filter show dev eth0 parent 1:0"
            )
        res = suite.docker_exec(name, cmd, check=False)
        logs.append(f"=== {idx}:{name} ===\n{res.stdout}\nrc={res.returncode}\n")
    return weak_ip, "\n".join(logs)


def inject_wave(meta, scenario, test_dir):
    apply_high_bw_all(meta)
    if scenario["bandwidth"]:
        weak_ip, tc_log = apply_weak_node_network(meta)
        (test_dir / "weak_network_tc.log").write_text(f"weak_node={WEAK_NODE} weak_ip={weak_ip}\n" + tc_log, encoding="utf-8")
    if scenario["compute"]:
        suite.run(["docker", "update", "--cpus", COMPUTE_SLOW_CPU, meta["all_names"][WEAK_NODE]], check=False)


def pred_samples(log_text):
    out = []
    for line in log_text.splitlines():
        m = re.search(r"Pred\s+([0-9]+) ms Sync\s+([0-9]+) ms \| pos=([0-9]+)", line)
        if m:
            out.append((int(m.group(3)), int(m.group(1)) + int(m.group(2))))
    return out


def avg_ms(samples, pred):
    vals = [ms for pos, ms in samples if pred(pos)]
    return sum(vals) / len(vals) if vals else 0.0


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
    by = {(r.scenario, r.variant): r for r in rows}
    comps = []
    for sc in SCENARIOS:
        s = sc["scenario"]
        vg = by.get((s, "with_vg_absorb"))
        nv = by.get((s, "no_vg_pp_no_absorb"))
        if not vg or not nv:
            continue
        post_delta_pct = ((vg.effective_post_tokens_s - nv.effective_post_tokens_s) / nv.effective_post_tokens_s * 100.0) if nv.effective_post_tokens_s else 0.0
        wall_delta = vg.wall_ms - nv.wall_ms
        comps.append({
            "scenario": s,
            "with_vg_wall_ms": round(vg.wall_ms, 3),
            "no_vg_wall_ms": round(nv.wall_ms, 3),
            "with_minus_no_vg_wall_ms": round(wall_delta, 3),
            "with_vg_t_recover_ms": round(vg.t_recover_ms, 3),
            "with_vg_post_wave_avg_ms": round(vg.post_wave_avg_ms, 3),
            "no_vg_post_wave_avg_ms": round(nv.post_wave_avg_ms, 3),
            "with_vg_post_tokens_s": round(vg.effective_post_tokens_s, 5),
            "no_vg_post_tokens_s": round(nv.effective_post_tokens_s, 5),
            "with_vg_vs_no_vg_post_tokens_s_delta_pct": round(post_delta_pct, 3),
            "verdict": "VG absorption better" if post_delta_pct > 0 else "No-VG better/mixed",
            "with_apply_seen": vg.apply_seen,
            "with_root_exit": vg.root_exit,
            "no_vg_root_exit": nv.root_exit,
        })
    if comps:
        fields = list(comps[0].keys())
        with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(comps)
    (OUT / "summary.json").write_text(json.dumps({
        "run_id": RUN_ID,
        "output": str(OUT),
        "rows": [asdict(r) for r in rows],
        "comparisons": comps,
        "failed_rows": [asdict(r) for r in rows if r.root_exit != 0 or (r.variant == "with_vg_absorb" and not r.apply_seen)],
    }, indent=2), encoding="utf-8")
    lines = [
        "# Pure VG Absorption Ablation",
        "",
        f"- Run: `{RUN_ID}`",
        f"- Output: `{OUT}`",
        f"- Model: `{MODEL}`",
        f"- Steps: `{STEPS}`",
        "",
        "## Summary",
        "",
    ]
    if comps:
        lines.append("| scenario | with VG wall | no VG wall | with recover | with post ms | no VG post ms | with tok/s | no VG tok/s | delta % | verdict |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for c in comps:
            lines.append(f"| {c['scenario']} | {c['with_vg_wall_ms']} | {c['no_vg_wall_ms']} | {c['with_vg_t_recover_ms']} | {c['with_vg_post_wave_avg_ms']} | {c['no_vg_post_wave_avg_ms']} | {c['with_vg_post_tokens_s']} | {c['no_vg_post_tokens_s']} | {c['with_vg_vs_no_vg_post_tokens_s_delta_pct']} | {c['verdict']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "with_vg_absorb uses local Stage1 head migration to absorb Node4's wave inside the VG.",
        "no_vg_pp_no_absorb has no VG; Node4 remains its own PP stage, so the same wave appears as slower post-wave token latency.",
    ]
    (OUT / "vg_absorption_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_variant(sc, variant):
    if variant == "with_vg_absorb":
        ratios = VG_332
        extra = "--enable-plan-barrier --enable-kv-redundancy-during-migration 1 --kv-redundancy 2"
        topology = "3/3/2 VG"
    else:
        ratios = NO_VG_PP
        extra = ""
        topology = "singleton PP, no VG"
    suite.RATIOS = ratios
    suite.ACTIVE_WORKER_COUNT = 7
    suite.BASE_NET_MBPS = [HIGH_BW_MBPS] * 8
    suite.CPU_LIMITS = [2.0] * 8
    suite.ROOT_THREADS = 2
    suite.WORKER_THREADS = 2
    meta = suite.start_cluster(MODEL, f"{sc['scenario']}_{variant}", "vg_absorption_ablation", steps=STEPS, max_seq_len=MAX_SEQ_LEN, extra_args=extra, prompt=PROMPT)
    test_dir = Path(meta["test_dir"])
    root_exit = 999
    t_cmd = 0.0
    t_recover = 0.0
    apply_seen = False
    apply_line = ""
    t0 = time.time()
    try:
        if not suite.wait_for_position(meta["root"], EVENT_POS, timeout_s=360):
            raise RuntimeError(f"root did not reach pos {EVENT_POS}")
        wave_t = time.time()
        inject_wave(meta, sc, test_dir)
        if variant == "with_vg_absorb":
            _rc, t_cmd, cmd_out = suite.uds_set_plan(Path(meta["sock_host"]), 1, WEAK_NODE, 3, stage=1, head=1, ffn=0)
            (test_dir / "uds_cmd.json").write_text(cmd_out, encoding="utf-8")
            apply_seen, t_recover, apply_line = suite.poll_apply_any(meta["all_names"], 1, wave_t, timeout_s=120)
        wait = suite.run(["docker", "wait", meta["root"]], check=False, timeout=1200)
        root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
        wall_ms = (time.time() - t0) * 1000.0
        root_log = suite.read_logs(meta["root"])
        samples = pred_samples(root_log)
        pre_avg = avg_ms(samples, lambda pos: pos < EVENT_POS)
        post_avg = avg_ms(samples, lambda pos: pos >= EVENT_POS)
        eff = 1000.0 / post_avg if post_avg else 0.0
        _eval_ts, _pred_ts, _post_ts, _avg_pred, rejects = suite.parse_root_metrics(root_log)
        rows.append(Row(
            run_id=RUN_ID,
            scenario=sc["scenario"],
            variant=variant,
            topology=topology,
            ratios=ratios,
            root_exit=root_exit,
            steps=STEPS,
            event_pos=EVENT_POS,
            weak_node=WEAK_NODE,
            compute_skew=bool(sc["compute"]),
            bandwidth_skew=bool(sc["bandwidth"]),
            t_cmd_ms=t_cmd,
            t_recover_ms=t_recover,
            apply_seen=apply_seen,
            wall_ms=wall_ms,
            pre_wave_avg_ms=pre_avg,
            post_wave_avg_ms=post_avg,
            effective_post_tokens_s=eff,
            reject_count=rejects,
            apply_line=apply_line,
            output_dir=str(test_dir),
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
            print(f"RUN {sc['scenario']} with_vg_absorb", flush=True)
            run_variant(sc, "with_vg_absorb")
            print(f"RUN {sc['scenario']} no_vg_pp_no_absorb", flush=True)
            run_variant(sc, "no_vg_pp_no_absorb")
    finally:
        suite.cleanup_prefix("b01ab_")
        write_outputs()
    print(f"DONE OUT={OUT}", flush=True)

if __name__ == "__main__":
    main()

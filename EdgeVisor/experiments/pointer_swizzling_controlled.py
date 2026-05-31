#!/usr/bin/env python3
"""Controlled Pointer Swizzling ablation for the 8-container 3/3/2 setup.

This script keeps the full multi-stage topology active.  The controlled
straggler is Stage 1 Node 4.  Each run performs three head migrations:
4->3, 4->5, 4->3.  After the third migration, Node 4 should have
kvHeads=0/qHeads=0 for the Stage-1 attention path.

Unlike the broad ablation suite, this experiment splits binding cost from
runtime barrier waiting:
  - T_bind_ms: operator/weight rebinding work, or 0 for pointer swizzling.
  - T_cmd_ack_ms: UDS command round-trip.
  - T_cmd_to_apply_ms: time from UDS ack to plan apply at the fixed trigger.
  - T_total_controlled_ms: T_bind + T_cmd_ack + T_cmd_to_apply.
"""

import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import edgevisor_ablation_suite as suite


suite.RUN_ID = time.strftime("pc_%Y%m%d_%H%M%S")
suite.OUT = suite.BASE / "ptr_ctrl" / suite.RUN_ID
suite.SOCK_DIR = suite.OUT / "s"


@dataclass
class PointerControlledRow:
    model: str
    variant: str
    event_index: int
    stage: int
    trigger_pos: int
    trigger_layer: int
    from_node: int
    to_node: int
    bind_kind: str
    bind_bytes_or_ops: int
    t_bind_ms: float
    t_cmd_ack_ms: float
    t_cmd_to_apply_ms: float
    t_total_controlled_ms: float
    apply_seen: bool
    apply_line: str
    node4_work_line: str
    root_exit: int
    post_recovery_tokens_s: float
    avg_pred_ms_after_apply: float
    output_dir: str


rows = []


def operator_rebind_work_controlled(container, ops):
    """Materialize and rewrite a larger local execution-DAG descriptor set."""
    cmd = (
        "perl -MTime::HiRes=time -e '$n="
        + str(int(ops))
        + '; $t=time(); @d=(); '
        + 'for($i=0;$i<$n;$i++){ $d[$i]={op=>"attn_$i",in=>[$i,$i+1,$i+2,$i+3],out=>$i%257,wait=>$i%31,send=>$i%17,recv=>$i%19}; } '
        + 'for($r=0;$r<40;$r++){ for($i=0;$i<$n;$i++){ $d[$i]{wait}=($d[$i]{wait}+$r+$i)%37; $d[$i]{send}=($d[$i]{send}+$d[$i]{out})%29; }} '
        + 'printf qq(ms=%.3f ops=%d\\n),(time()-$t)*1000,$n;' + "'"
    )
    p = suite.docker_exec(container, cmd, check=False, timeout=600)
    m = re.search(r"ms=([0-9.]+)", p.stdout or "")
    return float(m.group(1)) if m else 0.0, p.returncode == 0, p.stdout or ""


def weight_materialize_work_controlled(container, nbytes, repeat=4):
    """Copy new contiguous weight views into fresh buffers, not just references."""
    mib = max(1, int((nbytes + 1024 * 1024 - 1) // (1024 * 1024)))
    cmd = (
        "perl -MTime::HiRes=time -e '$mb="
        + str(mib)
        + "; $rep="
        + str(int(repeat))
        + '; $t=time(); $total=0; '
        + 'for($r=0;$r<$rep;$r++){ $dst=""; for($i=0;$i<$mb;$i++){ $chunk=("A"x1048576); $dst .= $chunk; $total += length($chunk); } substr($dst,0,1)="B"; } '
        + 'printf qq(ms=%.3f bytes=%d\\n),(time()-$t)*1000,$total;' + "'"
    )
    p = suite.docker_exec(container, cmd, check=False, timeout=600)
    m = re.search(r"ms=([0-9.]+)", p.stdout or "")
    return float(m.group(1)) if m else 0.0, p.returncode == 0, p.stdout or ""


def write_csvs():
    path = suite.OUT / "pointer_controlled.csv"
    if rows:
        fields = list(asdict(rows[0]).keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    else:
        path.write_text("")
    summary = {
        "run_id": suite.RUN_ID,
        "output": str(suite.OUT),
        "rows": len(rows),
        "failed_rows": [asdict(r) for r in rows if not r.apply_seen or r.root_exit not in (-1, 0)],
        "design": {
            "ratios": suite.RATIOS,
            "vg_groups": suite.VG_GROUPS,
            "active_worker_count": suite.ACTIVE_WORKER_COUNT,
            "migration_sequence": "Stage1 Node4: 4->3, 4->5, 4->3",
            "trigger_positions": [16, 32, 48],
            "controlled_metrics": ["T_bind_ms", "T_cmd_ack_ms", "T_cmd_to_apply_ms", "T_total_controlled_ms"],
        },
    }
    (suite.OUT / "summary.json").write_text(json.dumps(summary, indent=2))


def find_epoch_work_line(containers, epoch, node=4, stage=1):
    pattern = f"node={node} stage={stage} epoch={epoch}"
    last = ""
    for container in containers:
        text = suite.read_logs(container)
        for line in text.splitlines():
            if "[plan][work]" in line and pattern in line:
                last = line
    return last


def run_model_variant(model_name, variant):
    exp = "pointer_swizzling_controlled"
    extra = "--enable-plan-barrier --enable-kv-redundancy-during-migration 1 --kv-redundancy 2"
    meta = suite.start_cluster(model_name, variant, exp, steps=64, max_seq_len=2048, extra_args=extra)
    test_dir = Path(meta["test_dir"])
    sock = Path(meta["sock_host"])
    model_info = suite.MODELS[model_name]
    trigger_layer = 11 if model_name == "3B" else 12
    moves = [(4, 3), (4, 5), (4, 3)]
    trigger_positions = [16, 32, 48]
    first_apply_pos = None
    root_exit = -1
    post_ts = 0.0
    avg_post = 0.0
    try:
        if not suite.wait_for_uds(sock):
            raise RuntimeError(f"UDS not ready: {sock}")
        for seq, ((frm, to), trigger_pos) in enumerate(zip(moves, trigger_positions), start=1):
            bind_kind = "pointer_swizzling"
            bind_units = model_info["layers"] * 32
            bind_log = ""
            t_bind = 0.0
            bind_start = time.time()
            if variant == "without_pointer_operator_rebind":
                bind_kind = "operator_rebind"
                bind_units = model_info["layers"] * 32 * 80
                t_bind, _, bind_log = operator_rebind_work_controlled(meta["all_names"][to], bind_units)
            elif variant == "without_pointer_weight_rebind":
                bind_kind = "weight_materialize"
                bind_units = suite.weight_materialize_bytes(model_info, moved_q_heads=3)
                t_bind, _, bind_log = weight_materialize_work_controlled(meta["all_names"][to], bind_units, repeat=4)
            bind_end = time.time()

            if not suite.wait_for_position(meta["root"], trigger_pos - 6, timeout_s=480 if model_name == "8B" else 360):
                raise RuntimeError(f"root did not reach pos {trigger_pos - 6} before pointer event {seq}")

            cmd_start = time.time()
            _, t_cmd, cmd_out = suite.uds_set_plan(
                sock,
                seq,
                frm,
                to,
                stage=1,
                head=1,
                ffn=0,
                mode="exact",
                trigger_pos=trigger_pos,
                trigger_layer=trigger_layer,
            )
            cmd_end = time.time()
            seen, t_apply_wait, apply_line = suite.poll_apply_any(meta["all_names"], seq, cmd_end, timeout_s=180)
            work_line = find_epoch_work_line(meta["all_names"], seq, node=4, stage=1)
            if first_apply_pos is None:
                mm = re.search(r"pos=([0-9]+)", apply_line or "")
                if mm:
                    first_apply_pos = int(mm.group(1))
            (test_dir / f"event_{seq}_bind.log").write_text(bind_log)
            (test_dir / f"event_{seq}_cmd.json").write_text(cmd_out)
            rows.append(
                PointerControlledRow(
                    model=model_name,
                    variant=variant,
                    event_index=seq,
                    stage=1,
                    trigger_pos=trigger_pos,
                    trigger_layer=trigger_layer,
                    from_node=frm,
                    to_node=to,
                    bind_kind=bind_kind,
                    bind_bytes_or_ops=int(bind_units),
                    t_bind_ms=float(t_bind if t_bind else (bind_end - bind_start) * 1000.0),
                    t_cmd_ack_ms=float(t_cmd),
                    t_cmd_to_apply_ms=float(t_apply_wait),
                    t_total_controlled_ms=float((t_bind if t_bind else 0.0) + t_cmd + t_apply_wait),
                    apply_seen=seen,
                    apply_line=apply_line,
                    node4_work_line=work_line,
                    root_exit=-1,
                    post_recovery_tokens_s=0.0,
                    avg_pred_ms_after_apply=0.0,
                    output_dir=str(test_dir),
                )
            )
            write_csvs()

        wait = suite.run(["docker", "wait", meta["root"]], check=False, timeout=900)
        root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
        root_log = suite.read_logs(meta["root"])
        _, _, post_ts, avg_post, _ = suite.parse_root_metrics(root_log, first_apply_pos)
        for row in rows:
            if row.model == model_name and row.variant == variant and row.output_dir == str(test_dir):
                row.root_exit = root_exit
                row.post_recovery_tokens_s = post_ts
                row.avg_pred_ms_after_apply = avg_post
    finally:
        suite.stop_cluster(meta)
        write_csvs()


def main():
    suite.ensure_dirs()
    suite.cleanup_prefix("b01ab_")
    print(f"OUT={suite.OUT}", flush=True)
    build = suite.run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"b01ab_build_{int(time.time())}",
            "-v",
            f"{suite.PROJ}:/workspace/EdgeVisor",
            "-w",
            "/workspace/EdgeVisor",
            suite.IMAGE,
            "bash",
            "-lc",
            "make -j$(nproc) dllama",
        ],
        check=False,
        timeout=900,
    )
    (suite.OUT / "build.log").write_text(build.stdout or "")
    if build.returncode != 0:
        print("BUILD_FAILED", file=sys.stderr)
        sys.exit(build.returncode)
    print("BUILD_OK", flush=True)

    smoke = os.environ.get("B01_POINTER_CONTROLLED_SMOKE") == "1"
    try:
        model_list = ["3B"] if smoke else ["3B", "8B"]
        variant_list = ["with_pointer_swizzling"] if smoke else [
            "with_pointer_swizzling",
            "without_pointer_operator_rebind",
            "without_pointer_weight_rebind",
        ]
        for model in model_list:
            for variant in variant_list:
                print(f"RUN pointer-controlled {model} {variant}", flush=True)
                run_model_variant(model, variant)
    finally:
        suite.cleanup_prefix("b01ab_")
        write_csvs()
    print(f"DONE OUT={suite.OUT}", flush=True)


if __name__ == "__main__":
    main()

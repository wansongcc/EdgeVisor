#!/usr/bin/env python3
"""Controlled JIT hierarchy ablation.

This is intentionally a controlled orchestration experiment, not an autonomous
scheduler benchmark.  It validates the cost hierarchy by selectively allowing:
  - full_jit: head-level intra-stage plan
  - no_sharding: PP layer-level plan after head plans are disabled
  - no_pipeline: no executable structural plan
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


suite.RUN_ID = time.strftime("jit_%Y%m%d_%H%M%S")
suite.OUT = suite.BASE / "jit_hierarchy_ablation_results" / suite.RUN_ID
suite.SOCK_DIR = suite.OUT / "s"


@dataclass
class JitRow:
    variant: str
    repeat: int
    plan_op: str
    expected_path: str
    shadow_kv_mode: str
    controlled_recovery_mode: str
    controlled_state_bytes: int
    controlled_state_ms: float
    controlled_state_ok: bool
    effective_recovery_ms: float
    disable_sharding_controller: bool
    disable_pipeline_balancer: bool
    command_sent: bool
    command_ok: bool
    head_probe_rejected: bool
    pp_probe_rejected: bool
    rejected: bool
    reject_reason: str
    apply_seen: bool
    recover_seen: bool
    first_apply_pos: int
    root_exit: int
    pred_tokens_s: float
    post_recovery_tokens_s: float
    avg_pred_ms_after_apply: float
    plan_emit_count: int
    plan_apply_count: int
    head_recover_count: int
    pp_recover_count: int
    rejected_event_count: int
    cumulative_stall_ms: float
    t_state_prepare_ms: float
    t_recover_ms: float
    state_transfer_bytes: int
    recompute_tokens_or_layers: int
    fallback_reasons: str
    output_dir: str


rows = []


def write_csvs():
    path = suite.OUT / "jit_hierarchy_controlled.csv"
    if not rows:
        path.write_text("")
        return
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    summary = {
        "run_id": suite.RUN_ID,
        "output": str(suite.OUT),
        "rows": len(rows),
        "variants": sorted({r.variant for r in rows}),
        "design": "controlled hierarchy ablation; no autonomous detection/escalation claim",
    }
    (suite.OUT / "summary.json").write_text(json.dumps(summary, indent=2))


def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def summarize_events(records):
    ids = [str(r.get("event_id", "")) for r in records]
    recover_records = [r for r in records if str(r.get("event_id", "")) in {"head_migration_recover", "pp_migration_recover"}]
    fallback_reasons = sorted({str(r.get("fallback_reason", "")) for r in recover_records if str(r.get("fallback_reason", ""))})
    return {
        "plan_emit_count": sum(1 for x in ids if x in {"plan_command_emit", "pp_migration_emit"}),
        "plan_apply_count": sum(1 for x in ids if x in {"plan_command_apply", "pp_migration_apply"}),
        "head_recover_count": sum(1 for x in ids if x == "head_migration_recover"),
        "pp_recover_count": sum(1 for x in ids if x == "pp_migration_recover"),
        "rejected_event_count": sum(1 for x in ids if x == "jit_plan_rejected"),
        "cumulative_stall_ms": sum(float(r.get("stall_time_ms", 0.0) or 0.0) for r in recover_records),
        "t_state_prepare_ms": sum(float(r.get("t_state_prepare_ms", 0.0) or 0.0) for r in recover_records),
        "t_recover_ms": sum(float(r.get("t_recover_ms", 0.0) or 0.0) for r in recover_records),
        "state_transfer_bytes": sum(int(r.get("state_transfer_bytes", 0) or 0) for r in recover_records),
        "recompute_tokens_or_layers": sum(int(r.get("recompute_tokens_or_layers", 0) or 0) for r in recover_records),
        "fallback_reasons": ";".join(fallback_reasons),
    }


def pp_layer_kv_state_bytes(model_info, context_len, layer_count=1):
    return int(
        context_len
        * model_info["head_dim"]
        * model_info["kv_heads"]
        * 2
        * 4
        * max(1, int(layer_count))
    )


def wait_for_ablation_event(path, event_ids, timeout_s=30):
    deadline = time.time() + timeout_s
    event_ids = set(event_ids)
    while time.time() < deadline:
        for rec in load_jsonl(path):
            if str(rec.get("event_id", "")) in event_ids:
                return rec
        time.sleep(0.25)
    return None


def parse_uds_stdout(text):
    text = text or ""
    stripped = text.strip()
    if stripped:
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = stripped.find("{")
        end = stripped.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(stripped[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    last = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                last = obj
        except json.JSONDecodeError:
            continue
    return last


def send_pp_plan(sock, seq, from_node, to_node, stage, layer_count, mode="next_barrier", trigger_pos=None, trigger_layer=None):
    args = [
        "python3",
        str(suite.UDS_CLIENT),
        str(sock),
        "set_pp_migration",
        "--seq",
        str(seq),
        "--mode",
        mode,
        "--stage",
        str(stage),
        "--from",
        str(from_node),
        "--to",
        str(to_node),
        "--layer-count",
        str(layer_count),
    ]
    if trigger_pos is not None:
        args += ["--trigger-pos", str(trigger_pos)]
    if trigger_layer is not None:
        args += ["--trigger-layer", str(trigger_layer)]
    t0 = time.time()
    p = suite.run(args, check=False)
    return p.returncode, (time.time() - t0) * 1000.0, p.stdout


def variant_config(variant):
    if variant == "full_jit":
        return {
            "plan_op": "set_plan",
            "expected_path": "head_sharding_controller",
            "shadow_kv_mode": "enabled",
            "controlled_recovery_mode": "none",
            "disable_sharding": False,
            "disable_pipeline": False,
            "extra": (
                "--ablation-log-path /out/ablation.jsonl --experiment-id jit_full "
                "--enable-plan-barrier --kv-redundancy 2 "
                "--enable-kv-redundancy-during-migration 1 "
                "--enable-stage-full-weights --enable-pp-migration "
                "--runtime-redundant-boundary-layers 1 "
                "--runtime-active-seg-enabled 1 --runtime-redundant-seg-enabled 1"
            ),
        }
    if variant == "no_sharding":
        return {
            "plan_op": "set_pp_migration",
            "expected_path": "pipeline_balancer_fallback",
            "shadow_kv_mode": "enabled",
            "controlled_recovery_mode": "none",
            "disable_sharding": True,
            "disable_pipeline": False,
            "extra": (
                "--ablation-log-path /out/ablation.jsonl --experiment-id jit_no_sharding "
                "--enable-plan-barrier --kv-redundancy 2 "
                "--enable-kv-redundancy-during-migration 1 "
                "--enable-stage-full-weights --enable-pp-migration "
                "--runtime-redundant-boundary-layers 1 "
                "--runtime-active-seg-enabled 1 --runtime-redundant-seg-enabled 1 "
                "--disable-sharding-controller 1"
            ),
        }
    if variant == "no_sharding_transfer":
        return {
            "plan_op": "set_pp_migration",
            "expected_path": "pipeline_balancer_transfer_recovery",
            "shadow_kv_mode": "controlled_transfer",
            "controlled_recovery_mode": "transfer",
            "disable_sharding": True,
            "disable_pipeline": False,
            "extra": (
                "--ablation-log-path /out/ablation.jsonl --experiment-id jit_no_sharding_transfer "
                "--enable-plan-barrier --kv-redundancy 0 "
                "--enable-kv-redundancy-during-migration 0 "
                "--enable-stage-full-weights --enable-pp-migration "
                "--runtime-redundant-boundary-layers 1 "
                "--runtime-active-seg-enabled 1 --runtime-redundant-seg-enabled 0 "
                "--disable-sharding-controller 1"
            ),
        }
    if variant == "no_sharding_recompute":
        return {
            "plan_op": "set_pp_migration",
            "expected_path": "pipeline_balancer_recompute_recovery",
            "shadow_kv_mode": "controlled_recompute",
            "controlled_recovery_mode": "recompute",
            "disable_sharding": True,
            "disable_pipeline": False,
            "extra": (
                "--ablation-log-path /out/ablation.jsonl --experiment-id jit_no_sharding_recompute "
                "--enable-plan-barrier --kv-redundancy 0 "
                "--enable-kv-redundancy-during-migration 0 "
                "--enable-stage-full-weights --enable-pp-migration "
                "--runtime-redundant-boundary-layers 1 "
                "--runtime-active-seg-enabled 1 --runtime-redundant-seg-enabled 0 "
                "--disable-sharding-controller 1"
            ),
        }
    if variant == "no_pipeline":
        return {
            "plan_op": "none",
            "expected_path": "no_structural_recovery",
            "shadow_kv_mode": "enabled",
            "controlled_recovery_mode": "none",
            "disable_sharding": True,
            "disable_pipeline": True,
            "extra": (
                "--ablation-log-path /out/ablation.jsonl --experiment-id jit_no_pipeline "
                "--enable-plan-barrier --kv-redundancy 2 "
                "--enable-kv-redundancy-during-migration 1 "
                "--enable-stage-full-weights --enable-pp-migration "
                "--runtime-redundant-boundary-layers 1 "
                "--runtime-active-seg-enabled 1 --runtime-redundant-seg-enabled 1 "
                "--disable-sharding-controller 1 --disable-pipeline-balancer 1"
            ),
        }
    raise ValueError(f"unknown variant: {variant}")


def run_variant(variant, repeat, steps=96):
    cfg = variant_config(variant)
    exp = "jit_hierarchy_controlled"
    meta = suite.start_cluster(
        "3B",
        f"{variant}_r{repeat}",
        exp,
        steps=steps,
        max_seq_len=2048,
        extra_args=cfg["extra"],
    )
    test_dir = Path(meta["test_dir"])
    sock = Path(meta["sock_host"])
    first_apply_pos = -1
    command_ok = False
    head_probe_rejected = False
    pp_probe_rejected = False
    rejected = False
    reject_reason = ""
    apply_seen = False
    recover_seen = False
    controlled_state_bytes = 0
    controlled_state_ms = 0.0
    controlled_state_ok = True
    try:
        if not suite.wait_for_uds(sock, timeout_s=180):
            raise RuntimeError(f"UDS not ready: {sock}")
        if not suite.wait_for_position(meta["root"], 8, timeout_s=360):
            raise RuntimeError("root did not reach pos 8")

        command_sent = cfg["plan_op"] != "none" or variant == "no_pipeline"
        if cfg["disable_sharding"]:
            _rc, _t_cmd, out = suite.uds_set_plan(sock, 100, 1, 0, stage=0, head=1, ffn=0, mode="next_barrier")
            (test_dir / "jit_head_probe.json").write_text(out)
            resp = parse_uds_stdout(out)
            head_probe_rejected = bool(resp.get("rejected", False))
            if resp.get("reason"):
                reject_reason = str(resp.get("reason", ""))

        if variant == "no_pipeline":
            _rc, _t_cmd, out = send_pp_plan(sock, 101, 2, 3, stage=0, layer_count=1, mode="next_barrier")
            (test_dir / "jit_pp_probe.json").write_text(out)
            resp = parse_uds_stdout(out)
            pp_probe_rejected = bool(resp.get("rejected", False))
            rejected = head_probe_rejected and pp_probe_rejected
            if resp.get("reason"):
                reject_reason = str(resp.get("reason", reject_reason))
        elif cfg["plan_op"] == "set_plan":
            t0 = time.time()
            _rc, _t_cmd, out = suite.uds_set_plan(sock, 1, 1, 0, stage=0, head=1, ffn=0, mode="next_barrier")
            (test_dir / "jit_cmd.json").write_text(out)
            resp = parse_uds_stdout(out)
            command_ok = bool(resp.get("ok", False))
            rejected = bool(resp.get("rejected", False))
            reject_reason = str(resp.get("reason", ""))
            apply_seen, _t_apply, line = suite.poll_apply_any(meta["all_names"], 1, t0, timeout_s=90)
            if apply_seen:
                mm = re.search(r"pos=([0-9]+)", line or "")
                first_apply_pos = int(mm.group(1)) if mm else -1
            recover_seen = apply_seen
        elif cfg["plan_op"] == "set_pp_migration":
            if cfg["controlled_recovery_mode"] != "none":
                context_len = int(os.environ.get("B01_JIT_STATE_CONTEXT", "4096"))
                layer_count = 1
                controlled_state_bytes = pp_layer_kv_state_bytes(suite.MODELS["3B"], context_len, layer_count=layer_count)
                if cfg["controlled_recovery_mode"] == "transfer":
                    controlled_state_ms, controlled_state_ok, state_log = suite.transfer_bytes(
                        meta["all_names"][2],
                        meta["all_names"][3],
                        controlled_state_bytes,
                        port=26000 + repeat * 10,
                    )
                elif cfg["controlled_recovery_mode"] == "recompute":
                    controlled_state_ms, controlled_state_ok, state_log = suite.recompute_bytes(
                        meta["all_names"][3],
                        controlled_state_bytes,
                    )
                else:
                    state_log = ""
                    controlled_state_ok = False
                (test_dir / "controlled_state.log").write_text(
                    json.dumps(
                        {
                            "mode": cfg["controlled_recovery_mode"],
                            "context_len": context_len,
                            "layer_count": layer_count,
                            "state_bytes": controlled_state_bytes,
                            "state_ms": controlled_state_ms,
                            "ok": controlled_state_ok,
                        },
                        indent=2,
                    )
                    + "\n--- raw ---\n"
                    + (state_log or "")
                )
            _rc, _t_cmd, out = send_pp_plan(sock, 1, 2, 3, stage=0, layer_count=1, mode="next_barrier")
            (test_dir / "jit_cmd.json").write_text(out)
            resp = parse_uds_stdout(out)
            command_ok = bool(resp.get("ok", False))
            rejected = head_probe_rejected or bool(resp.get("rejected", False))
            reject_reason = str(resp.get("reason", ""))
            rec = wait_for_ablation_event(test_dir / "ablation.jsonl", {"pp_migration_apply", "pp_migration_recover"}, timeout_s=45)
            if rec:
                apply_seen = True
                first_apply_pos = int(rec.get("trigger_pos", -1) or -1)
            recover_seen = "pp_migration_recover" in "\n".join(suite.read_logs(name) for name in meta["all_names"])
        elif cfg["plan_op"] == "none":
            command_sent = False

        wait = suite.run(["docker", "wait", meta["root"]], check=False, timeout=900)
        root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
        root_log = suite.read_logs(meta["root"])
        _eval_ts, pred_ts, post_ts, avg_post, _rejects = suite.parse_root_metrics(root_log, first_apply_pos if first_apply_pos >= 0 else None)
        records = load_jsonl(test_dir / "ablation.jsonl")
        ev = summarize_events(records)
        if not apply_seen and ev["plan_apply_count"] > 0:
            apply_seen = True
        if first_apply_pos < 0:
            for rec in records:
                if str(rec.get("event_id", "")) in {"plan_command_apply", "pp_migration_apply"}:
                    first_apply_pos = int(rec.get("trigger_pos", -1) or -1)
                    break
        effective_recovery_ms = ev["cumulative_stall_ms"] + ev["t_state_prepare_ms"] + controlled_state_ms
        rows.append(
            JitRow(
                variant=variant,
                repeat=repeat,
                plan_op=cfg["plan_op"],
                expected_path=cfg["expected_path"],
                shadow_kv_mode=cfg["shadow_kv_mode"],
                controlled_recovery_mode=cfg["controlled_recovery_mode"],
                controlled_state_bytes=controlled_state_bytes,
                controlled_state_ms=controlled_state_ms,
                controlled_state_ok=controlled_state_ok,
                effective_recovery_ms=effective_recovery_ms,
                disable_sharding_controller=cfg["disable_sharding"],
                disable_pipeline_balancer=cfg["disable_pipeline"],
                command_sent=command_sent,
                command_ok=command_ok,
                head_probe_rejected=head_probe_rejected,
                pp_probe_rejected=pp_probe_rejected,
                rejected=rejected,
                reject_reason=reject_reason,
                apply_seen=apply_seen,
                recover_seen=recover_seen or ev["head_recover_count"] > 0 or ev["pp_recover_count"] > 0,
                first_apply_pos=first_apply_pos,
                root_exit=root_exit,
                pred_tokens_s=pred_ts,
                post_recovery_tokens_s=post_ts,
                avg_pred_ms_after_apply=avg_post,
                plan_emit_count=ev["plan_emit_count"],
                plan_apply_count=ev["plan_apply_count"],
                head_recover_count=ev["head_recover_count"],
                pp_recover_count=ev["pp_recover_count"],
                rejected_event_count=ev["rejected_event_count"],
                cumulative_stall_ms=ev["cumulative_stall_ms"],
                t_state_prepare_ms=ev["t_state_prepare_ms"],
                t_recover_ms=ev["t_recover_ms"],
                state_transfer_bytes=ev["state_transfer_bytes"],
                recompute_tokens_or_layers=ev["recompute_tokens_or_layers"],
                fallback_reasons=ev["fallback_reasons"],
                output_dir=str(test_dir),
            )
        )
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
            suite.CONTAINER_ENGINE_DIR,
            suite.IMAGE,
            "bash",
            "-lc",
            "make clean >/dev/null 2>&1 || true; make -j$(nproc) dllama",
        ],
        check=False,
        timeout=900,
    )
    (suite.OUT / "build.log").write_text(build.stdout or "")
    if build.returncode != 0:
        print("BUILD_FAILED", file=sys.stderr)
        sys.exit(build.returncode)
    print("BUILD_OK", flush=True)
    try:
        repeats = int(os.environ.get("B01_JIT_REPEAT", "2"))
        steps = int(os.environ.get("B01_JIT_STEPS", "96"))
        variants_env = os.environ.get("B01_JIT_VARIANTS", "")
        variants = [v.strip() for v in variants_env.split(",") if v.strip()]
        if not variants:
            variants = ["full_jit", "no_sharding", "no_sharding_transfer", "no_sharding_recompute", "no_pipeline"]
        if os.environ.get("B01_JIT_SMOKE") == "1":
            repeats = 1
            variants = ["full_jit"]
            steps = int(os.environ.get("B01_JIT_STEPS", "48"))
        for repeat in range(1, repeats + 1):
            for variant in variants:
                print(f"RUN jit {variant} repeat={repeat}", flush=True)
                run_variant(variant, repeat, steps=steps)
                write_csvs()
    finally:
        suite.cleanup_prefix("b01ab_")
        write_csvs()
    print(f"DONE OUT={suite.OUT}", flush=True)


if __name__ == "__main__":
    main()

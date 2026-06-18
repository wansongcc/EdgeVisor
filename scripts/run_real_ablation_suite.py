#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("B01_ROOT", "/home/byh/B01"))
DEFAULT_PYTHON = DEFAULT_ROOT / "agent_langgraph_venv/bin/python"
DEFAULT_EPISODE = PROJECT_ROOT / "agent_bench/episodes/agentic_ablation_episode.json"
EDGE_ENGINE = PROJECT_ROOT / "EdgeVisor"
DLLAMA = EDGE_ENGINE / "dllama"
MODEL = DEFAULT_ROOT / "models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOKENIZER = DEFAULT_ROOT / "models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"

VARIANTS: Dict[str, Dict[str, str]] = {
    "full": {"shadow": "enabled", "pointer": "enabled", "jit": "enabled", "vg": "enabled"},
    "shadow_transfer": {"shadow": "disabled_transfer", "pointer": "enabled", "jit": "enabled", "vg": "enabled"},
    "shadow_recompute": {"shadow": "disabled_recompute", "pointer": "enabled", "jit": "enabled", "vg": "enabled"},
    "pointer_rebuild": {"shadow": "enabled", "pointer": "operator_rebuild", "jit": "enabled", "vg": "enabled"},
    "pointer_rematerialize": {"shadow": "enabled", "pointer": "weight_rematerialize", "jit": "enabled", "vg": "enabled"},
    "jit_static": {"shadow": "enabled", "pointer": "enabled", "jit": "static", "vg": "enabled"},
    "jit_greedy": {"shadow": "enabled", "pointer": "enabled", "jit": "greedy", "vg": "enabled"},
    "vg_flat": {"shadow": "enabled", "pointer": "enabled", "jit": "enabled", "vg": "flat"},
    "vg_pure_pp": {"shadow": "enabled", "pointer": "enabled", "jit": "enabled", "vg": "pure_pp"},
}

SHADOW_VARIANTS = ["full", "shadow_transfer", "shadow_recompute"]
SMOKE_VARIANTS = list(SHADOW_VARIANTS)
FULL_FLUCTUATIONS = ["compute", "intra_stage_bw", "inter_stage_bw", "mixed_bw"]
SHADOW_SCOPES = ["intra_stage_heads", "inter_stage_layers"]


class PerturbationError(RuntimeError):
    pass


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> List[int]:
    out: List[int] = []
    for item in parse_csv(value):
        out.append(int(item))
    return out


def run_checked(cmd: List[str], *, input_text: Optional[str] = None, timeout_s: int = 30) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        rendered = " ".join(shlex.quote(x) for x in cmd)
        raise PerturbationError(f"command failed rc={proc.returncode}: {rendered}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


def network_proxy_rate(fluctuation: str, args: argparse.Namespace) -> Optional[float]:
    if fluctuation == "mixed_bw":
        return scope_network_rate_mib(args)
    rates = {
        "intra_stage_bw": float(args.network_intra_mib_s),
        "inter_stage_bw": float(args.network_inter_mib_s),
    }
    return rates.get(fluctuation)


def scope_network_rate_mib(args: argparse.Namespace) -> float:
    if args.shadow_scope == "intra_stage_heads":
        return float(args.network_intra_mib_s)
    return float(args.network_inter_mib_s)


def scope_network_nodes(args: argparse.Namespace) -> List[str]:
    if args.network_proxy_nodes:
        return parse_csv(args.network_proxy_nodes)
    if args.shadow_scope == "intra_stage_heads":
        return ["1", "4", "7"]
    return ["3", "4", "5", "6", "7"]


def scope_compute_target_gpu(args: argparse.Namespace) -> int:
    if args.compute_target_gpu >= 0:
        return int(args.compute_target_gpu)
    return 0 if args.shadow_scope == "intra_stage_heads" else 2


def scope_kv_redundancy(args: argparse.Namespace, variant_name: str) -> str:
    if variant_name != "full" or args.shadow_scope != "intra_stage_heads":
        return "0"

    pad = max(0, int(args.intra_shadow_kv_pad))
    values = [0] * 8
    stage_nodes = {
        0: (0, 1, 2),
        1: (3, 4, 5),
        2: (6, 7),
    }
    for node in stage_nodes.get(int(args.intra_stage_index), stage_nodes[0]):
        values[node] = pad
    return ",".join(str(v) for v in values)


def stage_move_pair(stage_index: int, head_move: int) -> List[Dict[str, int]]:
    moves = {
        0: (0, 1),
        1: (3, 4),
        2: (6, 7),
    }
    from_node, to_node = moves.get(int(stage_index), moves[0])
    return [
        {"fromNodeIndex": from_node, "toNodeIndex": to_node, "cmdKind": 1, "headMove": head_move, "ffnMove": 0},
        {"fromNodeIndex": to_node, "toNodeIndex": from_node, "cmdKind": 1, "headMove": head_move, "ffnMove": 0},
    ]


def intra_stage_single_move(stage_index: int, head_move: int, *, reverse: bool = False) -> Dict[str, int]:
    pair = stage_move_pair(stage_index, head_move)
    return pair[1] if reverse else pair[0]


def intra_stage_multi_move(stage_index: int, head_move: int, idx: int, pattern: str) -> Dict[str, int]:
    if pattern == "same_edge":
        return intra_stage_single_move(stage_index, head_move, reverse=False)
    if pattern == "right_chain":
        chains = {
            0: [(0, 1), (1, 2)],
            1: [(3, 4), (4, 5)],
            2: [(6, 7)],
        }
        edges = chains.get(int(stage_index), chains[0])
        from_node, to_node = edges[idx % len(edges)]
        return {"fromNodeIndex": from_node, "toNodeIndex": to_node, "cmdKind": 1, "headMove": head_move, "ffnMove": 0}
    return intra_stage_single_move(stage_index, head_move, reverse=(idx % 2 == 1))


def intra_stage_nodes(stage_index: int) -> List[int]:
    stage_nodes = {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7],
    }
    return list(stage_nodes.get(int(stage_index), stage_nodes[0]))


def scope_recovery_event(args: argparse.Namespace, variant_name: str) -> str:
    if variant_name == "full":
        return "shadow_precomputed"
    if args.shadow_scope == "intra_stage_heads":
        if variant_name == "shadow_transfer":
            return "head_migration_recover:shadow_kv_disabled_head_real_transfer"
        if variant_name == "shadow_recompute":
            return "head_migration_recover:shadow_kv_disabled_head_real_recompute_replay"
    if variant_name == "shadow_transfer":
        return "pp_migration_recover:shadow_kv_disabled_real_transfer"
    if variant_name == "shadow_recompute":
        return "pp_migration_recover:shadow_kv_disabled_real_recompute_replay"
    return ""


def stage_bounds(stage_index: int, *, total_layers: int = 28, stage_weights: Iterable[int] = (3, 3, 2)) -> tuple[int, int]:
    weights = [max(0, int(x)) for x in stage_weights]
    if not weights or sum(weights) <= 0:
        raise ValueError("stage weights must be positive")
    idx = int(stage_index)
    if idx < 0 or idx >= len(weights):
        idx = 0
    total_weight = float(sum(weights))
    allocated = 0
    start = 0
    for i, weight in enumerate(weights):
        if i + 1 == len(weights):
            n_layers = total_layers - allocated
        else:
            n_layers = int(round(total_layers * (float(weight) / total_weight)))
            if allocated + n_layers > total_layers:
                n_layers = total_layers - allocated
        end = start + n_layers
        if i == idx:
            return start, end
        allocated += n_layers
        start = end
    return 0, total_layers


def stage_last_layer(stage_index: int) -> int:
    start, end = stage_bounds(stage_index)
    if end <= start:
        return start
    return end - 1


def make_prefill_text(args: argparse.Namespace) -> str:
    custom = str(getattr(args, "prefill_text", "") or "").strip()
    if custom:
        return custom
    tokens = max(0, int(getattr(args, "prefill_tokens", 0) or 0))
    if tokens <= 0:
        return ""
    unit = "kv-cache calibration token sequence for late migration measurement. "
    return ("Prefill context: " + (unit * tokens)).strip()


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


def port_block_is_free(port_base: int, worker_count: int = 7) -> bool:
    ports = [port_base + i + 1 for i in range(worker_count)]
    ports.extend(port_base + 100 + i + 1 for i in range(worker_count))
    return all(port_is_free(port) for port in ports)


def choose_port_base(preferred_base: int, *, worker_count: int = 7, attempts: int = 100) -> int:
    for attempt in range(attempts):
        candidate = preferred_base + attempt * 200
        if port_block_is_free(candidate, worker_count=worker_count):
            return candidate
    raise PerturbationError(f"could not find a free EdgeVisor port block starting at {preferred_base}")


def start_compute_stress(
    run_root: Path,
    target_gpu: int,
    cuda_visible: str,
    steps: int,
    model: Path,
    tokenizer: Path,
) -> subprocess.Popen[str]:
    if not DLLAMA.exists():
        raise PerturbationError(f"missing dllama binary: {DLLAMA}")
    for path in [model, tokenizer]:
        if not path.exists():
            raise PerturbationError(f"missing model/tokenizer file: {path}")
    log_path = run_root / f"compute_stress_gpu{target_gpu}.log"
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    inner = [
        str(DLLAMA),
        "inference",
        "--prompt",
        "background compute stress for EdgeVisor ablation",
        "--steps",
        str(steps),
        "--model",
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--buffer-float-type",
        "q80",
        "--nthreads",
        "1",
        "--max-seq-len",
        "2048",
        "--temperature",
        "0",
        "--seed",
        "7",
        "--gpu-index",
        str(target_gpu),
    ]
    loop = "while true; do " + " ".join(shlex.quote(part) for part in inner) + "; done"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    proc = subprocess.Popen(
        ["bash", "-lc", loop],
        cwd=str(EDGE_ENGINE),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._edgevisor_log_f = log_f  # type: ignore[attr-defined]
    return proc


def terminate_proc(proc: Optional[subprocess.Popen[Any]]) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
    log_f = getattr(proc, "_edgevisor_log_f", None)
    if log_f:
        log_f.close()


def write_episode_copy(
    src: Path,
    dst: Path,
    fluctuation: str,
    variant_name: str,
    repeat: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    episode = json.loads(src.read_text(encoding="utf-8"))
    plan = dict(episode.get("edgevisor_dynamic_plan", {}))
    migration_layer_count = max(1, int(args.migration_layer_count))
    trigger_generation = str(getattr(args, "trigger_generation", "agent_step_02"))
    plan_mode = str(getattr(args, "plan_mode", "next_barrier"))
    trigger_position = int(getattr(args, "trigger_position", -1) or -1)
    base_plan = {
        "enabled": True,
        "trigger_generation": trigger_generation,
        "mode": plan_mode,
        "delay_s": 0.0,
        "consume_timeout_s": float(getattr(args, "consume_timeout_s", 30.0)),
        "fluctuation_type": fluctuation,
        "candidate_count": 2,
        "fallback_count": 0,
        "experiment_variant": variant_name,
        "shadow_scope": args.shadow_scope,
        "repeat": repeat,
    }
    prefill_text = make_prefill_text(args)
    if prefill_text:
        episode["prefill_context"] = {
            "enabled": True,
            "trigger_generation": trigger_generation,
            "text": prefill_text,
            "requested_tokens": int(getattr(args, "prefill_tokens", 0) or 0),
        }
    if args.shadow_scope == "intra_stage_heads":
        head_move = max(1, int(args.intra_head_move))
        stage_index = int(args.intra_stage_index)
        trigger_layer = stage_last_layer(stage_index)
        base_plan.update(
            {
                "plan_op": "set_plan",
                "mode": plan_mode,
                "intended_trigger_layer": trigger_layer,
                "stage": stage_index,
                "moves": [intra_stage_single_move(stage_index, head_move)],
                "logical_group": "intra_stage_head_migration_single_stage",
                "migration_scope": "single_stage",
                "selected_stage_nodes": intra_stage_nodes(stage_index),
                "kv_redundancy_scope": "selected_stage_only",
                "triggerLayer": trigger_layer,
                "trigger_layer": trigger_layer,
            }
        )
        if plan_mode == "exact":
            base_plan["triggerPos"] = max(0, trigger_position)
            base_plan["trigger_pos"] = max(0, trigger_position)
        else:
            base_plan["trigger_pos_strategy"] = "status_offset"
            base_plan["trigger_pos_offset"] = max(1, int(args.intra_trigger_pos_offset))
    else:
        trigger_layer = stage_last_layer(1)
        base_plan.update(
            {
                "plan_op": "set_pp_migration",
                "mode": plan_mode,
                "stage": 1,
                "from_node": 5,
                "to_node": 6,
                "layerCount": migration_layer_count,
                "triggerLayer": trigger_layer,
                "trigger_layer": trigger_layer,
                "logical_group": "inter_stage_layer_migration",
            }
        )
        if plan_mode == "exact":
            base_plan["triggerPos"] = max(0, trigger_position)
            base_plan["trigger_pos"] = max(0, trigger_position)
    plan.update(base_plan)
    plan.pop("simulated_stall_ms", None)
    plan.pop("simulated_recovery_latency_ms", None)
    if args.shadow_scope != "intra_stage_heads":
        plan.pop("moves", None)
    if variant_name == "jit_static" and args.shadow_scope == "inter_stage_layers":
        plan.update({"from_node": 3, "to_node": 6, "candidate_count": 1, "fallback_count": 1})
    elif variant_name == "jit_greedy" and args.shadow_scope == "inter_stage_layers":
        plan.update({"from_node": 5, "to_node": 7, "candidate_count": 2, "fallback_count": 1})
    episode["edgevisor_dynamic_plan"] = plan
    dst.write_text(json.dumps(episode, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def write_multi_trigger_episode_copy(
    src: Path,
    dst: Path,
    fluctuation: str,
    variant_name: str,
    repeat: int,
    args: argparse.Namespace,
    trigger_positions: List[int],
    inter_command_delay_s: float = 0.0,
) -> Dict[str, Any]:
    episode = json.loads(src.read_text(encoding="utf-8"))
    single = write_episode_copy(src, dst, fluctuation, variant_name, repeat, args)
    episode = json.loads(dst.read_text(encoding="utf-8"))
    commands = []
    for idx, trigger_pos in enumerate(trigger_positions):
        cmd = dict(single)
        cmd["seq"] = 100000 + repeat * 100 + idx
        cmd["trigger_generation"] = str(getattr(args, "trigger_generation", "agent_step_02"))
        cmd["triggerPos"] = max(0, int(trigger_pos))
        cmd["trigger_pos"] = max(0, int(trigger_pos))
        cmd["trigger_pos_strategy"] = "exact"
        cmd["delay_s"] = 0.0 if idx == 0 else inter_command_delay_s
        if args.shadow_scope == "intra_stage_heads":
            cmd["moves"] = [
                intra_stage_multi_move(
                    int(args.intra_stage_index),
                    max(1, int(args.intra_head_move)),
                    idx,
                    str(getattr(args, "multi_migration_pattern", "roundtrip")),
                )
            ]
            cmd["migration_scope"] = "single_stage_roundtrip_step"
            cmd["logical_group"] = "intra_stage_head_migration_multi_roundtrip"
        elif args.shadow_scope == "inter_stage_layers":
            cmd["plan_op"] = "set_pp_migration"
            if idx % 2 == 1:
                cmd["from_node"], cmd["to_node"] = int(cmd.get("to_node", 6)), int(cmd.get("from_node", 5))
        commands.append(cmd)
    episode["edgevisor_dynamic_plan"] = {
        "enabled": True,
        "trigger_generation": str(getattr(args, "trigger_generation", "agent_step_02")),
        "mode": str(getattr(args, "plan_mode", "exact")),
        "delay_s": 0.0,
        "consume_timeout_s": float(getattr(args, "consume_timeout_s", 30.0)),
        "fluctuation_type": fluctuation,
        "candidate_count": len(commands),
        "fallback_count": 0,
        "experiment_variant": variant_name,
        "shadow_scope": args.shadow_scope,
        "repeat": repeat,
        "seq_base": 100000 + repeat * 100,
        "expected_command_count": len(commands),
        "commands": commands,
    }
    dst.write_text(json.dumps(episode, indent=2, ensure_ascii=False), encoding="utf-8")
    return episode["edgevisor_dynamic_plan"]


def build_episode_command(
    args: argparse.Namespace,
    variant_name: str,
    modes: Dict[str, str],
    fluctuation: str,
    repeat: int,
    run_root: Path,
    episode_copy: Path,
    port_base: int,
) -> List[str]:
    kv_redundancy = scope_kv_redundancy(args, variant_name)
    cmd = [
        str(args.python_bin),
        "-m",
        "agent_bench.run_loop_episode",
        "--backend",
        "edgevisor_ablation",
        "--episode",
        str(episode_copy),
        "--out-root",
        str(run_root),
        "--cuda-visible",
        args.cuda_visible,
        "--edge-steps",
        str(args.edge_steps),
        "--ctx",
        str(args.ctx),
        "--shadow-kv-mode",
        modes["shadow"],
        "--pointer-swizzling-mode",
        modes["pointer"],
        "--jit-mode",
        modes["jit"],
        "--vg-mode",
        modes["vg"],
        "--fallback-policy",
        "disabled_unless_necessary",
        "--experiment-id",
        f"{variant_name}_{fluctuation}_rep{repeat}",
        "--edge-virtual-pp-tp-3stage",
        "--edge-virtual-ratios",
        "1:1:1*1:1:1*1:1",
        "--edge-virtual-node-gpus",
        "0,1,2,0,1,2,0,1",
        "--edge-virtual-launch-stagger-s",
        str(args.edge_virtual_launch_stagger_s),
        "--edge-fixed-port-base",
        str(port_base),
        "--runtime-redundant-boundary-layers",
        str(args.runtime_redundant_boundary_layers),
    ]
    if getattr(args, "bubble_shadow_kv", False):
        cmd.append("--bubble-shadow-kv")
    if getattr(args, "edge_benchmark", False):
        cmd.append("--edge-benchmark")
    if args.shadow_scope == "inter_stage_layers":
        cmd.append("--enable-pp-migration")
    if args.shadow_scope == "intra_stage_heads":
        cmd.append("--allow-head-kv-migration")

    config: Dict[str, Any] = {
        "shadow_scope": args.shadow_scope,
        "kv_redundancy": kv_redundancy,
        "kv_redundancy_design": (
            "full+intra_stage_heads keeps KV pad only inside the selected migrating stage; "
            "inter_stage_layers and no-shadow variants use zero head KV redundancy"
        ),
        "expected_recovery_event": scope_recovery_event(args, variant_name),
        "plan_mode": str(getattr(args, "plan_mode", "next_barrier")),
        "trigger_position": int(getattr(args, "trigger_position", -1) or -1),
        "multi_trigger_positions": parse_int_csv(str(getattr(args, "multi_trigger_positions", "") or "")),
        "multi_migration_pattern": str(getattr(args, "multi_migration_pattern", "roundtrip")),
        "expected_command_count": len(parse_int_csv(str(getattr(args, "multi_trigger_positions", "") or ""))) or 1,
        "prefill_tokens": int(getattr(args, "prefill_tokens", 0) or 0),
    }
    rate = network_proxy_rate(fluctuation, args)
    if rate is not None:
        nodes = scope_network_nodes(args)
        config.update(
            {
                "network_proxy_rate_mbps": rate,
                "network_proxy_mode": "user_space_tcp_rate_proxy",
                "network_proxy_scope": args.network_proxy_scope,
                "network_proxy_nodes": nodes,
                "network_proxy_from_node": int(nodes[0]) if nodes else 5,
                "network_proxy_to_node": int(nodes[-1]) if nodes else 6,
                "network_proxy_start_throttled": False,
            }
        )
    config_path = run_root / "ablation_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd.extend(["--edgevisor-ablation-config", str(config_path)])
    return cmd


def append_manifest(path: Path, item: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def run_episode_process(cmd: List[str], run_root: Path, timeout_s: int, cuda_visible: str) -> int:
    stdout_path = run_root / "run_stdout.json"
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible},
            stdout=stdout_f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
            return 124


def selected_variants(args: argparse.Namespace) -> List[str]:
    if args.variants:
        names = parse_csv(args.variants)
    elif args.smoke:
        names = list(SMOKE_VARIANTS)
    elif getattr(args, "shadow_scope", ""):
        names = list(SHADOW_VARIANTS)
    else:
        names = list(VARIANTS)
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {','.join(unknown)}")
    return names


def selected_fluctuations(args: argparse.Namespace) -> List[str]:
    if args.fluctuations:
        names = parse_csv(args.fluctuations)
    elif args.smoke:
        names = ["mixed_bw"]
    elif getattr(args, "shadow_scope", ""):
        names = ["mixed_bw"]
    else:
        names = ["mixed_bw"]
    unknown = [name for name in names if name not in FULL_FLUCTUATIONS]
    if unknown:
        raise SystemExit(f"unknown fluctuations: {','.join(unknown)}")
    return names


def validate_no_simulated_fields(root: Path) -> None:
    offenders: List[str] = []
    for path in root.rglob("episode.json"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "simulated_stall_ms" in text or "simulated_recovery_latency_ms" in text:
            offenders.append(str(path))
    if offenders:
        raise SystemExit("simulated fields found in generated episodes:\n" + "\n".join(offenders))


def validate_all_runs_succeeded(root: Path) -> None:
    offenders: List[str] = []
    for item in _load_jsonl(root / "manifest_results.jsonl"):
        rc = int(item.get("rc", 1) or 0)
        if rc == 0:
            continue
        offenders.append(
            f"{item.get('variant', 'unknown')}/{item.get('fluctuation', 'unknown')}/rep_{item.get('repeat', '?')}: "
            f"rc={rc} run_root={item.get('run_root', '')}"
        )
    if offenders:
        raise SystemExit("ablation run failures; refusing to summarize incomplete real experiment:\n" + "\n".join(offenders))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _latest_trace_for_run(run_root: Path) -> Optional[Path]:
    traces = sorted(run_root.rglob("trace.json"), key=lambda p: p.stat().st_mtime)
    return traces[-1] if traces else None


def validate_real_recovery_events(root: Path) -> None:
    manifest = _load_jsonl(root / "manifest.jsonl")
    results = {
        str(Path(str(item.get("run_root", ""))).resolve()): int(item.get("rc", 1) or 0)
        for item in _load_jsonl(root / "manifest_results.jsonl")
    }
    offenders: List[str] = []
    for item in manifest:
        variant = str(item.get("variant", ""))
        if variant not in {"full", "shadow_transfer", "shadow_recompute"}:
            continue
        run_root = Path(str(item.get("run_root", ""))).resolve()
        if results.get(str(run_root), 1) != 0:
            continue
        trace_path = _latest_trace_for_run(run_root)
        if trace_path is None:
            offenders.append(f"{variant}: missing trace.json under {run_root}")
            continue
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception as exc:
            offenders.append(f"{variant}: cannot read trace {trace_path}: {exc}")
            continue
        metrics = trace.get("agent_metrics", {})
        if not isinstance(metrics, dict):
            offenders.append(f"{variant}: missing agent_metrics in {trace_path}")
            continue
        if not bool(metrics.get("task_success", False)):
            offenders.append(f"{variant}: task_success=false in {trace_path}")
        llm_events = trace.get("llm_events")
        if not isinstance(llm_events, list):
            llm_events = trace.get("events", [])
        ablation_events: List[Dict[str, Any]] = []
        if isinstance(llm_events, list):
            for ev in llm_events:
                ev_metrics = ev.get("metrics", {}) if isinstance(ev, dict) else {}
                if isinstance(ev_metrics, dict):
                    for rec in ev_metrics.get("ablation_events", []) or []:
                        if isinstance(rec, dict):
                            ablation_events.append(rec)

        dynamic_plan = item.get("dynamic_plan", {}) if isinstance(item.get("dynamic_plan"), dict) else {}
        expected_command_count = max(1, int(dynamic_plan.get("expected_command_count", 1) or 1))
        scope = str(dynamic_plan.get("shadow_scope") or "")
        if not scope:
            suite_config = root / "suite_config.json"
            if suite_config.exists():
                try:
                    scope = str(json.loads(suite_config.read_text(encoding="utf-8")).get("shadow_scope", ""))
                except Exception:
                    scope = ""
        apply_field = "plan_apply_count"
        apply_count = int(metrics.get(apply_field, 0) or 0)
        if apply_count < expected_command_count:
            offenders.append(f"{variant}: {apply_field}={apply_count} < expected {expected_command_count} in {trace_path}")
        dynamic_plan_event_count = int(metrics.get("dynamic_plan_event_count", 0) or 0)
        if dynamic_plan_event_count < expected_command_count:
            offenders.append(
                f"{variant}: dynamic_plan_event_count={dynamic_plan_event_count} < expected {expected_command_count} in {trace_path}"
            )
        log_paths = [
            Path(str(ev.get("log_path", "")))
            for ev in (llm_events if isinstance(llm_events, list) else [])
            if isinstance(ev, dict) and ev.get("log_path")
        ]
        for log_path in log_paths:
            try:
                if log_path.exists() and "late-trigger" in log_path.read_text(encoding="utf-8", errors="replace"):
                    offenders.append(f"{variant}: late-trigger detected in {log_path}")
            except Exception:
                continue

        recover_field = "head_migration_recover_count" if scope == "intra_stage_heads" else "pp_migration_recover_count"
        recover_count = int(metrics.get(recover_field, 0) or 0)
        if variant == "full":
            if scope == "intra_stage_heads" and recover_count > 0:
                offenders.append(f"{variant}: unexpected {recover_field}={recover_count} in {trace_path}")
            if scope == "inter_stage_layers":
                if recover_count < expected_command_count:
                    offenders.append(f"{variant}: expected {recover_field}>={expected_command_count} in {trace_path}")
                recover_event_id = "pp_migration_recover"
                ok_recover_events = [
                    rec for rec in ablation_events
                    if str(rec.get("event_id", "")) == recover_event_id and bool(rec.get("apply_success", False))
                ]
                if not any(str(rec.get("fallback_reason", "")) == "shadow_kv_precomputed_redundant_state" for rec in ok_recover_events):
                    offenders.append(f"{variant}: missing shadow_kv_precomputed_redundant_state in {trace_path}")
            continue
        if recover_count < expected_command_count:
            offenders.append(f"{variant}: {recover_field}={recover_count} < expected {expected_command_count} in {trace_path}")
        recover_event_id = "head_migration_recover" if scope == "intra_stage_heads" else "pp_migration_recover"
        ok_recover_events = [
            rec for rec in ablation_events
            if str(rec.get("event_id", "")) == recover_event_id and bool(rec.get("apply_success", False))
        ]
        if not ok_recover_events:
            offenders.append(f"{variant}: no successful {recover_event_id} apply_success=true in {trace_path}")
        expected_reasons = {
            ("intra_stage_heads", "shadow_transfer"): "shadow_kv_disabled_head_real_transfer",
            ("intra_stage_heads", "shadow_recompute"): "shadow_kv_disabled_head_real_recompute_replay",
            ("inter_stage_layers", "shadow_transfer"): "shadow_kv_disabled_real_transfer",
            ("inter_stage_layers", "shadow_recompute"): "shadow_kv_disabled_real_recompute_replay",
        }
        expected_reason = expected_reasons.get((scope, variant))
        if expected_reason:
            real_reason_events = [
                rec for rec in ok_recover_events
                if str(rec.get("fallback_reason", "")) == expected_reason
            ]
            if not real_reason_events:
                offenders.append(f"{variant}: missing real recovery fallback_reason={expected_reason} in {trace_path}")
        if variant == "shadow_transfer" and int(metrics.get("state_transfer_bytes_total", 0) or 0) <= 0:
            offenders.append(f"{variant}: state_transfer_bytes_total=0 in {trace_path}")
        if variant == "shadow_recompute" and int(metrics.get("recompute_tokens_or_layers_total", 0) or 0) <= 0:
            offenders.append(f"{variant}: recompute_tokens_or_layers_total=0 in {trace_path}")
    if offenders:
        raise SystemExit("real recovery validation failed:\n" + "\n".join(offenders))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EdgeVisor agentic ablations with real compute and TCP-rate perturbations.")
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--out-root", type=Path, default=Path(f"/home/byh/B01/agentic_ablation_results/real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "3")))
    parser.add_argument("--smoke", action="store_true", default=os.environ.get("SMOKE", "0") == "1")
    parser.add_argument("--variants", default=os.environ.get("VARIANTS", ""))
    parser.add_argument("--fluctuations", default=os.environ.get("FLUCTUATIONS", ""))
    parser.add_argument("--shadow-scope", choices=SHADOW_SCOPES, default=os.environ.get("SHADOW_SCOPE", "inter_stage_layers"))
    parser.add_argument("--cuda-visible", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2"))
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--edge-steps", type=int, default=256)
    parser.add_argument("--edge-virtual-launch-stagger-s", type=float, default=2.0)
    parser.add_argument("--runtime-redundant-boundary-layers", type=int, default=0)
    parser.add_argument("--bubble-shadow-kv", action="store_true", default=os.environ.get("BUBBLE_SHADOW_KV", "0") == "1")
    parser.add_argument("--edge-benchmark", action="store_true", default=os.environ.get("EDGE_BENCHMARK", "0") == "1")
    parser.add_argument(
        "--migration-layer-count",
        type=int,
        default=5,
        help="Number of PP boundary layers to migrate in set_pp_migration commands.",
    )
    parser.add_argument("--intra-head-move", type=int, default=2)
    parser.add_argument("--intra-shadow-kv-pad", type=int, default=4)
    parser.add_argument("--intra-stage-index", type=int, default=0)
    parser.add_argument("--intra-trigger-pos-offset", type=int, default=1)
    parser.add_argument("--trigger-generation", default="agent_step_02")
    parser.add_argument("--plan-mode", choices=["next_barrier", "exact"], default="next_barrier")
    parser.add_argument(
        "--trigger-position",
        type=int,
        default=-1,
        help="For --plan-mode exact, trigger migration at this KV position after real prefill.",
    )
    parser.add_argument(
        "--multi-trigger-positions",
        default="",
        help="Comma-separated exact KV positions for multiple migrations in the same generation, e.g. 128,256.",
    )
    parser.add_argument(
        "--multi-trigger-inter-command-delay-s",
        type=float,
        default=0.0,
        help="Delay before each command after the first when --multi-trigger-positions is used.",
    )
    parser.add_argument(
        "--multi-migration-pattern",
        choices=["roundtrip", "right_chain", "same_edge"],
        default="roundtrip",
        help="Intra-stage move sequence for --multi-trigger-positions.",
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=0,
        help="Append deterministic prefill text on the trigger generation so KV cache is larger before migration.",
    )
    parser.add_argument("--prefill-text", default="")
    parser.add_argument("--consume-timeout-s", type=float, default=30.0)
    parser.add_argument("--fixed-port-base", type=int, default=32000)
    parser.add_argument("--compute-target-gpu", type=int, default=-1)
    parser.add_argument("--compute-stress-steps", type=int, default=1024)
    parser.add_argument("--network-intra-mib-s", type=float, default=100.0)
    parser.add_argument("--network-inter-mib-s", type=float, default=10.0)
    parser.add_argument("--network-mixed-mib-s", type=float, default=10.0)
    parser.add_argument(
        "--network-proxy-scope",
        choices=["all_worker_connections", "selected_worker_nodes", "migration_nodes"],
        default="migration_nodes",
        help="Scope for user-space TCP rate proxy injection.",
    )
    parser.add_argument(
        "--network-proxy-nodes",
        default="",
        help="Logical worker nodes to throttle when the scope is selected_worker_nodes or migration_nodes.",
    )
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()

    if args.cuda_visible != "0,1,2":
        raise SystemExit("Refusing to run: CUDA_VISIBLE_DEVICES must be exactly 0,1,2 so GPU3 is not used.")
    if not args.episode.exists():
        raise SystemExit(f"missing episode file: {args.episode}")
    if not args.python_bin.exists():
        raise SystemExit(f"missing python interpreter: {args.python_bin}")
    multi_trigger_positions = parse_int_csv(args.multi_trigger_positions)
    if args.plan_mode == "exact" and int(args.trigger_position) < 0 and not multi_trigger_positions:
        raise SystemExit("--plan-mode exact requires --trigger-position >= 0")
    if multi_trigger_positions and args.plan_mode != "exact":
        raise SystemExit("--multi-trigger-positions requires --plan-mode exact")
    if len(multi_trigger_positions) > 6:
        raise SystemExit("--multi-trigger-positions supports at most 6 migrations for this ablation")

    repeats = 1 if args.smoke else max(1, args.repeats)
    if args.runtime_redundant_boundary_layers <= 0:
        args.runtime_redundant_boundary_layers = max(1, int(args.migration_layer_count))
    variants = selected_variants(args)
    fluctuations = selected_fluctuations(args)
    effective_compute_gpu = scope_compute_target_gpu(args)
    effective_network_nodes = scope_network_nodes(args)
    effective_network_rate = scope_network_rate_mib(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    suite_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "episode": str(args.episode),
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "variants": variants,
        "fluctuations": fluctuations,
        "repeats": repeats,
        "shadow_scope": args.shadow_scope,
        "topology": {
            "logical_nodes": 8,
            "stage_layout": [3, 3, 2],
            "ratios": "1:1:1*1:1:1*1:1",
            "node_gpu_map": "0,1,2,0,1,2,0,1",
            "gpu3_allowed": False,
            "migration_layer_count": args.migration_layer_count,
            "intra_head_move": args.intra_head_move,
            "intra_shadow_kv_pad": args.intra_shadow_kv_pad,
            "intra_stage_index": args.intra_stage_index,
            "intra_trigger_pos_offset": args.intra_trigger_pos_offset,
            "trigger_generation": args.trigger_generation,
            "plan_mode": args.plan_mode,
            "trigger_position": args.trigger_position,
            "multi_trigger_positions": multi_trigger_positions,
            "multi_migration_pattern": args.multi_migration_pattern,
            "expected_migration_count": len(multi_trigger_positions) if multi_trigger_positions else 1,
            "prefill_tokens": args.prefill_tokens,
            "runtime_redundant_boundary_layers": args.runtime_redundant_boundary_layers,
            "bubble_shadow_kv": bool(args.bubble_shadow_kv),
            "edge_benchmark": bool(args.edge_benchmark),
        },
        "perturbation": {
            "compute": "background dllama inference on target GPU",
            "compute_target_gpu": effective_compute_gpu,
            "network": "user-space TCP rate proxy on selected root-worker TCP links; actual bytes are throttled, no simulated stalls",
            "network_proxy_scope": args.network_proxy_scope,
            "network_proxy_nodes": effective_network_nodes,
            "network_rate_mib_per_s": effective_network_rate,
            "simulated_fields_allowed": False,
        },
    }
    (args.out_root / "suite_config.json").write_text(json.dumps(suite_config, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        for variant_idx, variant_name in enumerate(variants):
            modes = VARIANTS[variant_name]
            for fluct_idx, fluctuation in enumerate(fluctuations):
                for repeat in range(1, repeats + 1):
                    run_root = args.out_root / variant_name / fluctuation / f"rep_{repeat}"
                    run_root.mkdir(parents=True, exist_ok=True)
                    episode_copy = run_root / "episode.json"
                    if multi_trigger_positions:
                        plan = write_multi_trigger_episode_copy(
                            args.episode,
                            episode_copy,
                            fluctuation,
                            variant_name,
                            repeat,
                            args,
                            multi_trigger_positions,
                            inter_command_delay_s=float(args.multi_trigger_inter_command_delay_s),
                        )
                    else:
                        plan = write_episode_copy(args.episode, episode_copy, fluctuation, variant_name, repeat, args)
                    preferred_port_base = args.fixed_port_base + variant_idx * 2000 + fluct_idx * 600 + repeat * 200
                    port_base = choose_port_base(preferred_port_base, worker_count=7)
                    cmd = build_episode_command(args, variant_name, modes, fluctuation, repeat, run_root, episode_copy, port_base)
                    perturbation_record: Dict[str, Any] = {"fluctuation": fluctuation, "compute": None, "network": None}
                    stress_proc: Optional[subprocess.Popen[str]] = None
                    started_at = time.time()
                    rc = 1
                    try:
                        if fluctuation in {"compute", "mixed_bw"}:
                            stress_proc = start_compute_stress(
                                run_root,
                                target_gpu=effective_compute_gpu,
                                cuda_visible=args.cuda_visible,
                                steps=args.compute_stress_steps,
                                model=args.model,
                                tokenizer=args.tokenizer,
                            )
                            perturbation_record["compute"] = {
                                "enabled": True,
                                "target_gpu": effective_compute_gpu,
                                "pid": stress_proc.pid,
                                "method": "background_dllama_inference",
                            }
                            time.sleep(2.0)
                        rate = network_proxy_rate(fluctuation, args)
                        if rate is not None:
                            perturbation_record["network"] = {
                                "enabled": True,
                                "method": "user_space_tcp_rate_proxy",
                                "scope": args.network_proxy_scope,
                                "nodes": effective_network_nodes,
                                "rate_mib_per_s": rate,
                            }
                        manifest_item = {
                            "variant": variant_name,
                            "modes": modes,
                            "fluctuation": fluctuation,
                            "repeat": repeat,
                            "run_root": str(run_root),
                            "episode": str(episode_copy),
                            "dynamic_plan": plan,
                            "port_base": port_base,
                            "command": cmd,
                            "perturbation": perturbation_record,
                            "simulated_fields": False,
                            "started_at": datetime.now().isoformat(timespec="seconds"),
                        }
                        append_manifest(manifest_path, manifest_item)
                        print(f"[ablation] start variant={variant_name} fluctuation={fluctuation} repeat={repeat}", flush=True)
                        rc = run_episode_process(cmd, run_root, timeout_s=args.timeout_s, cuda_visible=args.cuda_visible)
                    finally:
                        terminate_proc(stress_proc)
                    elapsed_s = time.time() - started_at
                    result_item = {
                        "variant": variant_name,
                        "fluctuation": fluctuation,
                        "repeat": repeat,
                        "run_root": str(run_root),
                        "rc": rc,
                        "elapsed_s": elapsed_s,
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    append_manifest(args.out_root / "manifest_results.jsonl", result_item)
                    print(
                        f"[ablation] done variant={variant_name} fluctuation={fluctuation} repeat={repeat} rc={rc} elapsed_s={elapsed_s:.1f}",
                        flush=True,
                    )
                    if rc != 0:
                        print(f"[ablation] warning: episode failed rc={rc}; see {run_root / 'run_stdout.json'}", flush=True)
    finally:
        pass

    validate_no_simulated_fields(args.out_root)
    validate_all_runs_succeeded(args.out_root)
    validate_real_recovery_events(args.out_root)
    summary_cmd = [str(args.python_bin), str(PROJECT_ROOT / "scripts/summarize_ablation_results.py"), str(args.out_root)]
    print("[ablation] summarizing", " ".join(shlex.quote(x) for x in summary_cmd), flush=True)
    run_checked(summary_cmd, timeout_s=120)
    print(json.dumps({"out_root": str(args.out_root), "manifest": str(manifest_path), "xlsx": str(args.out_root / "ablation_results.xlsx")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

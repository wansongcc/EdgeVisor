#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
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

SMOKE_VARIANTS = ["full", "shadow_transfer", "shadow_recompute", "pointer_rebuild", "jit_static", "vg_flat"]
FULL_FLUCTUATIONS = ["compute", "intra_stage_bw", "inter_stage_bw", "mixed_bw"]


class PerturbationError(RuntimeError):
    pass


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
    rates = {
        "intra_stage_bw": float(args.network_intra_mib_s),
        "inter_stage_bw": float(args.network_inter_mib_s),
        "mixed_bw": float(args.network_mixed_mib_s),
    }
    return rates.get(fluctuation)


def start_compute_stress(
    run_root: Path,
    target_gpu: int,
    cuda_visible: str,
    steps: int,
) -> subprocess.Popen[str]:
    if not DLLAMA.exists():
        raise PerturbationError(f"missing dllama binary: {DLLAMA}")
    for path in [MODEL, TOKENIZER]:
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
        str(MODEL),
        "--tokenizer",
        str(TOKENIZER),
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


def write_episode_copy(src: Path, dst: Path, fluctuation: str, variant_name: str, repeat: int) -> Dict[str, Any]:
    episode = json.loads(src.read_text(encoding="utf-8"))
    plan = dict(episode.get("edgevisor_dynamic_plan", {}))
    plan.update(
        {
            "enabled": True,
            "trigger_generation": "agent_step_02",
            "plan_op": "set_pp_migration",
            "mode": "next_barrier",
            "stage": 1,
            "from_node": 5,
            "to_node": 6,
            "layerCount": 1,
            "delay_s": 0.5,
            "consume_timeout_s": 30.0,
            "fluctuation_type": fluctuation,
            "candidate_count": 2,
            "state_transfer_bytes": 0,
            "recompute_tokens_or_layers": 0,
            "fallback_count": 0,
            "experiment_variant": variant_name,
            "repeat": repeat,
        }
    )
    plan.pop("simulated_stall_ms", None)
    plan.pop("simulated_recovery_latency_ms", None)
    plan.pop("moves", None)
    if variant_name == "jit_static":
        plan.update({"from_node": 3, "to_node": 6, "candidate_count": 1, "fallback_count": 1})
    elif variant_name == "jit_greedy":
        plan.update({"from_node": 5, "to_node": 7, "candidate_count": 2, "fallback_count": 1})
    episode["edgevisor_dynamic_plan"] = plan
    dst.write_text(json.dumps(episode, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


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
        "--enable-pp-migration",
        "--runtime-redundant-boundary-layers",
        str(args.runtime_redundant_boundary_layers),
    ]
    rate = network_proxy_rate(fluctuation, args)
    if rate is not None:
        config_path = run_root / "ablation_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "network_proxy_rate_mbps": rate,
                    "network_proxy_mode": "user_space_tcp_rate_proxy",
                    "network_proxy_scope": args.network_proxy_scope,
                    "network_proxy_nodes": parse_csv(args.network_proxy_nodes),
                    "network_proxy_from_node": 5,
                    "network_proxy_to_node": 6,
                    "network_proxy_start_throttled": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
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
        names = ["compute"]
    else:
        names = list(FULL_FLUCTUATIONS)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EdgeVisor agentic ablations with real compute and TCP-rate perturbations.")
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--out-root", type=Path, default=Path(f"/home/byh/B01/agentic_ablation_results/real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "3")))
    parser.add_argument("--smoke", action="store_true", default=os.environ.get("SMOKE", "0") == "1")
    parser.add_argument("--variants", default=os.environ.get("VARIANTS", ""))
    parser.add_argument("--fluctuations", default=os.environ.get("FLUCTUATIONS", ""))
    parser.add_argument("--cuda-visible", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2"))
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--edge-steps", type=int, default=256)
    parser.add_argument("--edge-virtual-launch-stagger-s", type=float, default=2.0)
    parser.add_argument("--runtime-redundant-boundary-layers", type=int, default=1)
    parser.add_argument("--fixed-port-base", type=int, default=32000)
    parser.add_argument("--compute-target-gpu", type=int, default=2)
    parser.add_argument("--compute-stress-steps", type=int, default=1024)
    parser.add_argument("--network-intra-mib-s", type=float, default=10.0)
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
        default="5,6",
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

    repeats = 1 if args.smoke else max(1, args.repeats)
    variants = selected_variants(args)
    fluctuations = selected_fluctuations(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    suite_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "episode": str(args.episode),
        "variants": variants,
        "fluctuations": fluctuations,
        "repeats": repeats,
        "topology": {
            "logical_nodes": 8,
            "stage_layout": [3, 3, 2],
            "ratios": "1:1:1*1:1:1*1:1",
            "node_gpu_map": "0,1,2,0,1,2,0,1",
            "gpu3_allowed": False,
        },
        "perturbation": {
            "compute": "background dllama inference on target GPU",
            "network": "user-space TCP rate proxy on selected root-worker TCP links; actual bytes are throttled, no simulated stalls",
            "network_proxy_scope": args.network_proxy_scope,
            "network_proxy_nodes": parse_csv(args.network_proxy_nodes),
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
                    plan = write_episode_copy(args.episode, episode_copy, fluctuation, variant_name, repeat)
                    port_base = args.fixed_port_base + variant_idx * 1000 + fluct_idx * 100 + repeat * 10
                    cmd = build_episode_command(args, variant_name, modes, fluctuation, repeat, run_root, episode_copy, port_base)
                    perturbation_record: Dict[str, Any] = {"fluctuation": fluctuation, "compute": None, "network": None}
                    stress_proc: Optional[subprocess.Popen[str]] = None
                    started_at = time.time()
                    rc = 1
                    try:
                        if fluctuation in {"compute", "mixed_bw"}:
                            stress_proc = start_compute_stress(
                                run_root,
                                target_gpu=args.compute_target_gpu,
                                cuda_visible=args.cuda_visible,
                                steps=args.compute_stress_steps,
                            )
                            perturbation_record["compute"] = {
                                "enabled": True,
                                "target_gpu": args.compute_target_gpu,
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
                                "nodes": parse_csv(args.network_proxy_nodes),
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
                        raise SystemExit(f"episode failed rc={rc}; see {run_root / 'run_stdout.json'}")
    finally:
        pass

    validate_no_simulated_fields(args.out_root)
    summary_cmd = [str(args.python_bin), str(PROJECT_ROOT / "scripts/summarize_ablation_results.py"), str(args.out_root)]
    print("[ablation] summarizing", " ".join(shlex.quote(x) for x in summary_cmd), flush=True)
    run_checked(summary_cmd, timeout_s=120)
    print(json.dumps({"out_root": str(args.out_root), "manifest": str(manifest_path), "xlsx": str(args.out_root / "ablation_results.xlsx")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

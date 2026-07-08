#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("B01_ROOT", "/home/byh/B01"))
DEFAULT_PYTHON = DEFAULT_ROOT / "agent_langgraph_venv/bin/python"


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_command(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path, timeout_s: int = 36000) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(shlex.quote(part) for part in cmd)
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"$ {rendered}\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 124


def capture(cmd: List[str], *, cwd: Path, env: Dict[str, str], timeout_s: int = 60) -> str:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout_s)
        return proc.stdout.strip()
    except Exception as exc:
        return f"capture_failed: {exc}"


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def copy_jsonl_children(out_root: Path, filename: str) -> None:
    target = out_root / filename
    target.write_text("", encoding="utf-8")
    for path in sorted(out_root.rglob(filename)):
        if path == target:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if text and not text.endswith("\n"):
            text += "\n"
        with target.open("a", encoding="utf-8") as f:
            f.write(text)


def run_ablation(
    args: argparse.Namespace,
    *,
    label: str,
    subdir: str,
    variants: str,
    fluctuations: str,
    repeats: int,
    extra: List[str] | None = None,
) -> int:
    subroot = args.out_root / "raw_runs" / subdir
    cmd = [
        str(args.python_bin),
        str(PROJECT_ROOT / "scripts/run_real_ablation_suite.py"),
        "--out-root",
        str(subroot),
        "--repeats",
        str(repeats),
        "--variants",
        variants,
        "--fluctuations",
        fluctuations,
        "--shadow-scope",
        args.shadow_scope,
        "--ctx",
        str(args.ctx),
        "--edge-steps",
        str(args.edge_steps),
        "--plan-mode",
        args.plan_mode,
        "--timeout-s",
        str(args.timeout_s),
        "--network-proxy-throttle-ttl-s",
        str(args.network_proxy_throttle_ttl_s),
        "--network-inter-mib-s",
        str(args.network_inter_mib_s),
        "--network-intra-mib-s",
        str(args.network_intra_mib_s),
    ]
    if extra:
        cmd.extend(extra)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible
    append_jsonl(
        args.out_root / "motivation_invocations.jsonl",
        {
            "label": label,
            "subroot": str(subroot),
            "command": cmd,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    rc = run_command(cmd, cwd=PROJECT_ROOT, env=env, log_path=args.out_root / "logs" / f"{subdir}.log", timeout_s=args.timeout_s * 20)
    append_jsonl(
        args.out_root / "motivation_invocation_results.jsonl",
        {
            "label": label,
            "subroot": str(subroot),
            "rc": rc,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return rc


def write_run_commands(path: Path, commands: List[str]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", "export CUDA_VISIBLE_DEVICES=0,1,2", ""]
    lines.extend(commands)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EdgeVisor Motivation Observation experiments.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--cuda-visible", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2"))
    parser.add_argument("--skip-build", action="store_true", default=os.environ.get("SKIP_BUILD", "0") == "1")
    parser.add_argument("--skip-smoke", action="store_true", default=os.environ.get("SKIP_SMOKE", "0") == "1")
    parser.add_argument("--skip-a", action="store_true")
    parser.add_argument("--skip-b", action="store_true")
    parser.add_argument("--a-repeats", type=int, default=3)
    parser.add_argument("--b-repeats", type=int, default=3)
    parser.add_argument("--a-fluctuation", default="mixed_bw")
    parser.add_argument("--b-fluctuation", default="mixed_bw")
    parser.add_argument("--a-variants", default="stable,boundary_only,full")
    parser.add_argument("--b-variants", default="full,shadow_transfer")
    parser.add_argument("--b-prefill-tokens", default="0,256,512,1024")
    parser.add_argument("--shadow-scope", choices=["inter_stage_layers", "intra_stage_heads"], default="inter_stage_layers")
    parser.add_argument("--plan-mode", choices=["next_barrier", "exact"], default="next_barrier")
    parser.add_argument("--trigger-position", type=int, default=-1)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--edge-steps", type=int, default=256)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--network-proxy-throttle-ttl-s", type=float, default=float(os.environ.get("NETWORK_PROXY_THROTTLE_TTL_S", "15")))
    parser.add_argument("--network-inter-mib-s", type=float, default=float(os.environ.get("NETWORK_INTER_MIB_S", "64")))
    parser.add_argument("--network-intra-mib-s", type=float, default=float(os.environ.get("NETWORK_INTRA_MIB_S", "64")))
    args = parser.parse_args()

    if args.cuda_visible != "0,1,2":
        raise SystemExit("Refusing to run: CUDA_VISIBLE_DEVICES must be exactly 0,1,2 so GPU3 is not used.")
    if args.plan_mode == "exact" and args.trigger_position < 0:
        raise SystemExit("--plan-mode exact requires --trigger-position >= 0")

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible

    environment = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "project_root": str(PROJECT_ROOT),
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, env=env),
        "git_status_short": capture(["git", "status", "--short", "--branch"], cwd=PROJECT_ROOT, env=env),
        "cuda_visible_devices": args.cuda_visible,
        "gpu_inventory": capture(["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true"], cwd=PROJECT_ROOT, env=env),
        "model_path": str(DEFAULT_ROOT / "models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"),
        "tokenizer_path": str(DEFAULT_ROOT / "models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"),
        "ctx": args.ctx,
        "edge_steps": args.edge_steps,
        "shadow_scope": args.shadow_scope,
        "plan_mode": args.plan_mode,
        "trigger_position": args.trigger_position,
        "network_proxy_throttle_ttl_s": args.network_proxy_throttle_ttl_s,
        "network_inter_mib_s": args.network_inter_mib_s,
        "network_intra_mib_s": args.network_intra_mib_s,
    }
    (args.out_root / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")

    commands_for_rerun: List[str] = []
    rc_total = 0
    if not args.skip_build:
        build_cmd = ["bash", "build.sh"]
        commands_for_rerun.append("bash build.sh")
        rc = run_command(build_cmd, cwd=PROJECT_ROOT, env=env, log_path=args.out_root / "logs" / "build.log", timeout_s=7200)
        rc_total += rc
        if rc != 0:
            print(f"[motivation] build failed rc={rc}; see {args.out_root / 'logs/build.log'}", flush=True)
            write_run_commands(args.out_root / "run_commands.sh", commands_for_rerun)
            return rc

    if not args.skip_smoke:
        smoke_root = args.out_root / "smoke"
        smoke_cmd = ["bash", "scripts/run_agentic_ablation_suite.sh"]
        smoke_env = dict(env)
        smoke_env.update({"SMOKE": "1", "OUT_ROOT": str(smoke_root)})
        commands_for_rerun.append(f"SMOKE=1 OUT_ROOT={shlex.quote(str(smoke_root))} bash scripts/run_agentic_ablation_suite.sh")
        rc = run_command(smoke_cmd, cwd=PROJECT_ROOT, env=smoke_env, log_path=args.out_root / "logs" / "smoke.log", timeout_s=args.timeout_s * 8)
        rc_total += rc
        if rc != 0:
            print(f"[motivation] smoke failed rc={rc}; continuing so failure is preserved", flush=True)

    if not args.skip_a:
        if "stable" in parse_csv(args.a_variants):
            rc_total += run_ablation(args, label="A-stable", subdir="obs1_stable", variants="stable", fluctuations="none", repeats=args.a_repeats)
            commands_for_rerun.append(
                f"{shlex.quote(str(args.python_bin))} scripts/run_real_ablation_suite.py --out-root {shlex.quote(str(args.out_root / 'raw_runs/obs1_stable'))} --repeats {args.a_repeats} --variants stable --fluctuations none"
            )
        non_stable = ",".join(v for v in parse_csv(args.a_variants) if v != "stable")
        if non_stable:
            rc_total += run_ablation(args, label="A-degraded", subdir="obs1_degraded", variants=non_stable, fluctuations=args.a_fluctuation, repeats=args.a_repeats)
            commands_for_rerun.append(
                f"{shlex.quote(str(args.python_bin))} scripts/run_real_ablation_suite.py --out-root {shlex.quote(str(args.out_root / 'raw_runs/obs1_degraded'))} --repeats {args.a_repeats} --variants {shlex.quote(non_stable)} --fluctuations {shlex.quote(args.a_fluctuation)}"
            )

    if not args.skip_b:
        trigger_args: List[str] = []
        if args.plan_mode == "exact":
            trigger_args = ["--trigger-position", str(args.trigger_position)]
        for prefill in parse_csv(args.b_prefill_tokens):
            subdir = f"obs2_prefill_{prefill}"
            extra = ["--prefill-tokens", str(int(prefill))]
            extra.extend(trigger_args)
            rc_total += run_ablation(
                args,
                label=f"B-prefill-{prefill}",
                subdir=subdir,
                variants=args.b_variants,
                fluctuations=args.b_fluctuation,
                repeats=args.b_repeats,
                extra=extra,
            )
            commands_for_rerun.append(
                f"{shlex.quote(str(args.python_bin))} scripts/run_real_ablation_suite.py --out-root {shlex.quote(str(args.out_root / 'raw_runs' / subdir))} --repeats {args.b_repeats} --variants {shlex.quote(args.b_variants)} --fluctuations {shlex.quote(args.b_fluctuation)} --prefill-tokens {prefill}"
            )

    copy_jsonl_children(args.out_root, "manifest.jsonl")
    copy_jsonl_children(args.out_root, "manifest_results.jsonl")
    write_run_commands(args.out_root / "run_commands.sh", commands_for_rerun)
    readme = [
        "# EdgeVisor Motivation Experiments",
        "",
        f"Created: {environment['created_at']}",
        f"Git commit: `{environment['git_commit']}`",
        f"CUDA_VISIBLE_DEVICES: `{args.cuda_visible}`",
        "",
        "Raw run directories are under `raw_runs/`. The root `manifest*.jsonl` files merge all child manifests for summarization.",
    ]
    (args.out_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "rc_total": rc_total}, indent=2), flush=True)
    return 0 if rc_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

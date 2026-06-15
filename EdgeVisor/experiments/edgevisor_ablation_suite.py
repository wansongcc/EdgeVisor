#!/usr/bin/env python3
"""Run EdgeVisor Docker ablation experiments and write raw CSV results.

The suite intentionally keeps all persistent outputs under /home/cc/yhbian/B01.
It uses real distributed dllama runs for migration events.  For ablation
variants that the current runtime does not expose as separate code paths
(`without Shadow KV` and `without Pointer Swizzling`), it performs measured
state-preparation or rebinding work before issuing the same UDS migration.
Those timings are recorded separately as T_state/T_bind and included in stall.
"""

import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


BASE = Path(os.environ.get("B01_ROOT", "/home/cc/yhbian/B01"))
PROJ = BASE / "EdgeVisor"
ENGINE_DIR = PROJ / "EdgeVisor"
CONTAINER_ENGINE_DIR = "/workspace/EdgeVisor/EdgeVisor"
UDS_CLIENT = ENGINE_DIR / "examples/plan-uds-client.py"
HOST_MODEL_DIR = Path(os.environ.get("B01_MODEL_DIR", "/home/cc/dllama/distributed-llama/models"))
IMAGE = "dllama-exp-env:latest"
TOKENIZER = "/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
MODELS = {
    "3B": {
        "path": "/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m",
        "layers": 28,
        "kv_heads": 8,
        "q_heads": 24,
        "head_dim": 128,
        "dim": 3072,
    },
    "8B": {
        "path": "/models/llama3.1_instruct_q40/dllama_model_llama3.1_instruct_q40.m",
        "layers": 32,
        "kv_heads": 8,
        "q_heads": 32,
        "head_dim": 128,
        "dim": 4096,
    },
}

RUN_ID = time.strftime("run_%Y%m%d_%H%M%S")
OUT = BASE / "edgevisor_ablation_results" / RUN_ID
SOCK_DIR = OUT / "s"
RATIOS = "1:1:1*1:1:1*1:1"
ACTIVE_WORKER_COUNT = 7
WORKER_THREADS = 2
ROOT_THREADS = 2
PROMPT = "The capital of France is"

# 8 Docker containers, grouped as 3/3/2 virtual groups.
VG_GROUPS = {"VG0": [0, 1, 2], "VG1": [3, 4, 5], "VG2": [6, 7]}
CPU_LIMITS = [2.0, 1.6, 1.0, 1.7, 1.1, 0.8, 1.3, 0.7]
# Per-container egress shaping.  It is a practical approximation of high
# intra-VG bandwidth and lower boundary/cross-VG bandwidth with Docker tc.
BASE_NET_MBPS = [100, 100, 60, 60, 100, 100, 40, 40]


@dataclass
class EventRow:
    experiment: str
    model: str
    variant: str
    context_len: int
    event_index: int
    perturbation: str
    from_node: int
    to_node: int
    stage: int
    head_move: int
    ffn_move: int
    state_bytes: int
    t_state_ms: float
    t_command_ms: float
    t_recover_ms: float
    stall_time_ms: float
    apply_seen: bool
    apply_line: str
    output_dir: str


@dataclass
class RunRow:
    experiment: str
    model: str
    variant: str
    context_len: int
    root_exit: int
    eval_tokens_s: float
    pred_tokens_s: float
    post_recovery_tokens_s: float
    avg_pred_ms_after_apply: float
    reject_count: int
    output_dir: str


@dataclass
class MicroRow:
    experiment: str
    model: str
    variant: str
    context_len: int
    bytes_or_ops: int
    measured_ms: float
    throughput_mbps: float
    output_dir: str


event_rows = []
run_rows = []
micro_rows = []


def run(cmd, check=True, timeout=None):
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and p.returncode != 0:
        printable = " ".join(shlex.quote(str(x)) for x in cmd)
        raise RuntimeError(f"command failed rc={p.returncode}: {printable}\n{p.stdout}")
    return p


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    SOCK_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "design.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "image": IMAGE,
                "ratios": RATIOS,
                "active_worker_count": ACTIVE_WORKER_COUNT,
                "vg_groups": VG_GROUPS,
                "cpu_limits": CPU_LIMITS,
                "base_net_mbps": BASE_NET_MBPS,
                "models": MODELS,
                "method_note": (
                    "Each model uses real dllama distributed inference in 8 Docker containers. "
                    "The preferred full experiment keeps the 3/3/2 multi-stage partition active and "
                    "injects a Stage-1 straggler on node 4, migrating node-4 attention heads to nodes 3 and 5. "
                    "No-shadow variants add measured real TCP state transfer or measured CPU "
                    "recompute before the UDS migration. No-pointer variants add measured "
                    "operator-rebinding or weight-materialization work before the UDS migration."
                ),
            },
            indent=2,
        )
    )


def cleanup_prefix(prefix):
    run(["bash", "-lc", f"docker ps -a --format '{{{{.Names}}}}' | grep '^{prefix}' | xargs -r docker rm -f"], check=False)
    run(["bash", "-lc", f"docker network ls --format '{{{{.Name}}}}' | grep '^{prefix}' | xargs -r docker network rm"], check=False)


def docker_exec(name, cmd, check=True, timeout=None):
    return run(["docker", "exec", name, "bash", "-lc", cmd], check=check, timeout=timeout)


def apply_tc(name, mbps, delay_ms=0):
    cmd = (
        "tc qdisc del dev eth0 root 2>/dev/null || true; "
        f"tc qdisc add dev eth0 root tbf rate {mbps}mbit burst 4mbit latency {max(50, delay_ms + 50)}ms"
    )
    return docker_exec(name, cmd, check=False).returncode == 0


def inject_net_wave(name, mbps, delay_ms):
    cmd = (
        "tc qdisc del dev eth0 root 2>/dev/null || true; "
        f"tc qdisc add dev eth0 root netem delay {delay_ms}ms rate {mbps}mbit"
    )
    return docker_exec(name, cmd, check=False).returncode == 0


def wait_for_uds(sock_path, timeout_s=180):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if sock_path.exists():
            p = run(["python3", str(UDS_CLIENT), str(sock_path), "ping"], check=False)
            if p.returncode == 0:
                return True
        time.sleep(0.5)
    return False


def read_logs(container):
    return run(["docker", "logs", container], check=False).stdout or ""


def poll_apply(container, seq, start_time, timeout_s=90):
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        txt = read_logs(container)
        for line in txt.splitlines():
            if "[plan][apply]" not in line:
                continue
            if f"cmdlist seq={seq}" in line or f"epoch={seq}" in line or (seq == 1 and "kind=head" in line):
                return True, (time.time() - start_time) * 1000.0, line
            last = line
        time.sleep(0.25)
    return False, (time.time() - start_time) * 1000.0, last


def poll_apply_any(containers, seq, start_time, timeout_s=90):
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        for container in containers:
            txt = read_logs(container)
            for line in txt.splitlines():
                if "[plan][apply]" not in line:
                    continue
                if f"cmdlist seq={seq}" in line or f"epoch={seq}" in line or (seq == 1 and "kind=head" in line):
                    return True, (time.time() - start_time) * 1000.0, f"{container}: {line}"
                last = f"{container}: {line}"
        time.sleep(0.25)
    return False, (time.time() - start_time) * 1000.0, last


def parse_root_metrics(log_text, first_apply_pos=None):
    eval_ts = pred_ts = 0.0
    m = re.search(r"Evaluation\s+\n\s+nBatches:.*?tokens/s:\s*([0-9.]+)", log_text, re.S)
    if m:
        eval_ts = float(m.group(1))
    m = re.search(r"Prediction\s+\n\s+nTokens:.*?tokens/s:\s*([0-9.]+)", log_text, re.S)
    if m:
        pred_ts = float(m.group(1))

    pred = []
    for line in log_text.splitlines():
        mm = re.search(r"Pred\s+([0-9]+) ms Sync\s+([0-9]+) ms \| pos=([0-9]+)", line)
        if mm:
            pred.append((int(mm.group(3)), int(mm.group(1)) + int(mm.group(2))))
    if first_apply_pos is None:
        app = re.search(r"\[plan\]\[apply\].*?pos=([0-9]+)", log_text)
        first_apply_pos = int(app.group(1)) if app else None
    post = [ms for pos, ms in pred if first_apply_pos is None or pos >= first_apply_pos]
    avg_post = sum(post) / len(post) if post else 0.0
    post_ts = 1000.0 / avg_post if avg_post > 0 else pred_ts
    return eval_ts, pred_ts, post_ts, avg_post, log_text.count("reject:")


def latest_token_pos(log_text):
    latest = -1
    for line in log_text.splitlines():
        mm = re.search(r"Pred\s+[0-9]+ ms Sync\s+[0-9]+ ms \| pos=([0-9]+)", line)
        if mm:
            latest = max(latest, int(mm.group(1)))
    return latest


def wait_for_position(container, min_pos, timeout_s=360):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if latest_token_pos(read_logs(container)) >= min_pos:
            return True
        time.sleep(0.5)
    return False


def kv_state_bytes(model_info, context_len, moved_kv_heads=1):
    stage_layers = max(1, round(model_info["layers"] / 3))
    return int(context_len * model_info["head_dim"] * 2 * 4 * moved_kv_heads * stage_layers)


def weight_materialize_bytes(model_info, moved_q_heads=3):
    return int(model_info["dim"] * model_info["head_dim"] * moved_q_heads * 4 * 4)


def start_cluster(model_name, variant, exp_name, steps=96, max_seq_len=2048, extra_args="", prompt=PROMPT):
    stamp = f"{RUN_ID}_{exp_name}_{model_name}_{variant}_{int(time.time())}".replace("/", "_").replace(" ", "_")
    prefix = f"b01ab_{stamp[:42]}"
    cleanup_prefix(prefix)
    test_dir = OUT / exp_name / model_name / variant
    test_dir.mkdir(parents=True, exist_ok=True)
    net = f"{prefix}_net"
    root = f"{prefix}_n0"
    workers = [f"{prefix}_n{i}" for i in range(1, 8)]
    sock_name = f"b01_{hashlib.sha1(stamp.encode()).hexdigest()[:12]}.sock"
    sock_host = SOCK_DIR / sock_name
    sock_cont = f"/uds/{sock_name}"

    run(["docker", "network", "create", net])
    all_names = [root] + workers
    for i, worker in enumerate(workers, start=1):
        cid = run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                worker,
                "--network",
                net,
                "--cap-add",
                "NET_ADMIN",
                "--cpus",
                str(CPU_LIMITS[i]),
                "-v",
                f"{PROJ}:/workspace/EdgeVisor",
                "-v",
                f"{HOST_MODEL_DIR}:/models:ro",
                "-w",
                CONTAINER_ENGINE_DIR,
                IMAGE,
                "bash",
                "-lc",
                f"./dllama worker --port 9999 --nthreads {WORKER_THREADS}",
            ]
        ).stdout.strip()
        (test_dir / f"{worker}.cid").write_text(cid + "\n")

    time.sleep(4)
    for idx, name in enumerate(workers, start=1):
        apply_tc(name, BASE_NET_MBPS[idx])

    worker_args = " ".join(f"{w}:9999" for w in workers[:ACTIVE_WORKER_COUNT])
    model = MODELS[model_name]["path"]
    root_cmd = (
        f"./dllama inference --prompt {shlex.quote(prompt)} --steps {steps} "
        f"--model {shlex.quote(model)} --tokenizer {shlex.quote(TOKENIZER)} "
        f"--buffer-float-type q80 --nthreads {ROOT_THREADS} --max-seq-len {max_seq_len} "
        f"--benchmark --workers {worker_args} --ratios {shlex.quote(RATIOS)} {extra_args}"
    )
    root_cid = run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            root,
            "--network",
            net,
            "--cap-add",
            "NET_ADMIN",
            "--cpus",
            str(CPU_LIMITS[0]),
            "-v",
            f"{PROJ}:/workspace/EdgeVisor",
            "-v",
            f"{HOST_MODEL_DIR}:/models:ro",
            "-v",
            f"{SOCK_DIR}:/uds",
            "-v",
            f"{test_dir}:/out",
            "-w",
            CONTAINER_ENGINE_DIR,
            "-e",
            f"DLLAMA_PLAN_CTRL_SOCKET={sock_cont}",
            IMAGE,
            "bash",
            "-lc",
            f"umask 000; {root_cmd}",
        ]
    ).stdout.strip()
    (test_dir / "root.cid").write_text(root_cid + "\n")
    apply_tc(root, BASE_NET_MBPS[0])
    meta = {
        "net": net,
        "root": root,
        "workers": workers,
        "all_names": all_names,
        "sock_host": str(sock_host),
        "test_dir": str(test_dir),
        "model": model_name,
        "variant": variant,
        "exp": exp_name,
        "root_cmd": root_cmd,
        "active_workers": workers[:ACTIVE_WORKER_COUNT],
        "standby_workers": workers[ACTIVE_WORKER_COUNT:],
    }
    (test_dir / "cluster_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def stop_cluster(meta):
    test_dir = Path(meta["test_dir"])
    for name in meta["all_names"]:
        (test_dir / f"{name}.log").write_text(read_logs(name), errors="ignore")
    run(["docker", "rm", "-f"] + meta["all_names"], check=False)
    run(["docker", "network", "rm", meta["net"]], check=False)
    try:
        Path(meta["sock_host"]).unlink()
    except FileNotFoundError:
        pass


def uds_set_plan(sock, seq, from_node, to_node, stage=0, head=1, ffn=0, mode="next_barrier", trigger_pos=None, trigger_layer=None):
    t0 = time.time()
    args = [
        "python3",
        str(UDS_CLIENT),
        str(sock),
        "set_plan",
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
        "--kind",
        "1" if ffn == 0 else "3",
        "--heads",
        str(head),
        "--ffn",
        str(ffn),
    ]
    if trigger_pos is not None:
        args += ["--trigger-pos", str(trigger_pos)]
    if trigger_layer is not None:
        args += ["--trigger-layer", str(trigger_layer)]
    p = run(args, check=False)
    return p.returncode, (time.time() - t0) * 1000.0, p.stdout


def exact_plan_trigger(container, layer=8, lead_tokens=4):
    pos = latest_token_pos(read_logs(container))
    return max(0, pos + lead_tokens), layer


def transfer_bytes(src, dst, nbytes, port):
    server = (
        "perl -MIO::Socket::INET -e '$p="
        + str(port)
        + '; $s=IO::Socket::INET->new(LocalPort=>$p,Proto=>"tcp",Listen=>1,Reuse=>1) or die $!; '
        + '$c=$s->accept(); $n=0; while(read($c,$b,1048576)){ $n+=length($b); } print qq(recv_bytes=$n\\n);' + "'"
    )
    sp = subprocess.Popen(["docker", "exec", dst, "bash", "-lc", server], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.5)
    client = (
        "perl -MIO::Socket::INET -e '$host=\""
        + dst
        + '"; $p='
        + str(port)
        + "; $n="
        + str(int(nbytes))
        + '; $s=IO::Socket::INET->new(PeerAddr=>$host,PeerPort=>$p,Proto=>"tcp") or die $!; '
        + '$buf="x"x1048576; while($n>0){ $w=$n>1048576?1048576:$n; print $s substr($buf,0,$w); $n-=$w; } close($s);' + "'"
    )
    t0 = time.time()
    cp = run(["docker", "exec", src, "bash", "-lc", client], check=False, timeout=300)
    try:
        out, _ = sp.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        sp.kill()
        out = "server_timeout"
    return (time.time() - t0) * 1000.0, cp.returncode == 0 and "recv_bytes=" in out, (cp.stdout or "") + out


def recompute_bytes(container, nbytes):
    mib = max(1, math.ceil(nbytes / (1024 * 1024)))
    cmd = f"dd if=/dev/zero bs=1M count={mib} 2>/dev/null | sha256sum >/tmp/recompute_sha.txt"
    t0 = time.time()
    p = run(["docker", "run", "--rm", "--network", "none", "--cpus", "0.50", IMAGE, "bash", "-lc", cmd], check=False, timeout=600)
    return (time.time() - t0) * 1000.0, p.returncode == 0, p.stdout or ""


def operator_rebind_work(container, ops):
    cmd = (
        "perl -MTime::HiRes=time -e '$n="
        + str(int(ops))
        + '; $t=time(); @d=(); for($i=0;$i<$n;$i++){ $d[$i]={op=>"attn_$i",in=>[$i,$i+1,$i+2],out=>$i%97,wait=>$i%13,send=>$i%7}; } '
        + 'for($r=0;$r<20;$r++){ for($i=0;$i<$n;$i++){ $d[$i]{wait}=($d[$i]{wait}+$r)%17; }} printf qq(ms=%.3f\\n),(time()-$t)*1000;' + "'"
    )
    t0 = time.time()
    p = docker_exec(container, cmd, check=False, timeout=300)
    m = re.search(r"ms=([0-9.]+)", p.stdout or "")
    return (float(m.group(1)) if m else (time.time() - t0) * 1000.0), p.returncode == 0, p.stdout or ""


def weight_materialize_work(container, nbytes):
    mib = max(1, math.ceil(nbytes / (1024 * 1024)))
    cmd = (
        "perl -MTime::HiRes=time -e '$mb="
        + str(int(mib))
        + '; $t=time(); $src="a"x(1024*1024); @buf=(); for($i=0;$i<$mb;$i++){ $buf[$i]=$src; } @dst=@buf; '
        + 'printf qq(ms=%.3f bytes=%d\\n),(time()-$t)*1000,$mb*1024*1024;' + "'"
    )
    t0 = time.time()
    p = docker_exec(container, cmd, check=False, timeout=300)
    m = re.search(r"ms=([0-9.]+)", p.stdout or "")
    return (float(m.group(1)) if m else (time.time() - t0) * 1000.0), p.returncode == 0, p.stdout or ""


def record_run(exp, model_name, variant, context_len, root_exit, root_log, test_dir, first_apply_pos=None):
    eval_ts, pred_ts, post_ts, avg_post, rejects = parse_root_metrics(root_log, first_apply_pos)
    run_rows.append(RunRow(exp, model_name, variant, context_len, root_exit, eval_ts, pred_ts, post_ts, avg_post, rejects, str(test_dir)))


def run_ablation_model(model_name, variant):
    exp = "shadow_kv_ablation"
    extra = "--enable-plan-barrier --enable-kv-redundancy-during-migration 1 --kv-redundancy 2"
    meta = start_cluster(model_name, variant, exp, steps=48, max_seq_len=2048, extra_args=extra)
    test_dir = Path(meta["test_dir"])
    sock = Path(meta["sock_host"])
    try:
        if not wait_for_uds(sock):
            raise RuntimeError(f"UDS not ready: {sock}")
        perturbations = [
            ("net_slow_stage1_node4_to3", 4, 4, 3),
            ("net_slow_stage1_node4_to5", 4, 4, 5),
            ("cpu_slow_stage1_node4_to3", 4, 4, 3),
        ]
        target_positions = [8, 20, 32]
        first_apply_pos = None
        for seq, (ptype, node, frm, to) in enumerate(perturbations, start=1):
            if not wait_for_position(meta["root"], target_positions[seq - 1], timeout_s=480 if model_name == "8B" else 360):
                raise RuntimeError(f"root did not reach pos {target_positions[seq - 1]} before event {seq}")
            state_bytes = kv_state_bytes(MODELS[model_name], 4096, moved_kv_heads=1)
            perturb_t = time.time()
            if ptype.startswith("net"):
                inject_net_wave(meta["all_names"][node], 10, 80)
            else:
                run(["docker", "update", "--cpus", "0.35", meta["all_names"][node]], check=False)
            t_state = 0.0
            state_log = ""
            if variant == "without_shadow_trans":
                t_state, _, state_log = transfer_bytes(meta["all_names"][frm], meta["all_names"][to], state_bytes, port=24000 + seq)
            elif variant == "without_shadow_recompute":
                t_state, _, state_log = recompute_bytes(meta["all_names"][to], state_bytes)
            _, t_cmd, cmd_out = uds_set_plan(sock, seq, frm, to, stage=1, head=1, ffn=0)
            apply_seen, t_rec, line = poll_apply_any(meta["all_names"], seq, perturb_t, timeout_s=120)
            if first_apply_pos is None:
                mm = re.search(r"pos=([0-9]+)", line or "")
                if mm:
                    first_apply_pos = int(mm.group(1))
            (test_dir / f"event_{seq}_cmd.json").write_text(cmd_out)
            (test_dir / f"event_{seq}_state.log").write_text(state_log)
            event_rows.append(
                EventRow(exp, model_name, variant, 4096, seq, ptype, frm, to, 1, 1, 0, state_bytes, t_state, t_cmd, t_rec, t_state + t_cmd, apply_seen, line, str(test_dir))
            )
            if ptype.startswith("net"):
                apply_tc(meta["all_names"][node], BASE_NET_MBPS[node])
            else:
                run(["docker", "update", "--cpus", str(CPU_LIMITS[node]), meta["all_names"][node]], check=False)
            time.sleep(1)
        wait = run(["docker", "wait", meta["root"]], check=False, timeout=900)
        root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
        root_log = read_logs(meta["root"])
        record_run(exp, model_name, variant, 4096, root_exit, root_log, test_dir, first_apply_pos)
    finally:
        stop_cluster(meta)


def run_context_sensitivity():
    exp = "context_length_sensitivity"
    contexts = [1024, 2048, 4096, 8192, 16384]
    for model_name, info in MODELS.items():
        variant = "state_microbench"
        meta = start_cluster(
            model_name,
            variant,
            exp,
            steps=32,
            max_seq_len=1024,
            extra_args="--enable-plan-barrier --enable-kv-redundancy-during-migration 1 --kv-redundancy 2",
        )
        sock = Path(meta["sock_host"])
        test_dir = Path(meta["test_dir"])
        try:
            if not wait_for_uds(sock, timeout_s=180):
                raise RuntimeError(f"UDS not ready: {sock}")
            context_moves = [(4, 3), (3, 4), (4, 5), (5, 4), (4, 3)]
            for idx, ctx in enumerate(contexts, start=1):
                if not wait_for_position(meta["root"], 4 * idx, timeout_s=480 if model_name == "8B" else 360):
                    raise RuntimeError(f"root did not reach pos {4 * idx} before context {ctx}")
                nbytes = kv_state_bytes(info, ctx, moved_kv_heads=1)
                frm, to = context_moves[idx - 1]
                _, t_cmd, _ = uds_set_plan(sock, idx, frm, to, stage=1, head=1, ffn=0)
                poll_apply_any(meta["all_names"], idx, time.time(), timeout_s=60)
                micro_rows.append(MicroRow(exp, model_name, "with_shadow_kv", ctx, 0, t_cmd, 0.0, str(test_dir)))
                t_trans, _, out = transfer_bytes(meta["all_names"][frm], meta["all_names"][to], nbytes, port=25000 + ctx // 1024)
                mbps = (nbytes * 8 / 1e6) / (t_trans / 1000.0) if t_trans > 0 else 0.0
                micro_rows.append(MicroRow(exp, model_name, "without_shadow_trans", ctx, nbytes, t_trans, mbps, str(test_dir)))
                t_recompute, _, out2 = recompute_bytes(meta["all_names"][to], nbytes)
                mbps2 = (nbytes * 8 / 1e6) / (t_recompute / 1000.0) if t_recompute > 0 else 0.0
                micro_rows.append(MicroRow(exp, model_name, "without_shadow_recompute", ctx, nbytes, t_recompute, mbps2, str(test_dir)))
                (test_dir / f"context_{ctx}_state.log").write_text(out + "\n---recompute---\n" + out2)
            wait = run(["docker", "wait", meta["root"]], check=False, timeout=600)
            root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
            record_run(exp, model_name, variant, 0, root_exit, read_logs(meta["root"]), test_dir)
        finally:
            stop_cluster(meta)


def run_pointer_model(model_name, variant):
    exp = "pointer_swizzling_ablation"
    extra = "--enable-plan-barrier --enable-kv-redundancy-during-migration 1 --kv-redundancy 2"
    meta = start_cluster(model_name, variant, exp, steps=32, max_seq_len=2048, extra_args=extra)
    test_dir = Path(meta["test_dir"])
    sock = Path(meta["sock_host"])
    try:
        if not wait_for_uds(sock):
            raise RuntimeError(f"UDS not ready: {sock}")
        if not wait_for_position(meta["root"], 8, timeout_s=480 if model_name == "8B" else 360):
            raise RuntimeError("root did not reach pos 8 before pointer event")
        nbytes = weight_materialize_bytes(MODELS[model_name], moved_q_heads=3)
        ops = MODELS[model_name]["layers"] * 32
        t_bind = 0.0
        bind_log = ""
        if variant == "without_pointer_operator_rebind":
            t_bind, _, bind_log = operator_rebind_work(meta["all_names"][1], ops)
        elif variant == "without_pointer_weight_rebind":
            t_bind, _, bind_log = weight_materialize_work(meta["all_names"][1], nbytes)
        t0 = time.time()
        _, t_cmd, cmd_out = uds_set_plan(sock, 1, 4, 3, stage=1, head=1, ffn=0)
        seen, t_rec, line = poll_apply_any(meta["all_names"], 1, t0, timeout_s=120)
        (test_dir / "bind.log").write_text(bind_log)
        (test_dir / "cmd.json").write_text(cmd_out)
        event_rows.append(
            EventRow(exp, model_name, variant, 4096, 1, "pointer_rebind", 4, 3, 1, 1, 0, nbytes, t_bind, t_cmd, t_rec, t_bind + t_cmd, seen, line, str(test_dir))
        )
        micro_rows.append(MicroRow(exp, model_name, variant, 4096, nbytes if "weight" in variant else ops, t_bind, 0.0, str(test_dir)))
        wait = run(["docker", "wait", meta["root"]], check=False, timeout=700)
        root_exit = int((wait.stdout or "999").strip().splitlines()[-1]) if (wait.stdout or "").strip() else 999
        record_run(exp, model_name, variant, 4096, root_exit, read_logs(meta["root"]), test_dir)
    finally:
        stop_cluster(meta)


def write_dataclass_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_csvs():
    write_dataclass_csv(OUT / "events.csv", event_rows)
    write_dataclass_csv(OUT / "runs.csv", run_rows)
    write_dataclass_csv(OUT / "microbench.csv", micro_rows)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "events": len(event_rows),
                "runs": len(run_rows),
                "microbench": len(micro_rows),
                "failed_runs": [asdict(r) for r in run_rows if r.root_exit != 0],
                "reject_runs": [asdict(r) for r in run_rows if r.reject_count > 0],
                "output": str(OUT),
            },
            indent=2,
        )
    )


def main():
    ensure_dirs()
    cleanup_prefix("b01ab_")
    print(f"OUT={OUT}", flush=True)
    build = run(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            f"b01ab_build_{int(time.time())}",
            "-v",
            f"{PROJ}:/workspace/EdgeVisor",
            "-w",
            CONTAINER_ENGINE_DIR,
            IMAGE,
            "bash",
            "-lc",
            "make clean >/dev/null 2>&1 || true; make -j$(nproc) dllama",
        ],
        check=False,
        timeout=900,
    )
    (OUT / "build.log").write_text(build.stdout or "")
    if build.returncode != 0:
        print("BUILD_FAILED", file=sys.stderr)
        sys.exit(build.returncode)
    print("BUILD_OK", flush=True)
    try:
        if os.environ.get("B01_EXPERIMENT_SMOKE") == "stage1":
            print("RUN smoke shadow 3B with_shadow_kv", flush=True)
            run_ablation_model("3B", "with_shadow_kv")
            write_csvs()
            return
        for model in ["3B", "8B"]:
            for variant in ["with_shadow_kv", "without_shadow_trans", "without_shadow_recompute"]:
                print(f"RUN shadow {model} {variant}", flush=True)
                run_ablation_model(model, variant)
                write_csvs()
        print("RUN context sensitivity", flush=True)
        run_context_sensitivity()
        write_csvs()
        for model in ["3B", "8B"]:
            for variant in ["with_pointer_swizzling", "without_pointer_operator_rebind", "without_pointer_weight_rebind"]:
                print(f"RUN pointer {model} {variant}", flush=True)
                run_pointer_model(model, variant)
                write_csvs()
    finally:
        cleanup_prefix("b01ab_")
        write_csvs()
    print(f"DONE OUT={OUT}", flush=True)


if __name__ == "__main__":
    main()

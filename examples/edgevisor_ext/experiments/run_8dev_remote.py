#!/usr/bin/env python3
"""
8-device heterogeneous Docker experiment orchestrator (run on remote host).

Implements:
- Round A: homogeneous 100MB/s full-connect assumption (approximated by per-container 800mbit shaping)
- Round B: heterogeneous baseline topology + two persistent disturbance injections
- Initialization (K in {3,4}) + warmup selection
- Runtime optimizer auto-first, fallback manual triggers for layer/token
- Windowed pre/post metrics around apply points (40 tokens, skip first 5 after apply)
- Markdown report generation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class NodeProfile:
    node_id: int
    name: str
    kind: str
    compute_flops: float
    memory_gb: float


@dataclasses.dataclass(frozen=True)
class LaunchCfg:
    ratios: str
    workers: str
    kv_redundancy: List[int]
    runtime_redundant_boundary_layers: int


NODES: List[NodeProfile] = [
    NodeProfile(0, "ev8_n0", "NX16", 2.2e11, 16.0),
    NodeProfile(1, "ev8_n1", "NX16", 2.2e11, 16.0),
    NodeProfile(2, "ev8_n2", "NANO8", 1.2e11, 8.0),
    NodeProfile(3, "ev8_n3", "NANO8", 1.2e11, 8.0),
    NodeProfile(4, "ev8_n4", "MI13", 1.5e11, 8.0),
    NodeProfile(5, "ev8_n5", "RPI4", 3.5e10, 4.0),
    NodeProfile(6, "ev8_n6", "RPI4", 3.5e10, 4.0),
    NodeProfile(7, "ev8_n7", "RPI4", 3.5e10, 4.0),
]

K_CANDIDATES = [3, 4]

PRED_LINE_RE = re.compile(r"Pred\s*(\d+)\s*ms\s+Sync\s*(\d+)\s*ms\s*\|\s*pos=(\d+)")


def _run(cmd: str, check: bool = True, timeout_s: Optional[int] = None) -> subprocess.CompletedProcess:
    cp = subprocess.run(
        ["bash", "-lc", cmd],
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"command failed ({cp.returncode}): {cmd}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    return cp


def _run_json(cmd: str) -> Dict:
    cp = _run(cmd, check=True)
    out = cp.stdout.strip()
    if not out:
        raise RuntimeError(f"empty json output: {cmd}")
    try:
        return json.loads(out)
    except Exception as e:
        raise RuntimeError(f"json parse failed: {e}\nOUT:\n{out}\nERR:\n{cp.stderr}")


def _docker_exec(container: str, inner_cmd: str, detach: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    dflag = "-d " if detach else ""
    cmd = f"docker exec {dflag}{container} bash -lc {shlex.quote(inner_cmd)}"
    return _run(cmd, check=check)


def _compose_yaml(image: str, code_dir: str, model_dir: str) -> str:
    lines = ["services:"]
    for n in NODES:
        lines.extend(
            [
                f"  {n.name}:",
                f"    image: {image}",
                f"    container_name: {n.name}",
                "    working_dir: /workspace",
                "    command: [\"bash\", \"-lc\", \"sleep infinity\"]",
                "    cap_add:",
                "      - NET_ADMIN",
                "    volumes:",
                f"      - {code_dir}:/workspace",
                f"      - {model_dir}:/models:ro",
                "    networks:",
                "      - rpi-net",
            ]
        )
    lines.extend(
        [
            "networks:",
            "  rpi-net:",
            "    external: true",
        ]
    )
    return "\n".join(lines) + "\n"


def _matrix_bw_round_b(src_kind: str, dst_kind: str) -> float:
    pair = tuple(sorted((src_kind, dst_kind)))
    table = {
        ("NX16", "NX16"): 0.80,
        ("NANO8", "NX16"): 0.50,
        ("MI13", "NX16"): 0.60,
        ("NX16", "RPI4"): 0.12,
        ("NANO8", "NANO8"): 0.40,
        ("MI13", "NANO8"): 0.45,
        ("NANO8", "RPI4"): 0.10,
        ("MI13", "RPI4"): 0.15,
        ("RPI4", "RPI4"): 0.06,
        ("MI13", "MI13"): 0.60,
    }
    return table[pair]


def _links_for_node(node: NodeProfile, round_name: str) -> str:
    specs = []
    for other in NODES:
        if other.node_id == node.node_id:
            continue
        if round_name == "A":
            bw = 0.8
        else:
            bw = _matrix_bw_round_b(node.kind, other.kind)
        specs.append(f"{other.node_id}:{bw:.3f}")
    return ",".join(specs)


def _apply_stable_network(round_name: str) -> None:
    # Coarse egress shaping (pairwise matrix is expressed in init links, while runtime uses per-node egress buckets)
    for n in NODES:
        if round_name == "A":
            rate = "800mbit"
            delay = "5ms"
        else:
            if n.kind == "NX16":
                rate, delay = "800mbit", "5ms"
            elif n.kind == "NANO8":
                rate, delay = "500mbit", "10ms"
            elif n.kind == "MI13":
                rate, delay = "600mbit", "8ms"
            else:
                rate, delay = "180mbit", "18ms"
        _docker_exec(
            n.name,
            f"tc qdisc replace dev eth0 root handle 1: tbf rate {rate} burst 256kbit latency 200ms; "
            f"tc qdisc replace dev eth0 parent 1:1 handle 10: netem delay {delay}",
        )


def _clear_network_limits() -> None:
    for n in NODES:
        _docker_exec(n.name, "tc qdisc del dev eth0 root 2>/dev/null || true", check=False)


def _set_all_cpus(cpus: float) -> None:
    for n in NODES:
        _run(f"docker update --cpus {cpus} {n.name}", check=False)


def _stop_workloads() -> None:
    patt = r"(dllama worker|dllama inference|runtime_optimizer.py|init_root.py)"
    for n in NODES:
        _docker_exec(
            n.name,
            f"ps -eo pid,args | awk '/{patt}/ && !/awk/ {{print $1}}' | xargs -r kill -9",
            check=False,
        )


def _ensure_tools() -> None:
    for n in NODES:
        _docker_exec(
            n.name,
            "which tc >/dev/null 2>&1 || (apt-get update && apt-get install -y iproute2)",
            check=False,
        )


def _wait_for_file(container: str, path: str, timeout_s: int = 120) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        cp = _docker_exec(container, f"test -f {shlex.quote(path)} && echo ok || true", check=False)
        if cp.stdout.strip() == "ok":
            return True
        time.sleep(1)
    return False


def _start_workers(log_dir: str, nthreads: int = 2) -> None:
    for n in NODES:
        if n.node_id == 0:
            continue
        extra = ""
        if n.node_id == 2:
            extra = "export DLLAMA_LAYER_PROF_PATH=/workspace/test_logs/edge8/layer_prof_stage1.bin; "
        _docker_exec(
            n.name,
            f"cd /workspace; {extra}nohup ./dllama worker --port 9999 --nthreads {nthreads} > {log_dir}/{n.name}_worker.log 2>&1 & disown",
            detach=False,
        )


def _wait_socket(socket_path: str, timeout_s: int = 90) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        cp = _docker_exec("ev8_n0", f"test -S {shlex.quote(socket_path)} && echo ok || true", check=False)
        if cp.stdout.strip() == "ok":
            return True
        time.sleep(1)
    return False


def _build_launch_cfg_from_init(init_plan: Dict) -> LaunchCfg:
    stages = {int(s["stage_index"]): s for s in init_plan.get("stages", [])}
    order = [int(x) for x in init_plan.get("pipeline_order", sorted(stages.keys()))]
    if not order:
        raise RuntimeError("empty pipeline_order in init_plan")

    ratios_parts: List[str] = []
    workers: List[str] = []
    seen_workers = set()
    for sid in order:
        st = stages[sid]
        tp = st.get("tp_ratios", [])
        if not tp:
            raise RuntimeError(f"stage {sid} missing tp_ratios")
        nl = int(st.get("n_layers", 0))
        if nl <= 0:
            raise RuntimeError(f"stage {sid} invalid n_layers={nl}")
        ratios_parts.append(":".join(str(int(x)) for x in tp) + f"@{nl}")
        for nid in st.get("node_ids", []):
            nid = int(nid)
            if nid == 0:
                continue
            name = f"ev8_n{nid}:9999"
            if name not in seen_workers:
                workers.append(name)
                seen_workers.add(name)

    if len(workers) != len(NODES) - 1:
        # Fallback to monotonic worker ordering for safety.
        workers = [f"{n.name}:9999" for n in NODES if n.node_id != 0]

    kv = [int(x) for x in init_plan.get("shadow_heads_per_device", [2] * len(NODES))]
    if len(kv) != len(NODES):
        kv = [2] * len(NODES)

    return LaunchCfg(
        ratios="*".join(ratios_parts),
        workers=" ".join(workers),
        kv_redundancy=kv,
        runtime_redundant_boundary_layers=int(init_plan.get("runtime_redundant_boundary_layers", 1)),
    )


def _start_inference(cfg: LaunchCfg, infer_log: str, steps: int, socket_path: str, nthreads: int = 2) -> None:
    cmd = (
        "cd /workspace; "
        f"python3 - <<'PY'\n"
        "import json, shlex, subprocess\n"
        f"workers={json.dumps(cfg.workers)}\n"
        f"ratios={json.dumps(cfg.ratios)}\n"
        f"kv={json.dumps(','.join(str(x) for x in cfg.kv_redundancy))}\n"
        f"rr={int(cfg.runtime_redundant_boundary_layers)}\n"
        f"socket={json.dumps(socket_path)}\n"
        f"infer_log={json.dumps(infer_log)}\n"
        "args=[\n"
        "'./dllama','inference','--benchmark','1',\n"
        "'--model','/models/qwen3_8b_q40/dllama_model_qwen3_8b_q40.m',\n"
        "'--tokenizer','/models/qwen3_8b_q40/dllama_tokenizer_qwen3_8b_q40.t',\n"
        "'--buffer-float-type','q80',\n"
        "'--prompt','hello edgevisor 8dev experiment',\n"
        "'--workers',*workers.split(),\n"
        "'--ratios',ratios,\n"
        "'--enable-plan-barrier',\n"
        "'--enable-stage-full-weights',\n"
        "'--enable-kv-redundancy-during-migration','1',\n"
        "'--enable-pp-migration','1',\n"
        "'--kv-redundancy',kv,\n"
        "'--runtime-redundant-boundary-layers',str(rr),\n"
        f"'--steps','{steps}','--nthreads','{nthreads}'\n"
        "]\n"
        "env=dict(__import__('os').environ)\n"
        "env['DLLAMA_PLAN_CTRL_SOCKET']=socket\n"
        "env['DLLAMA_LAYER_PROF_PATH']='/workspace/test_logs/edge8/layer_prof_stage0.bin'\n"
        "f=open(infer_log,'w')\n"
        "subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env)\n"
        "print('started')\n"
        "PY"
    )
    _docker_exec("ev8_n0", cmd)


def _wait_inference_exit(timeout_s: int = 480) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        cp = _docker_exec(
            "ev8_n0",
            "ps -ef | grep './dllama inference' | grep -v grep | wc -l",
            check=False,
        )
        try:
            n = int(cp.stdout.strip() or "0")
        except Exception:
            n = 0
        if n == 0:
            return True
        time.sleep(2)
    return False


def _infer_failed(log_text: str) -> bool:
    needles = (
        "Execution error:",
        "Critical error:",
        "syncWithRoot ack timeout/fail",
        "Socket closed",
    )
    return any(x in log_text for x in needles)


def _parse_pred_sync(log_text: str) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    for i, line in enumerate(log_text.splitlines(), start=1):
        m = PRED_LINE_RE.search(line)
        if not m:
            continue
        pred = int(m.group(1))
        sync = int(m.group(2))
        pos = int(m.group(3))
        rows.append({"line": i, "pred": pred, "sync": sync, "total": pred + sync, "pos": pos})
    return rows


def _warmup_avg(log_path: str) -> float:
    cp = _docker_exec("ev8_n0", f"cat {shlex.quote(log_path)}", check=False)
    if _infer_failed(cp.stdout):
        return 1e12
    rows = _parse_pred_sync(cp.stdout)
    if not rows:
        return 1e12
    # skip early unstable region if possible
    rows = [r for r in rows if r["pos"] >= 20]
    if len(rows) < 20:
        rows = _parse_pred_sync(cp.stdout)
    vals = [r["total"] for r in rows[:40]]
    return float(sum(vals) / max(1, len(vals)))


def _read_json_in_container(path: str) -> Dict:
    cp = _docker_exec("ev8_n0", f"cat {shlex.quote(path)}")
    return json.loads(cp.stdout)


def _run_init_once(round_name: str, k: int, out_root: str) -> Dict:
    out_dir = f"{out_root}/{round_name}/k{k}"
    _docker_exec("ev8_n0", f"mkdir -p {shlex.quote(out_dir)}")

    init_log = f"{out_dir}/init_root.log"
    cmd = (
        "cd /workspace; "
        f"nohup python3 examples/edgevisor_ext/init_root.py "
        f"--host 0.0.0.0 --port 18080 --expected-nodes 8 --timeout-s 60 "
        f"--rragc-k {k} "
        "--total-layers 36 --activation-size-gb 0.1 --layer-total-flops 1e12 "
        "--layer-input-bytes 5e7 --layer-output-bytes 5e7 "
        f"--output-dir {shlex.quote(out_dir)} > {shlex.quote(init_log)} 2>&1 &"
    )
    _docker_exec("ev8_n0", cmd)

    time.sleep(2)
    for n in NODES:
        links = _links_for_node(n, round_name)
        mem_bytes = int(n.memory_gb * 1e9)
        agent_cmd = (
            "cd /workspace; "
            "python3 examples/edgevisor_ext/init_agent.py "
            f"--node-id {n.node_id} "
            "--root-url http://ev8_n0:18080/report "
            f"--compute-flops {n.compute_flops} "
            f"--memory-bytes {mem_bytes} "
            f"--host {n.name} "
            f"--links {shlex.quote(links)}"
        )
        _docker_exec(n.name, agent_cmd)

    init_path = f"{out_dir}/init_plan.json"
    launch_path = f"{out_dir}/launch_plan.json"
    ok = _wait_for_file("ev8_n0", launch_path, timeout_s=120) and _wait_for_file("ev8_n0", init_path, timeout_s=120)
    if not ok:
        cp = _docker_exec("ev8_n0", f"tail -n 120 {shlex.quote(init_log)}", check=False)
        raise RuntimeError(f"init_root did not output launch plan for round={round_name}, k={k}\n{cp.stdout}")

    warmup_log = f"{out_dir}/infer_warmup.log"
    _stop_workloads()
    init_plan = _read_json_in_container(init_path)
    launch_cfg = _build_launch_cfg_from_init(init_plan)
    _start_workers(out_dir, nthreads=2)
    _start_inference(launch_cfg, warmup_log, steps=60, socket_path="/tmp/dllama_plan.sock", nthreads=2)
    done = _wait_inference_exit(timeout_s=240)
    if not done:
        _stop_workloads()

    avg = _warmup_avg(warmup_log)
    plan = _read_json_in_container(launch_path)
    return {
        "round": round_name,
        "k": k,
        "warmup_avg_ms": avg,
        "init_plan_path": init_path,
        "init_plan": init_plan,
        "launch_cfg": dataclasses.asdict(launch_cfg),
        "launch_plan_path": launch_path,
        "launch_plan": plan,
        "out_dir": out_dir,
    }


def _choose_best_k(round_name: str, out_root: str) -> Dict:
    results = []
    for k in K_CANDIDATES:
        results.append(_run_init_once(round_name, k, out_root))
    results.sort(key=lambda x: x["warmup_avg_ms"])
    return {"selected": results[0], "all": results}


def _uds(op: str, extra_args: str = "") -> Dict:
    cmd = (
        "cd /workspace; "
        "python3 examples/plan-uds-client.py /tmp/dllama_plan.sock "
        f"{op} {extra_args}"
    )
    cp = _docker_exec("ev8_n0", cmd, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"uds call failed: {op} {extra_args}\n{cp.stdout}\n{cp.stderr}")
    return json.loads(cp.stdout)


def _wait_last_apply_seq(plan_seq: int, timeout_s: int = 90) -> Optional[Dict]:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            resp = _uds("last_apply")
            la = resp.get("last_apply", {})
            if isinstance(la, dict) and int(la.get("planSeq", -1)) == int(plan_seq):
                return la
        except Exception:
            pass
        time.sleep(1)
    return None


def _last_apply_anchor(min_seq: int, infer_log: str) -> Optional[Tuple[int, int]]:
    try:
        resp = _uds("last_apply")
        la = resp.get("last_apply", {})
        if not isinstance(la, dict):
            return None
        seq = int(la.get("planSeq", -1))
        pos = int(la.get("position", -1))
        if seq >= min_seq and pos >= 0:
            line, _ = _current_infer_line_and_pos(infer_log)
            return line, pos
    except Exception:
        return None
    return None


def _current_infer_line_and_pos(infer_log: str) -> Tuple[int, int]:
    cp = _docker_exec(
        "ev8_n0",
        (
            "python3 - <<'PY'\n"
            "import re\n"
            f"p={json.dumps(infer_log)}\n"
            "txt=open(p,'r',errors='ignore').read().splitlines() if __import__('os').path.exists(p) else []\n"
            "line=len(txt)\n"
            "pos=-1\n"
            "for s in reversed(txt):\n"
            " m=re.search(r'pos=(\\d+)',s)\n"
            " if m:\n"
            "  pos=int(m.group(1));break\n"
            "print(f'{line} {pos}')\n"
            "PY"
        ),
        check=False,
    )
    parts = (cp.stdout.strip() or "0 -1").split()
    return int(parts[0]), int(parts[1])


def _wait_min_pos(infer_log: str, min_pos: int, timeout_s: int = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        _line, pos = _current_infer_line_and_pos(infer_log)
        if pos >= min_pos:
            return True
        time.sleep(2)
    return False


def _read_runtime_events(runtime_log: str) -> List[Dict]:
    cp = _docker_exec("ev8_n0", f"cat {shlex.quote(runtime_log)}", check=False)
    events = []
    for line in cp.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def _max_runtime_step(runtime_log: str) -> int:
    mx = -1
    for ev in _read_runtime_events(runtime_log):
        try:
            mx = max(mx, int(ev.get("step", -1)))
        except Exception:
            continue
    return mx


def _detect_command_sent(runtime_log: str, kind: str, since_step: int = 0) -> Optional[Dict]:
    for ev in _read_runtime_events(runtime_log):
        if ev.get("event") != "command_sent":
            continue
        if int(ev.get("step", 0)) < since_step:
            continue
        decision = ev.get("decision", {})
        if decision.get("kind") == kind:
            return ev
    return None


def _find_apply_line(infer_log: str, kind: str, min_line: int = 0) -> Optional[int]:
    if kind == "layer":
        needle = "apply pp command"
    else:
        needle = "[plan][apply]"
    cp = _docker_exec(
        "ev8_n0",
        (
            "python3 - <<'PY'\n"
            "import json\n"
            f"p={json.dumps(infer_log)}\n"
            f"needle={json.dumps(needle)}\n"
            f"min_line={int(min_line)}\n"
            "lines=open(p,'r',errors='ignore').read().splitlines() if __import__('os').path.exists(p) else []\n"
            "ans=-1\n"
            "for i,s in enumerate(lines, start=1):\n"
            " if i<=min_line: continue\n"
            " if needle in s:\n"
            "  if needle=='[plan][apply]' and 'cmdlist' not in s:\n"
            "   continue\n"
            "  ans=i; break\n"
            "print(ans)\n"
            "PY"
        ),
        check=False,
    )
    try:
        v = int(cp.stdout.strip())
    except Exception:
        v = -1
    return None if v < 0 else v


def _compute_bottleneck_and_target() -> Tuple[int, int, int]:
    perf = _uds("perf")
    snap = _uds("plan_snapshot")
    stage_map = {int(s["stageIndex"]): s for s in snap["plan_snapshot"].get("stages", [])}
    stage_times: Dict[int, float] = {}
    for item in perf.get("perf", []):
        try:
            sid = int(item.get("stageIndex", 0))
            t = float(item.get("execUs", 0)) + float(item.get("syncUs", 0))
        except Exception:
            continue
        if sid not in stage_map:
            continue
        stage_times[sid] = max(stage_times.get(sid, 0.0), t)
    if not stage_times:
        raise RuntimeError("no stage times")
    bottleneck = max(stage_times, key=lambda x: stage_times[x])
    cands = [sid for sid in (bottleneck - 1, bottleneck + 1) if sid in stage_times and sid in stage_map]
    if not cands:
        raise RuntimeError("no adjacent target stage")
    target = min(cands, key=lambda sid: stage_times[sid])
    from_node = int(stage_map[bottleneck]["rootNodeIndex"])
    to_node = int(stage_map[target]["rootNodeIndex"])
    return bottleneck, from_node, to_node


def _pick_small_head_victim(plan_snapshot: Dict) -> Optional[Dict[str, int]]:
    stages = plan_snapshot.get("stages", [])
    splits = plan_snapshot.get("splits", {})
    kvc = splits.get("kvHeadComputeSplit", {}).get("lengths")
    kv = splits.get("kvHeadSplit", {}).get("lengths")
    lens = kvc if isinstance(kvc, list) and kvc else kv
    if not isinstance(lens, list) or not lens:
        return None

    best = None
    for st in stages:
        if not isinstance(st, dict):
            continue
        stage = int(st.get("stageIndex", -1))
        root = int(st.get("rootNodeIndex", -1))
        nodes = [int(x) for x in st.get("nodes", [])]
        if len(nodes) < 2:
            continue
        node_heads = [(n, int(lens[n]) if 0 <= n < len(lens) else 0) for n in nodes]
        node_heads = [(n, h) for n, h in node_heads if h > 0]
        if len(node_heads) < 2:
            continue
        # Prefer degrading a non-root worker with larger share in a 2-node stage
        # so "drain all heads" can yield observable recovery.
        non_root = [(n, h) for n, h in node_heads if n != root]
        if non_root:
            src, src_h = max(non_root, key=lambda x: x[1])
        else:
            src, src_h = min(node_heads, key=lambda x: x[1])
        src_idx = nodes.index(src)
        neigh_idxs = [i for i in (src_idx - 1, src_idx + 1) if 0 <= i < len(nodes)]
        if not neigh_idxs:
            continue
        dst = max((nodes[i] for i in neigh_idxs), key=lambda n: int(lens[n]) if 0 <= n < len(lens) else 0)
        dst_h = int(lens[dst]) if 0 <= dst < len(lens) else 0
        score = (0 if len(nodes) == 2 else 1, -src_h, stage)
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "stage": stage,
                "src": src,
                "dst": dst,
                "src_heads": src_h,
                "dst_heads": dst_h,
            }
    if best is None:
        return None
    best.pop("score", None)
    return best


def _pick_victim_by_node(plan_snapshot: Dict, node_id: int) -> Optional[Dict[str, int]]:
    stages = plan_snapshot.get("stages", [])
    splits = plan_snapshot.get("splits", {})
    kvc = splits.get("kvHeadComputeSplit", {}).get("lengths")
    kv = splits.get("kvHeadSplit", {}).get("lengths")
    lens = kvc if isinstance(kvc, list) and kvc else kv
    if not isinstance(lens, list) or node_id < 0 or node_id >= len(lens):
        return None
    src_h = int(lens[node_id])
    if src_h <= 0:
        return None
    for st in stages:
        if not isinstance(st, dict):
            continue
        nodes = [int(x) for x in st.get("nodes", [])]
        if node_id not in nodes or len(nodes) < 2:
            continue
        idx = nodes.index(node_id)
        neigh_idxs = [i for i in (idx - 1, idx + 1) if 0 <= i < len(nodes)]
        if not neigh_idxs:
            return None
        dst = max((nodes[i] for i in neigh_idxs), key=lambda n: int(lens[n]) if 0 <= n < len(lens) else 0)
        return {
            "stage": int(st.get("stageIndex", 0)),
            "src": node_id,
            "dst": int(dst),
            "src_heads": src_h,
            "dst_heads": int(lens[dst]) if 0 <= dst < len(lens) else 0,
        }
    return None


def _manual_layer_fallback(seq: int) -> Tuple[int, int, int, int]:
    bottleneck, from_node, to_node = _compute_bottleneck_and_target()
    last_err = None
    for cnt in (3, 2, 1):
        try:
            _uds(
                "set_pp_migration",
                f"--seq {seq} --mode next_barrier --from {from_node} --to {to_node} --layer-count {cnt}",
            )
            return bottleneck, from_node, to_node, cnt
        except Exception as e:  # noqa: PERF203
            last_err = e
    raise RuntimeError(f"layer fallback failed: {last_err}")


def _manual_layer_fallback_for_stage(seq: int, preferred_stage: int) -> Tuple[int, int, int, int]:
    perf = _uds("perf")
    snap = _uds("plan_snapshot")
    stage_map = {int(s["stageIndex"]): s for s in snap["plan_snapshot"].get("stages", [])}
    stage_times: Dict[int, float] = {}
    for item in perf.get("perf", []):
        try:
            sid = int(item.get("stageIndex", 0))
            t = float(item.get("execUs", 0)) + float(item.get("syncUs", 0))
        except Exception:
            continue
        if sid in stage_map:
            stage_times[sid] = max(stage_times.get(sid, 0.0), t)

    if preferred_stage not in stage_map:
        return _manual_layer_fallback(seq)
    cands = [sid for sid in (preferred_stage - 1, preferred_stage + 1) if sid in stage_map]
    if not cands:
        return _manual_layer_fallback(seq)
    target = min(cands, key=lambda sid: stage_times.get(sid, float("inf")))
    from_node = int(stage_map[preferred_stage]["rootNodeIndex"])
    to_node = int(stage_map[target]["rootNodeIndex"])
    last_err = None
    for cnt in (1,):
        try:
            _uds(
                "set_pp_migration",
                f"--seq {seq} --mode next_barrier --from {from_node} --to {to_node} --layer-count {cnt}",
            )
            return preferred_stage, from_node, to_node, cnt
        except Exception as e:  # noqa: PERF203
            last_err = e
    raise RuntimeError(f"layer targeted fallback failed: {last_err}")


def _manual_token_fallback(seq: int) -> Tuple[int, int, int, int]:
    perf = _uds("perf")
    snap = _uds("plan_snapshot")
    stage_map = {int(s["stageIndex"]): s for s in snap["plan_snapshot"].get("stages", [])}

    stage_times: Dict[int, float] = {}
    for item in perf.get("perf", []):
        try:
            sid = int(item.get("stageIndex", 0))
            t = float(item.get("execUs", 0)) + float(item.get("syncUs", 0))
        except Exception:
            continue
        if sid not in stage_map:
            continue
        stage_times[sid] = max(stage_times.get(sid, 0.0), t)
    if not stage_times:
        raise RuntimeError("no stage times for token fallback")

    multi_node_stages = []
    for sid, st in stage_map.items():
        nodes = [int(x) for x in st.get("nodes", [])]
        if len(nodes) >= 2 and sid in stage_times:
            multi_node_stages.append((sid, nodes))
    if not multi_node_stages:
        raise RuntimeError("no multi-node stage for token fallback")
    stage = max((sid for sid, _nodes in multi_node_stages), key=lambda sid: stage_times.get(sid, 0.0))
    st = stage_map[stage]
    nodes = [int(x) for x in st.get("nodes", [])]

    # Use lightweight heuristic: move heads from slowest node to adjacent fastest node.
    lp = _uds("layer_prof", f"--all --stage {stage} --root {int(st['rootNodeIndex'])}")
    layers = lp.get("layer_prof", {}).get("layers", [])
    score_by_node: Dict[int, int] = {n: 0 for n in nodes}
    for row in layers:
        if not isinstance(row, list):
            continue
        for cell in row:
            if not isinstance(cell, dict) or not cell.get("ok"):
                continue
            nid = int(cell.get("nodeIndex", -1))
            if nid in score_by_node:
                score_by_node[nid] = max(score_by_node[nid], int(cell.get("attnUs", 0)) + int(cell.get("ffnUs", 0)))

    src = max(nodes, key=lambda n: score_by_node.get(n, 0))
    src_idx = nodes.index(src)
    neighbor_idxs = [i for i in (src_idx - 1, src_idx + 1) if 0 <= i < len(nodes)]
    if not neighbor_idxs:
        raise RuntimeError("no neighbor for token fallback")
    dst = min((nodes[i] for i in neighbor_idxs), key=lambda n: score_by_node.get(n, 0))

    last_err = None
    for i, amt in enumerate((8, 6, 4, 2, 1)):
        cur_seq = seq + i
        try:
            _uds(
                "set_plan",
                f"--seq {cur_seq} --mode next_barrier --stage {stage} --move {src},{dst},1,{amt},0",
            )
            la = _wait_last_apply_seq(cur_seq, timeout_s=90)
            if isinstance(la, dict) and bool(la.get("ok")):
                return stage, src, dst, amt
            if isinstance(la, dict):
                last_err = RuntimeError(f"apply reject: {la.get('reason', 'unknown')}")
            else:
                last_err = RuntimeError("apply timeout")
        except Exception as e:  # noqa: PERF203
            last_err = e
    raise RuntimeError(f"token fallback failed: {last_err}")


def _manual_token_drain_fallback(seq: int, stage: int, src: int, dst: int, src_heads: int) -> Tuple[int, int, int, int]:
    last_err = None
    seq_offset = 0
    tried = []
    for amt in (src_heads, max(1, src_heads - 1), max(1, src_heads // 2), 1):
        if amt in tried:
            continue
        tried.append(amt)
        cur_seq = seq + seq_offset
        seq_offset += 1
        try:
            _uds(
                "set_plan",
                f"--seq {cur_seq} --mode next_barrier --stage {stage} --move {src},{dst},1,{amt},0",
            )
            la = _wait_last_apply_seq(cur_seq, timeout_s=90)
            if isinstance(la, dict) and bool(la.get("ok")):
                return stage, src, dst, amt
            if isinstance(la, dict):
                last_err = RuntimeError(f"apply reject: {la.get('reason', 'unknown')}")
            else:
                last_err = RuntimeError("apply timeout")
        except Exception as e:  # noqa: PERF203
            last_err = e
    raise RuntimeError(f"token drain fallback failed: {last_err}")


def _window_avg_by_pos(rows: List[Dict[str, int]], apply_pos: int, window: int = 40, skip: int = 5) -> Tuple[float, float, int, int]:
    by_pos: Dict[int, int] = {}
    for r in rows:
        by_pos[int(r["pos"])] = int(r["total"])
    pre_vals = [by_pos[p] for p in range(apply_pos - window, apply_pos) if p in by_pos]
    post_vals = [by_pos[p] for p in range(apply_pos + skip, apply_pos + skip + window) if p in by_pos]
    if len(pre_vals) < window or len(post_vals) < window:
        raise RuntimeError(
            f"insufficient rows for {window}-token window at apply_pos={apply_pos}: "
            f"pre={len(pre_vals)} post={len(post_vals)}"
        )
    pre_avg = statistics.mean(pre_vals)
    post_avg = statistics.mean(post_vals)
    return pre_avg, post_avg, apply_pos - 1, apply_pos + skip + window - 1


def _analyze(infer_log: str, layer_apply_pos: int, token_apply_pos: int) -> Dict:
    cp = _docker_exec("ev8_n0", f"cat {shlex.quote(infer_log)}")
    txt = cp.stdout
    rows = _parse_pred_sync(txt)
    if not rows:
        raise RuntimeError("no pred/sync lines in infer log")
    layer_pre, layer_post, layer_pre_pos_end, layer_post_pos_end = _window_avg_by_pos(rows, layer_apply_pos)
    token_pre, token_post, token_pre_pos_end, token_post_pos_end = _window_avg_by_pos(rows, token_apply_pos)
    last_pos = rows[-1]["pos"]
    return {
        "layer": {
            "pre_avg_ms": layer_pre,
            "post_avg_ms": layer_post,
            "improve_ratio": (layer_pre - layer_post) / max(1e-9, layer_pre),
            "pre_window_end_pos": layer_pre_pos_end,
            "post_window_end_pos": layer_post_pos_end,
        },
        "token": {
            "pre_avg_ms": token_pre,
            "post_avg_ms": token_post,
            "improve_ratio": (token_pre - token_post) / max(1e-9, token_pre),
            "pre_window_end_pos": token_pre_pos_end,
            "post_window_end_pos": token_post_pos_end,
        },
        "last_pos": last_pos,
    }


def _write_report(
    report_path: str,
    round_a: Dict,
    round_b: Dict,
    events: List[Dict],
    layer_mode: str,
    token_mode: str,
    layer_apply_line: int,
    token_apply_line: int,
    metrics: Dict,
    threshold: float,
) -> None:
    layer_pass = metrics["layer"]["improve_ratio"] >= threshold
    token_pass = metrics["token"]["improve_ratio"] >= threshold

    lines = []
    lines.append("# Phase1 8-Device Remote Experiment Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Round A selected K: {round_a['selected']['k']} (warmup avg {round_a['selected']['warmup_avg_ms']:.2f} ms)")
    lines.append(f"- Round B selected K: {round_b['selected']['k']} (warmup avg {round_b['selected']['warmup_avg_ms']:.2f} ms)")
    lines.append(f"- Layer trigger source: {layer_mode}")
    lines.append(f"- Token trigger source: {token_mode}")
    lines.append(f"- Improvement threshold: {threshold*100:.1f}%")
    lines.append("")
    lines.append("## K Trial Details")
    for rname, sel in (("A", round_a), ("B", round_b)):
        lines.append(f"### Round {rname}")
        for item in sel["all"]:
            lines.append(f"- K={item['k']}: warmup_avg_ms={item['warmup_avg_ms']:.2f}, launch_plan={item['launch_plan_path']}")
        lines.append("")

    lines.append("## Disturbance + Optimization Events")
    for e in events:
        lines.append(f"- {e['ts']}: {e['name']} (line={e['line']}, pos={e['pos']})")
    lines.append("")

    lines.append("## Apply Anchors")
    lines.append(f"- Layer apply line: {layer_apply_line}")
    lines.append(f"- Token apply line: {token_apply_line}")
    lines.append("")

    lines.append("## Metrics (Pred+Sync avg, 40-token windows, skip first 5 post-apply)")
    lines.append(
        f"- Layer: pre={metrics['layer']['pre_avg_ms']:.2f} ms, post={metrics['layer']['post_avg_ms']:.2f} ms, "
        f"improve={metrics['layer']['improve_ratio']*100:.2f}% -> {'PASS' if layer_pass else 'FAIL'}"
    )
    lines.append(
        f"- Token: pre={metrics['token']['pre_avg_ms']:.2f} ms, post={metrics['token']['post_avg_ms']:.2f} ms, "
        f"improve={metrics['token']['improve_ratio']*100:.2f}% -> {'PASS' if token_pass else 'FAIL'}"
    )
    lines.append(f"- Last generated token position: {metrics['last_pos']}")
    lines.append("")

    lines.append("## Acceptance")
    lines.append(f"- Layer >= {threshold*100:.1f}%: {'PASS' if layer_pass else 'FAIL'}")
    lines.append(f"- Token >= {threshold*100:.1f}%: {'PASS' if token_pass else 'FAIL'}")
    lines.append(f"- Stability (last_pos >= token_apply_anchor+300): {'PASS' if metrics['last_pos'] >= metrics['token']['post_window_end_pos'] + 255 else 'CHECK'}")

    _docker_exec("ev8_n0", f"cat > {shlex.quote(report_path)} <<'EOF'\n" + "\n".join(lines) + "\nEOF")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 8-device heterogeneous docker experiment on remote host")
    ap.add_argument("--compose-path", default="/home/cc/yhbian/docker_file/docker-compose-edgevisor-8dev.yml")
    ap.add_argument("--code-dir", default="/home/cc/yhbian/edgevisor_code")
    ap.add_argument("--model-dir", default="/home/cc/dllama/distributed-llama/models")
    ap.add_argument("--image", default="cc/dllama:dev")
    ap.add_argument("--out-root", default="/workspace/test_logs/edge8")
    ap.add_argument("--report-path", default="/workspace/examples/edgevisor_ext/phase1_8dev_remote_report.md")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--max-auto-wait", type=int, default=45)
    args = ap.parse_args()

    print("[1/9] write compose and start containers")
    compose = _compose_yaml(args.image, args.code_dir, args.model_dir)
    _run(f"cat > {shlex.quote(args.compose_path)} <<'EOF'\n{compose}EOF")
    _run(f"docker compose -f {shlex.quote(args.compose_path)} up -d --remove-orphans")

    print("[2/9] ensure tools / reset state")
    _ensure_tools()
    _clear_network_limits()
    _set_all_cpus(4.0)
    _stop_workloads()

    print("[3/9] round A init + K trial")
    _apply_stable_network("A")
    round_a = _choose_best_k("A", args.out_root)

    print("[4/9] round B init + K trial")
    _stop_workloads()
    _apply_stable_network("B")
    round_b = _choose_best_k("B", args.out_root)

    selected = round_b["selected"]
    launch_cfg = LaunchCfg(**selected["launch_cfg"])
    exp_dir = f"{selected['out_dir']}/main"
    _docker_exec("ev8_n0", f"mkdir -p {shlex.quote(exp_dir)}")

    print("[5/9] start long inference")
    _stop_workloads()
    _start_workers(exp_dir, nthreads=2)
    infer_log = f"{exp_dir}/infer_main.log"
    runtime_log = f"{exp_dir}/runtime_optimizer.log"
    _start_inference(launch_cfg, infer_log, steps=1600, socket_path="/tmp/dllama_plan.sock", nthreads=2)
    if not _wait_socket("/tmp/dllama_plan.sock", timeout_s=120):
        raise RuntimeError("UDS socket not ready")

    # precheck UDS
    _uds("ping")
    _uds("status")
    _uds("plan_snapshot")
    _uds("last_apply")

    events: List[Dict] = []

    def mark(name: str) -> Tuple[int, int]:
        line, pos = _current_infer_line_and_pos(infer_log)
        e = {"name": name, "line": line, "pos": pos, "ts": int(time.time() * 1000)}
        events.append(e)
        print(f"[event] {name}: line={line}, pos={pos}")
        return line, pos

    # baseline phase: ensure enough pre-window tokens before first injection.
    _wait_min_pos(infer_log, min_pos=60, timeout_s=420)
    mark("baseline_ready")

    print("[6/9] inject compute disturbance (persistent)")
    snap0 = _uds("plan_snapshot")["plan_snapshot"]
    victim = _pick_small_head_victim(snap0)
    if victim is not None:
        target_node = int(victim["src"])
    else:
        _bottleneck_stage, from_node, _to_node = _compute_bottleneck_and_target()
        target_node = from_node
    target_name = next(n.name for n in NODES if n.node_id == target_node)
    _run(f"docker update --cpus 0.2 {target_name}")
    if victim is not None:
        mark(
            f"inject_compute_node{target_node}_stage{victim['stage']}_"
            f"heads{victim['src_heads']}_to{victim['dst']}"
        )
    else:
        mark(f"inject_compute_node{target_node}")

    print("[6.5/9] manual-only mode (runtime optimizer disabled)")
    _docker_exec("ev8_n0", "pkill -f '[r]untime_optimizer.py' || true", check=False)
    _docker_exec("ev8_n0", f": > {shlex.quote(runtime_log)}")

    print("[7/9] manual layer command")
    layer_mode = "FALLBACK"
    if victim is not None:
        bottleneck_stage, ff, tt, cnt = _manual_layer_fallback_for_stage(seq=7001, preferred_stage=int(victim["stage"]))
    else:
        bottleneck_stage, ff, tt, cnt = _manual_layer_fallback(seq=7001)
    mark(f"layer_cmd_fallback_stage{bottleneck_stage}_{ff}_to_{tt}_layers{cnt}")

    layer_apply_line = None
    start_line = events[-1]["line"]
    t0 = time.time()
    while time.time() - t0 < 180:
        layer_apply_line = _find_apply_line(infer_log, "layer", min_line=start_line)
        if layer_apply_line is not None:
            break
        time.sleep(2)
    if layer_apply_line is None:
        anchor = _last_apply_anchor(min_seq=7001, infer_log=infer_log)
        if anchor is None:
            raise RuntimeError("layer apply not observed")
        layer_apply_line, layer_apply_pos = anchor
        mark(f"layer_apply_fallback_line{layer_apply_line}")
    else:
        _layer_line, layer_apply_pos = mark(f"layer_apply_line{layer_apply_line}")

    # Keep a clean layer-only observation window before adding network disturbance.
    _docker_exec("ev8_n0", "pkill -f '[r]untime_optimizer.py' || true", check=False)
    _wait_min_pos(infer_log, min_pos=layer_apply_pos + 50, timeout_s=360)

    print("[8/9] inject network disturbance (persistent overlay)")
    _docker_exec(
        target_name,
        "tc qdisc replace dev eth0 root handle 1: tbf rate 20mbit burst 32kbit latency 500ms; "
        "tc qdisc replace dev eth0 parent 1:1 handle 10: netem delay 60ms 15ms loss 1.0%",
    )
    _net_line, net_inject_pos = mark(f"inject_network_node{target_node}")

    # Build degraded token pre-window before token optimization commands.
    _wait_min_pos(infer_log, min_pos=net_inject_pos + 45, timeout_s=360)
    _docker_exec("ev8_n0", f": > {shlex.quote(runtime_log)}")
    _docker_exec(
        "ev8_n0",
        "cd /workspace; "
        f"nohup python3 -u examples/edgevisor_ext/runtime_optimizer.py /tmp/dllama_plan.sock "
        "--poll-ms 200 --min-improve-ratio 0.99 --layer-no-improve-epochs 1 --cooldown-steps 2 "
        f"> {shlex.quote(runtime_log)} 2>&1 &",
    )
    time.sleep(3)

    print("[9/9] manual token command, then analyze/report")
    # Stop runtime optimizer before manual fallback command to avoid command races.
    _docker_exec("ev8_n0", "pkill -f '[r]untime_optimizer.py' || true", check=False)
    time.sleep(1)
    token_mode = "FALLBACK"
    if victim is not None:
        snap_now = _uds("plan_snapshot")["plan_snapshot"]
        victim_now = _pick_victim_by_node(snap_now, int(victim["src"]))
        chosen = victim_now if victim_now is not None else victim
        stg, src, dst, amt = _manual_token_drain_fallback(
            seq=7002,
            stage=int(chosen["stage"]),
            src=int(chosen["src"]),
            dst=int(chosen["dst"]),
            src_heads=int(chosen["src_heads"]),
        )
    else:
        stg, src, dst, amt = _manual_token_fallback(seq=7002)
    mark(f"token_cmd_fallback_stage{stg}_{src}_to_{dst}_heads{amt}")

    token_apply_line = None
    start_line = events[-1]["line"]
    t1 = time.time()
    while time.time() - t1 < 180:
        token_apply_line = _find_apply_line(infer_log, "token", min_line=start_line)
        if token_apply_line is not None:
            break
        time.sleep(2)
    if token_apply_line is None:
        anchor = _last_apply_anchor(min_seq=7002, infer_log=infer_log)
        if anchor is None:
            raise RuntimeError("token apply not observed")
        token_apply_line, token_apply_pos = anchor
        mark(f"token_apply_fallback_line{token_apply_line}")
    else:
        _token_line, token_apply_pos = mark(f"token_apply_line{token_apply_line}")
    _docker_exec("ev8_n0", "pkill -f '[r]untime_optimizer.py' || true", check=False)

    # keep running for stability >=300 tokens after token apply anchor
    print("[stability] waiting for >=300 token progress after token apply")
    t2 = time.time()
    while time.time() - t2 < 900:
        _line, last = _current_infer_line_and_pos(infer_log)
        if last - token_apply_pos >= 300:
            break
        time.sleep(3)

    metrics = _analyze(infer_log, layer_apply_pos, token_apply_pos)

    _write_report(
        report_path=args.report_path,
        round_a=round_a,
        round_b=round_b,
        events=events,
        layer_mode=layer_mode,
        token_mode=token_mode,
        layer_apply_line=layer_apply_line,
        token_apply_line=token_apply_line,
        metrics=metrics,
        threshold=args.threshold,
    )

    print("Experiment finished.")
    print(json.dumps({
        "report_path": args.report_path,
        "layer_mode": layer_mode,
        "token_mode": token_mode,
        "layer_improve": metrics["layer"]["improve_ratio"],
        "token_improve": metrics["token"]["improve_ratio"],
        "last_pos": metrics["last_pos"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

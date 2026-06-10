from __future__ import annotations

import json
import os
import random
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(os.environ.get("B01_ROOT", "/home/byh/B01"))
EDGE_PROJECT = ROOT / "EdgeVisor"
EDGE_ENGINE = EDGE_PROJECT / "EdgeVisor"
EDGE_DLLAMA = EDGE_ENGINE / "dllama"
EDGE_MODEL = ROOT / "models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
EDGE_TOKENIZER = ROOT / "models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"

PRIMA_DIR = ROOT / "prima_cpp_work/prima.cpp"
PRIMA_CLI = PRIMA_DIR / "llama-cli"
PRIMA_MODEL = ROOT / "models/gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf"


@dataclass
class GenerationResult:
    backend: str
    content: str
    metrics: Dict[str, Any]
    command: List[str] = field(default_factory=list)
    log_path: Optional[str] = None
    dynamic_events: List[Dict[str, Any]] = field(default_factory=list)
    rc: int = 0


class BackendError(RuntimeError):
    pass


def render_messages(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def ensure_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise BackendError("missing required files:\n" + "\n".join(missing))


def base_env(cuda_visible: str = "0,1,2") -> Dict[str, str]:
    env = os.environ.copy()
    vulkan_bin = EDGE_PROJECT / "tools/vulkan_deps/root/usr/bin"
    vulkan_lib = EDGE_PROJECT / "tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu"
    env["PATH"] = f"{vulkan_bin}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{vulkan_lib}:/usr/local/cuda/lib64:/usr/local/lib:"
        f"{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    return env


def parse_prima_metrics(log_text: str) -> Dict[str, Any]:
    prompt_match = re.search(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens", log_text)
    eval_match = re.search(r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs", log_text)
    load_match = re.search(r"load time\s*=\s*([0-9.]+)\s*ms", log_text)
    total_match = re.search(r"total time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens", log_text)
    prompt_ms = float(prompt_match.group(1)) if prompt_match else None
    prompt_tokens = int(prompt_match.group(2)) if prompt_match else None
    decode_ms = float(eval_match.group(1)) if eval_match else None
    decode_runs = int(eval_match.group(2)) if eval_match else None
    return {
        "load_ms": float(load_match.group(1)) if load_match else None,
        "ttft_ms": prompt_ms,
        "prompt_eval_ms": prompt_ms,
        "prompt_tokens": prompt_tokens,
        "decode_eval_ms": decode_ms,
        "decode_runs": decode_runs,
        "tpot_ms_after_first": (decode_ms / decode_runs) if decode_ms is not None and decode_runs else None,
        "total_ms": float(total_match.group(1)) if total_match else None,
        "total_tokens": int(total_match.group(2)) if total_match else None,
    }


def parse_edge_metrics(log_text: str) -> Dict[str, Any]:
    eval_parts = [(int(a), int(b)) for a, b in re.findall(r"Eval\s+(\d+)\s+ms Sync\s+(\d+)\s+ms", log_text)]
    pred_parts = [(int(a), int(b)) for a, b in re.findall(r"Pred\s+(\d+)\s+ms Sync\s+(\d+)\s+ms", log_text)]
    eval_ms = sum(a + b for a, b in eval_parts)
    pred_ms = sum(a + b for a, b in pred_parts)
    first_pred_ms = pred_parts[0][0] + pred_parts[0][1] if pred_parts else None
    pred_match = re.search(
        r"Prediction\s+.*?nTokens:\s+(\d+)\s+.*?tokens/s:\s+([0-9.]+)\s+\(([0-9.]+)\s+ms/tok\)",
        log_text,
        re.S,
    )
    eval_match = re.search(
        r"Evaluation\s+.*?nTokens:\s+(\d+)\s+.*?tokens/s:\s+([0-9.]+)\s+\(([0-9.]+)\s+ms/tok\)",
        log_text,
        re.S,
    )
    n_pred = int(pred_match.group(1)) if pred_match else len(pred_parts)
    tpot = float(pred_match.group(3)) if pred_match else (pred_ms / n_pred if n_pred else None)
    tpot_after_first = None
    if len(pred_parts) > 1 and first_pred_ms is not None:
        tpot_after_first = (pred_ms - first_pred_ms) / (len(pred_parts) - 1)
    return {
        "eval_ms": eval_ms,
        "first_pred_ms": first_pred_ms,
        "ttft_ms": eval_ms + first_pred_ms if first_pred_ms is not None else None,
        "prediction_ms_total": pred_ms,
        "prediction_tokens": n_pred,
        "eval_tokens": int(eval_match.group(1)) if eval_match else None,
        "tpot_ms_all_pred": tpot,
        "tpot_ms_after_first": tpot_after_first,
        "plan_apply_seen": "[plan][apply]" in log_text,
        "plan_emit_seen": "[plan][emit]" in log_text,
    }


def strip_prima_content(log_text: str) -> str:
    match = re.search(r"generate:.*?\n\n(?P<content>.*?)(?:\nllama_perf_|llama_perf_|$)", log_text, re.S)
    if match:
        return match.group("content").strip()
    return re.split(r"\nllama_perf_|llama_perf_", log_text, maxsplit=1)[0].strip()[-1200:]


def strip_edge_content(log_text: str) -> str:
    pieces: List[str] = []
    for line in log_text.splitlines():
        if "Pred" in line and "Sync" in line and "|" in line:
            pieces.append(line.rsplit("|", 1)[-1])
    content = "".join(pieces).strip()
    if content:
        return content[-1200:]
    return "[edgevisor generation completed; generated text was not extractable from benchmark log]"


def estimate_edge_steps(prompt: str, max_tokens: int, ctx: int, floor: int) -> int:
    prompt_estimate = max(len(re.findall(r"\S+", prompt)) * 2, len(prompt) // 3)
    requested = max(floor, prompt_estimate + max_tokens + 64)
    return min(max(1, ctx - 8), requested)


def terminate_all(procs: Iterable[subprocess.Popen[Any]]) -> None:
    live = [p for p in procs if p and p.poll() is None]
    for proc in live:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    deadline = time.time() + 5
    for proc in live:
        if proc.poll() is None:
            try:
                proc.wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()


def popen_log(cmd: List[str], log_path: Path, cwd: Path, env: Dict[str, str]) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._agent_log_f = log_f  # type: ignore[attr-defined]
    return proc


def close_logs(procs: Iterable[subprocess.Popen[Any]]) -> None:
    for proc in procs:
        log_f = getattr(proc, "_agent_log_f", None)
        if log_f:
            log_f.close()


def uds_request(socket_path: str, req: Dict[str, Any], timeout_s: float = 2.0) -> Dict[str, Any]:
    data = (json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(socket_path)
        sock.sendall(data)
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                buf = buf.split(b"\n", 1)[0]
                break
    if not buf:
        raise BackendError("empty UDS response")
    return json.loads(buf.decode("utf-8"))


class Backend:
    name = "base"

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        raise NotImplementedError


class MockBackend(Backend):
    name = "mock"

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        content = f"[mock:{generation_name}] " + messages[-1]["content"][:80]
        return GenerationResult(
            backend=self.name,
            content=content,
            metrics={"ttft_ms": 0.0, "tpot_ms_after_first": 0.0, "total_ms": 0.0},
            dynamic_events=[],
        )


class PrimaBackend(Backend):
    name = "prima"

    def __init__(self, cuda_visible: str = "0,1,2", ctx: int = 512, timeout_s: int = 180):
        ensure_paths([PRIMA_CLI, PRIMA_MODEL])
        self.cuda_visible = cuda_visible
        self.ctx = ctx
        self.timeout_s = timeout_s

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        prompt = render_messages(messages)
        log_path = out_dir / f"{generation_name}_prima.log"
        cmd = [
            str(PRIMA_CLI),
            "-m",
            str(PRIMA_MODEL),
            "-c",
            str(self.ctx),
            "-n",
            str(max_tokens),
            "-p",
            prompt,
            "-ngl",
            "99",
            "--ignore-eos",
            "--no-display-prompt",
            "--temp",
            "0",
            "--seed",
            "1",
        ]
        env = base_env(self.cuda_visible)
        start = time.perf_counter()
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(PRIMA_DIR),
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.timeout_s,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        total_ms = (time.perf_counter() - start) * 1000.0
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        metrics = parse_prima_metrics(log_text)
        metrics["wall_ms"] = total_ms
        metrics["valid"] = rc == 0 and "error:" not in log_text.lower() and "failed" not in log_text.lower()
        return GenerationResult(
            backend=self.name,
            content=strip_prima_content(log_text),
            metrics=metrics,
            command=cmd,
            log_path=str(log_path),
            rc=rc,
        )


class EdgeVisorBackend(Backend):
    name = "edgevisor"

    def __init__(
        self,
        cuda_visible: str = "0,1,2",
        ctx: int = 512,
        steps: int = 128,
        timeout_s: int = 240,
        ratios: str = "1:1",
        worker_gpus: Optional[List[int]] = None,
    ):
        ensure_paths([EDGE_DLLAMA, EDGE_MODEL, EDGE_TOKENIZER])
        self.cuda_visible = cuda_visible
        self.ctx = ctx
        self.steps = steps
        self.timeout_s = timeout_s
        self.ratios = ratios
        self.worker_gpus = worker_gpus if worker_gpus is not None else [1]

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        prompt = render_messages(messages)
        env = base_env(self.cuda_visible)
        procs: List[subprocess.Popen[Any]] = []
        dynamic_events: List[Dict[str, Any]] = []
        port_base = 32000 + random.randint(0, 2000)
        worker_ports = [port_base + i + 1 for i in range(len(self.worker_gpus))]
        socket_path = None
        if dynamic_plan and dynamic_plan.get("enabled", True):
            socket_path = str(dynamic_plan.get("socket_path") or f"/tmp/edgevisor_agent_{os.getpid()}_{port_base}.sock")
            env["DLLAMA_PLAN_CTRL_SOCKET"] = socket_path
            env["DLLAMA_ENABLE_PLAN_BARRIER"] = "1"
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass

        logs = {
            "root": out_dir / f"{generation_name}_edgevisor_root.log",
        }
        for idx, gpu in enumerate(self.worker_gpus):
            logs[f"worker{idx}"] = out_dir / f"{generation_name}_edgevisor_worker_gpu{gpu}.log"

        try:
            steps = estimate_edge_steps(prompt, max_tokens, self.ctx, self.steps)
            for idx, gpu in enumerate(self.worker_gpus):
                worker_cmd = [
                    str(EDGE_DLLAMA),
                    "worker",
                    "--port",
                    str(worker_ports[idx]),
                    "--nthreads",
                    "1",
                    "--gpu-index",
                    str(gpu),
                ]
                procs.append(popen_log(worker_cmd, logs[f"worker{idx}"], EDGE_ENGINE, env))
            if self.worker_gpus:
                time.sleep(3.0)

            root_cmd = [
                str(EDGE_DLLAMA),
                "inference",
                "--prompt",
                prompt,
                "--steps",
                str(steps),
                "--model",
                str(EDGE_MODEL),
                "--tokenizer",
                str(EDGE_TOKENIZER),
                "--buffer-float-type",
                "q80",
                "--nthreads",
                "1",
                "--max-seq-len",
                str(self.ctx),
                "--temperature",
                "0",
                "--seed",
                "1",
                "--gpu-index",
                "0",
                "--benchmark",
            ]
            if self.worker_gpus:
                root_cmd.extend(["--workers", *[f"127.0.0.1:{p}" for p in worker_ports], "--ratios", self.ratios])
            if socket_path:
                root_cmd.extend(["--enable-plan-barrier", "--kv-redundancy", "2"])

            start = time.perf_counter()
            root_proc = popen_log(root_cmd, logs["root"], EDGE_ENGINE, env)
            procs.append(root_proc)

            if socket_path:
                dynamic_events.extend(self._drive_dynamic_plan(socket_path, root_proc, dynamic_plan or {}))

            try:
                rc = root_proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                rc = 124
            wall_ms = (time.perf_counter() - start) * 1000.0
        finally:
            terminate_all(procs)
            close_logs(procs)
            if socket_path:
                try:
                    os.unlink(socket_path)
                except FileNotFoundError:
                    pass

        log_text = logs["root"].read_text(encoding="utf-8", errors="replace") if logs["root"].exists() else ""
        metrics = parse_edge_metrics(log_text)
        metrics["wall_ms"] = wall_ms
        metrics["valid"] = rc == 0 and "Critical" not in log_text and "error" not in log_text.lower()
        return GenerationResult(
            backend=self.name,
            content=strip_edge_content(log_text),
            metrics=metrics,
            command=root_cmd,
            log_path=str(logs["root"]),
            dynamic_events=dynamic_events,
            rc=rc,
        )

    def _drive_dynamic_plan(
        self,
        socket_path: str,
        root_proc: subprocess.Popen[Any],
        dynamic_plan: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        deadline = time.time() + float(dynamic_plan.get("ready_timeout_s", 30.0))
        while time.time() < deadline and root_proc.poll() is None:
            try:
                resp = uds_request(socket_path, {"op": "ping"}, timeout_s=1.0)
                events.append({"event": "uds_ping", "response": resp})
                if resp.get("ok"):
                    break
            except Exception as exc:
                events.append({"event": "uds_wait", "error": str(exc)})
                time.sleep(0.2)
        else:
            events.append({"event": "uds_not_ready"})
            return events

        delay_s = float(dynamic_plan.get("delay_s", 0.5))
        time.sleep(max(0.0, delay_s))
        seq = int(dynamic_plan.get("seq", int(time.time()) % 100000))
        cmd: Dict[str, Any] = {
            "seq": seq,
            "mode": dynamic_plan.get("mode", "next_barrier"),
            "stageIndex": int(dynamic_plan.get("stage", 0)),
        }
        if "moves" in dynamic_plan:
            cmd["moves"] = dynamic_plan["moves"]
        else:
            cmd.update(
                {
                    "fromNodeIndex": int(dynamic_plan.get("from_node", 1)),
                    "toNodeIndex": int(dynamic_plan.get("to_node", 0)),
                    "cmdKind": int(dynamic_plan.get("kind", 3)),
                    "nHeadsToMove": int(dynamic_plan.get("heads", 1)),
                    "nFfnToMove": int(dynamic_plan.get("ffn", 256)),
                }
            )
        try:
            set_resp = uds_request(socket_path, {"op": "set_plan", "cmd": cmd}, timeout_s=2.0)
            events.append({"event": "set_plan", "request": cmd, "response": set_resp})
        except Exception as exc:
            events.append({"event": "set_plan_error", "error": str(exc), "request": cmd})
            return events

        consume_deadline = time.time() + float(dynamic_plan.get("consume_timeout_s", 20.0))
        while time.time() < consume_deadline and root_proc.poll() is None:
            try:
                st = uds_request(socket_path, {"op": "status"}, timeout_s=1.0)
                events.append({"event": "status", "response": st})
                cmd_state = st.get("cmd") if isinstance(st, dict) else None
                if isinstance(cmd_state, dict) and cmd_state.get("mode") == "none":
                    events.append({"event": "plan_consumed"})
                    break
            except Exception as exc:
                events.append({"event": "status_error", "error": str(exc)})
            time.sleep(0.2)
        return events


def make_backend(name: str, **kwargs: Any) -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "prima":
        return PrimaBackend(**kwargs)
    if name == "edgevisor":
        return EdgeVisorBackend(**kwargs)
    raise BackendError(f"unknown backend: {name}")

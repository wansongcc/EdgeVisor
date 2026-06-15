from __future__ import annotations

import json
import http.client
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(os.environ.get("B01_ROOT", "/home/byh/B01"))
EDGE_PROJECT = ROOT / "EdgeVisor"
EDGE_ENGINE = EDGE_PROJECT / "EdgeVisor"
EDGE_DLLAMA = EDGE_ENGINE / "dllama"
EDGE_DLLAMA_API = EDGE_ENGINE / "dllama-api"
EDGE_MODEL = ROOT / "models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
EDGE_TOKENIZER = ROOT / "models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"

EDGE_EXO_PROJECT = ROOT / "EdgeVisor-EXO"
EDGE_EXO_ENGINE = EDGE_EXO_PROJECT / "EdgeVisor"
EDGE_EXO_DLLAMA = EDGE_EXO_ENGINE / "dllama"
EDGE_EXO_PLAN = EDGE_EXO_ENGINE / "src/edgevisor_exo_plan.py"
EDGE_EXO_MODEL = EDGE_MODEL
EDGE_EXO_TOKENIZER = EDGE_TOKENIZER

EDGE_LL_PROJECT = ROOT / "EdgeVisor-LinguaLinked"
EDGE_LL_ENGINE = EDGE_LL_PROJECT / "EdgeVisor"
EDGE_LL_DLLAMA = EDGE_LL_ENGINE / "dllama"
EDGE_LL_PLAN = EDGE_LL_ENGINE / "src/lingualinked_plan.py"
EDGE_LL_MODEL = EDGE_MODEL
EDGE_LL_TOKENIZER = EDGE_TOKENIZER

DLLAMA_PROJECT = ROOT / "distributed-llama"
DLLAMA_ENGINE = DLLAMA_PROJECT
DLLAMA_CLI = DLLAMA_ENGINE / "dllama"
DLLAMA_MODEL = EDGE_MODEL
DLLAMA_TOKENIZER = EDGE_TOKENIZER

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


class TcpRateProxy:
    def __init__(
        self,
        listen_port: int,
        target_port: int,
        rate_bytes_per_s: float,
        log_path: Path,
        start_throttled: bool = True,
    ):
        self.listen_port = int(listen_port)
        self.target_port = int(target_port)
        self.rate_bytes_per_s = float(rate_bytes_per_s)
        self.log_path = log_path
        self._stop = threading.Event()
        self._throttle = threading.Event()
        if start_throttled:
            self._throttle.set()
        self._server: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", self.listen_port))
        server.listen(16)
        server.settimeout(0.5)
        self._server = server
        thread = threading.Thread(target=self._accept_loop, name=f"edgevisor-proxy-{self.listen_port}", daemon=True)
        thread.start()
        self._threads.append(thread)
        self._log(
            f"listen=127.0.0.1:{self.listen_port} target=127.0.0.1:{self.target_port} "
            f"rate_bytes_per_s={self.rate_bytes_per_s:.3f} throttled={self._throttle.is_set()}"
        )

    def activate(self) -> None:
        self._throttle.set()
        self._log("throttle_activated")

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        for thread in list(self._threads):
            thread.join(timeout=1.0)
        self._threads.clear()

    def _log(self, text: str) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{time.time():.6f} {text}\n")
        except OSError:
            pass

    def _connect_target(self) -> socket.socket:
        last_error: Optional[Exception] = None
        for _ in range(200):
            if self._stop.is_set():
                raise RuntimeError("proxy stopped")
            try:
                return socket.create_connection(("127.0.0.1", self.target_port), timeout=1.0)
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        raise RuntimeError(f"target 127.0.0.1:{self.target_port} not ready: {last_error}")

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                target = self._connect_target()
            except Exception as exc:
                self._log(f"connect_target_error={exc}")
                try:
                    client.close()
                except OSError:
                    pass
                continue
            self._log(f"accepted addr={addr}")
            for src, dst, label in ((client, target, "client_to_target"), (target, client, "target_to_client")):
                thread = threading.Thread(target=self._pipe, args=(src, dst, label), daemon=True)
                thread.start()
                self._threads.append(thread)

    def _pipe(self, src: socket.socket, dst: socket.socket, label: str) -> None:
        total = 0
        try:
            src.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    chunk = src.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                dst.sendall(chunk)
                total += len(chunk)
                if self.rate_bytes_per_s > 0.0 and self._throttle.is_set():
                    time.sleep(len(chunk) / self.rate_bytes_per_s)
        except OSError as exc:
            self._log(f"pipe_error label={label} error={exc}")
        finally:
            self._log(f"pipe_done label={label} bytes={total}")
            for sock in (src, dst):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
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


def base_env(cuda_visible: str = "0,1,2", vulkan_project: Path = EDGE_PROJECT) -> Dict[str, str]:
    env = os.environ.copy()
    vulkan_bin = vulkan_project / "tools/vulkan_deps/root/usr/bin"
    vulkan_lib = vulkan_project / "tools/vulkan_deps/root/usr/lib/x86_64-linux-gnu"
    if vulkan_bin.exists():
        env["PATH"] = f"{vulkan_bin}:{env.get('PATH', '')}"
    if vulkan_lib.exists():
        env["LD_LIBRARY_PATH"] = (
            f"{vulkan_lib}:/usr/local/cuda/lib64:/usr/local/lib:"
            f"{env.get('LD_LIBRARY_PATH', '')}"
        )
    else:
        env["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:/usr/local/lib:{env.get('LD_LIBRARY_PATH', '')}"
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
    stage_profile: List[Dict[str, Any]] = []
    for match in re.finditer(
        r"Stage\s+(\d+)\s+Node\s+(\d+):\s+per-fwd total=\s*([0-9.]+)\s*ms"
        r"\s+\(exec=\s*([0-9.]+)\s+sync=\s*([0-9.]+)\)"
        r"\s+\|\s+per-tok total=\s*([0-9.]+)\s*ms"
        r"\s+\(exec=\s*([0-9.]+)\s+sync=\s*([0-9.]+)\)"
        r"\s+\|\s+fwd=(\d+)\s+tok=(\d+)",
        log_text,
    ):
        stage_profile.append(
            {
                "stage": int(match.group(1)),
                "node": int(match.group(2)),
                "per_fwd_total_ms": float(match.group(3)),
                "per_fwd_exec_ms": float(match.group(4)),
                "per_fwd_sync_ms": float(match.group(5)),
                "per_tok_total_ms": float(match.group(6)),
                "per_tok_exec_ms": float(match.group(7)),
                "per_tok_sync_ms": float(match.group(8)),
                "forward_count": int(match.group(9)),
                "token_count": int(match.group(10)),
            }
        )
    stage_profile.sort(key=lambda x: int(x["stage"]))
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
        "stage_profile": stage_profile,
    }


def _jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def read_text_since(path: Path, offset: int) -> tuple[str, int]:
    if not path.exists():
        return "", offset
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        text = f.read()
        return text, f.tell()


def jsonl_records_since(path: Path, offset: int) -> tuple[List[Dict[str, Any]], int]:
    text, new_offset = read_text_since(path, offset)
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records, new_offset


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

    def close(self) -> None:
        return None


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

    def __init__(self, cuda_visible: str = "0,1,2", ctx: int = 2048, timeout_s: int = 180):
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
        ctx: int = 2048,
        steps: int = 128,
        timeout_s: int = 240,
        ratios: str = "1:1",
        worker_gpus: Optional[List[int]] = None,
        backend_name: Optional[str] = None,
        project_dir: Path = EDGE_PROJECT,
        engine_dir: Path = EDGE_ENGINE,
        dllama_path: Path = EDGE_DLLAMA,
        api_path: Path = EDGE_DLLAMA_API,
        model_path: Path = EDGE_MODEL,
        tokenizer_path: Path = EDGE_TOKENIZER,
        vulkan_project: Optional[Path] = None,
        root_gpu: int = 0,
        enable_benchmark: bool = True,
        extra_root_args: Optional[List[str]] = None,
        extra_env: Optional[Dict[str, str]] = None,
        ablation_config: Optional[Dict[str, Any]] = None,
        virtual_topology: Optional[Dict[str, Any]] = None,
    ):
        ensure_paths([dllama_path, model_path, tokenizer_path])
        if backend_name:
            self.name = backend_name
        self.cuda_visible = cuda_visible
        self.ctx = ctx
        self.steps = steps
        self.timeout_s = timeout_s
        self.ratios = ratios
        self.worker_gpus = worker_gpus if worker_gpus is not None else [1]
        self.project_dir = project_dir
        self.engine_dir = engine_dir
        self.dllama_path = dllama_path
        self.api_path = api_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.vulkan_project = vulkan_project or project_dir
        self.root_gpu = root_gpu
        self.enable_benchmark = enable_benchmark
        self.extra_root_args = list(extra_root_args or [])
        self.extra_env = dict(extra_env or {})
        self.ablation_config = dict(ablation_config or {})
        self.virtual_topology = self._normalize_virtual_topology(virtual_topology)
        if self.virtual_topology:
            self.root_gpu = int(self.virtual_topology["root_gpu"])
            self.worker_gpus = list(self.virtual_topology["worker_gpus"])
            self.ratios = str(self.virtual_topology["ratios"])
        self.last_plan: Optional[Dict[str, Any]] = None
        self.ablation_effective_topology = self._apply_ablation_topology()

    def _choose_port_base(self) -> int:
        fixed = (
            self.extra_env.get("EDGEVISOR_FIXED_PORT_BASE")
            or self.extra_env.get("EDGEVISOR_PORT_BASE")
            or os.environ.get("EDGEVISOR_FIXED_PORT_BASE")
            or os.environ.get("EDGEVISOR_PORT_BASE")
        )
        if fixed:
            port_base = int(fixed)
            if port_base <= 0:
                raise BackendError(f"invalid EDGEVISOR_FIXED_PORT_BASE: {fixed}")
            return port_base
        return 32000 + random.randint(0, 2000)

    def _normalize_virtual_topology(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not cfg or not cfg.get("enabled", False):
            return {}
        ratios = str(cfg.get("ratios") or "1:1:1*1:1:1*1:1")
        logical_node_gpus = cfg.get("logical_node_gpus") or [0, 1, 2, 0, 1, 2, 0, 1]
        logical_node_gpus = [int(x) for x in logical_node_gpus]
        if len(logical_node_gpus) < 2:
            raise BackendError("virtual topology requires at least root plus one worker node")
        root_gpu = int(cfg.get("root_gpu", logical_node_gpus[0]))
        worker_gpus = cfg.get("worker_gpus")
        if worker_gpus is None:
            worker_gpus = logical_node_gpus[1:]
        worker_gpus = [int(x) for x in worker_gpus]
        if len(worker_gpus) + 1 != len(logical_node_gpus):
            raise BackendError("virtual topology worker_gpus must match logical_node_gpus[1:] length")
        launch_stagger_s = float(cfg.get("launch_stagger_s", 2.0))
        node_gpu_map = {idx: gpu for idx, gpu in enumerate(logical_node_gpus)}
        return {
            "enabled": True,
            "name": str(cfg.get("name") or "virtual_pp_tp_3stage_3_3_2"),
            "ratios": ratios,
            "root_gpu": root_gpu,
            "worker_gpus": worker_gpus,
            "logical_node_gpus": logical_node_gpus,
            "node_gpu_map": node_gpu_map,
            "launch_stagger_s": launch_stagger_s,
            "stage_layout": cfg.get("stage_layout") or [3, 3, 2],
        }

    def _worker_log_path(self, out_dir: Path, prefix: str, worker_idx: int, gpu: int) -> Path:
        node_id = worker_idx + 1
        return out_dir / f"{prefix}_worker_node{node_id}_gpu{gpu}.log"

    def _node_metric_entry(self, label: str, path: Path, text: str) -> Dict[str, Any]:
        gpu = self.root_gpu if label == "root" else None
        logical_node = 0 if label == "root" else None
        if label.startswith("worker"):
            try:
                worker_idx = int(label.removeprefix("worker"))
                gpu = self.worker_gpus[worker_idx]
                logical_node = worker_idx + 1
            except Exception:
                gpu = None
                logical_node = None
        return {
            "node": label,
            "logical_node": logical_node,
            "gpu": gpu,
            "log_path": str(path),
            "metrics": parse_edge_metrics(text),
        }

    def _launch_worker(self, worker_cmd: List[str], log_path: Path, env: Dict[str, str]) -> subprocess.Popen[Any]:
        return popen_log(worker_cmd, log_path, self.engine_dir, env)

    def _network_proxy_rate_bytes_per_s(self) -> float:
        cfg = self.ablation_config or {}
        rate = float(cfg.get("network_proxy_rate_bytes_per_s", 0.0) or 0.0)
        if rate <= 0.0:
            rate_mbps = float(cfg.get("network_proxy_rate_mbps", 0.0) or 0.0)
            if rate_mbps > 0.0:
                rate = rate_mbps * 1024.0 * 1024.0
        return rate

    def _network_proxy_nodes(self) -> Optional[set[int]]:
        cfg = self.ablation_config or {}
        raw = cfg.get("network_proxy_nodes")
        scope = str(cfg.get("network_proxy_scope", "all_worker_connections"))
        if raw is None and scope == "all_worker_connections":
            return None
        if raw is None and scope in {"selected_worker_nodes", "migration_nodes"}:
            raw = [
                cfg.get("network_proxy_from_node"),
                cfg.get("network_proxy_to_node"),
            ]
        if raw is None:
            return None
        if isinstance(raw, str):
            if raw.strip().lower() in {"", "all", "*", "all_worker_connections"}:
                return None
            items = [item.strip() for item in raw.split(",")]
        elif isinstance(raw, (list, tuple, set)):
            items = list(raw)
        else:
            items = [raw]
        nodes: set[int] = set()
        for item in items:
            if item is None or str(item).strip() == "":
                continue
            node = int(item)
            if node <= 0:
                raise BackendError(f"network_proxy_nodes must name worker logical nodes >= 1, got {node}")
            nodes.add(node)
        return nodes or None

    def _make_worker_ports(self, port_base: int, out_dir: Path, prefix: str) -> tuple[List[int], List[int], List[TcpRateProxy]]:
        worker_ports = [port_base + i + 1 for i in range(len(self.worker_gpus))]
        proxy_rate = self._network_proxy_rate_bytes_per_s()
        if proxy_rate <= 0.0:
            return worker_ports, worker_ports, []
        worker_actual_ports = [port_base + 100 + i + 1 for i in range(len(self.worker_gpus))]
        proxies: List[TcpRateProxy] = []
        proxied_nodes = self._network_proxy_nodes()
        for idx, (listen_port, target_port) in enumerate(zip(worker_ports, worker_actual_ports)):
            node_id = idx + 1
            if proxied_nodes is not None and node_id not in proxied_nodes:
                worker_actual_ports[idx] = listen_port
                continue
            proxy = TcpRateProxy(
                listen_port=listen_port,
                target_port=target_port,
                rate_bytes_per_s=proxy_rate,
                log_path=out_dir / f"{prefix}_proxy_node{idx + 1}.log",
                start_throttled=bool((self.ablation_config or {}).get("network_proxy_start_throttled", True)),
            )
            proxy.start()
            proxies.append(proxy)
        return worker_actual_ports, worker_ports, proxies

    def _apply_ablation_topology(self) -> Dict[str, Any]:
        if not self.ablation_config:
            return {}
        cfg = self.ablation_config
        vg_mode = str(cfg.get("vg_mode", "enabled"))
        original_workers = list(self.worker_gpus)
        original_ratios = self.ratios
        effective_workers = list(self.worker_gpus)
        effective_ratios = self.ratios

        if vg_mode in {"flat", "pure_pp", "no_elastic_vg"}:
            effective_ratios = ":".join("1" for _ in range(1 + len(effective_workers)))
        elif vg_mode == "random":
            seed = str(cfg.get("experiment_id") or "edgevisor_ablation")
            rng = random.Random(seed)
            rng.shuffle(effective_workers)
            effective_ratios = ":".join("1" for _ in range(1 + len(effective_workers)))

        self.worker_gpus = effective_workers
        self.ratios = effective_ratios
        return {
            "vg_mode": vg_mode,
            "root_gpu": self.root_gpu,
            "original_worker_gpus": original_workers,
            "effective_worker_gpus": effective_workers,
            "original_ratios": original_ratios,
            "effective_ratios": effective_ratios,
            "virtual_topology": dict(self.virtual_topology),
        }

    def _plan_barrier_args(self) -> List[str]:
        if not self.ablation_config:
            return ["--enable-plan-barrier", "--kv-redundancy", "2"]
        shadow_mode = str(self.ablation_config.get("shadow_kv_mode", "enabled"))
        allow_head_kv_migration = bool(self.ablation_config.get("allow_head_kv_migration", False))
        kv_redundancy = str(self.ablation_config.get("kv_redundancy", "2"))
        if shadow_mode == "enabled":
            args = ["--enable-plan-barrier", "--kv-redundancy", kv_redundancy]
            if allow_head_kv_migration:
                args = [
                    "--enable-plan-barrier",
                    "--enable-stage-full-weights",
                    "--enable-kv-redundancy-during-migration",
                    "1",
                    "--kv-redundancy",
                    kv_redundancy,
                ]
            return self._append_pp_migration_args(args)
        args = [
            "--enable-plan-barrier",
            "--kv-redundancy",
            "0",
            "--enable-kv-redundancy-during-migration",
            "0",
        ]
        if allow_head_kv_migration:
            args.append("--allow-no-shadow-head-migration")
        return self._append_pp_migration_args(args)

    def _append_pp_migration_args(self, args: List[str]) -> List[str]:
        if not self.ablation_config or not self.ablation_config.get("enable_pp_migration", False):
            return args
        out = list(args)
        if "--enable-stage-full-weights" not in out:
            out.append("--enable-stage-full-weights")
        out.append("--enable-pp-migration")
        boundary_layers = int(self.ablation_config.get("runtime_redundant_boundary_layers", 1))
        out.extend(["--runtime-redundant-boundary-layers", str(max(0, boundary_layers))])
        out.extend(["--runtime-active-seg-enabled", "1"])
        shadow_mode = str(self.ablation_config.get("shadow_kv_mode", "enabled"))
        redundant_enabled = "1" if shadow_mode == "enabled" else "0"
        out.extend(["--runtime-redundant-seg-enabled", redundant_enabled])
        return out

    def _ablation_dynamic_plan(self, dynamic_plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self.ablation_config or dynamic_plan is None:
            return dynamic_plan
        cfg = self.ablation_config
        plan = dict(dynamic_plan)
        if isinstance(plan.get("commands"), list):
            base = {k: v for k, v in plan.items() if k != "commands"}
            seq_base = int(plan.get("seq_base", int(time.time()) % 100000))
            commands: List[Dict[str, Any]] = []
            for idx, item in enumerate(plan.get("commands", [])):
                if not isinstance(item, dict):
                    continue
                merged = dict(base)
                merged.update(item)
                merged.pop("commands", None)
                merged.setdefault("seq", seq_base + idx + 1)
                command_plan = self._ablation_dynamic_plan(merged)
                if isinstance(command_plan, dict):
                    commands.append(command_plan)
            plan["commands"] = commands
            plan["candidateCount"] = int(plan.get("candidate_count", len(commands) or 1))
            plan["expected_command_count"] = int(plan.get("expected_command_count", len(commands)))
            return plan
        jit_mode = str(cfg.get("jit_mode", "enabled"))
        vg_mode = str(cfg.get("vg_mode", "enabled"))
        shadow_mode = str(cfg.get("shadow_kv_mode", "enabled"))
        allow_head_kv_migration = bool(cfg.get("allow_head_kv_migration", False))
        is_pp_plan = str(plan.get("plan_op", "set_plan")) == "set_pp_migration"

        if jit_mode == "static" and not is_pp_plan:
            plan["stage"] = int(plan.get("stage", 0))
            plan["moves"] = [
                {
                    "fromNodeIndex": int(plan.get("static_from_node", 1)),
                    "toNodeIndex": int(plan.get("static_to_node", 0)),
                    "cmdKind": int(plan.get("static_kind", 3)),
                    "headMove": int(plan.get("static_heads", 1)),
                    "ffnMove": int(plan.get("static_ffn", 256)),
                }
            ]
            plan["jit_decision_reason"] = "static_fixed_rule"
        elif jit_mode == "greedy" and not is_pp_plan:
            from_node = int(plan.get("greedy_from_node", len(self.worker_gpus)))
            to_node = int(plan.get("greedy_to_node", 0))
            plan["moves"] = [
                {
                    "fromNodeIndex": from_node,
                    "toNodeIndex": to_node,
                    "cmdKind": 3,
                    "headMove": int(plan.get("heads", 1)),
                    "ffnMove": int(plan.get("ffn", 256)),
                }
            ]
            plan["jit_decision_reason"] = "greedy_resource_only_rule"
        elif jit_mode == "oracle":
            plan["jit_decision_reason"] = "oracle_offline_upper_bound_placeholder"
        else:
            plan["jit_decision_reason"] = "runtime_enabled"

        plan["candidateCount"] = int(plan.get("candidate_count", 1))
        plan["vgMappingBefore"] = plan.get("vg_mapping_before", f"vg_{vg_mode}:{self.ratios}")
        plan["vgMappingAfter"] = plan.get("vg_mapping_after", f"vg_{vg_mode}:{self.ratios}")
        plan["physicalDeviceGroup"] = plan.get(
            "physical_device_group",
            ",".join(str(x) for x in [self.root_gpu, *self.worker_gpus]),
        )
        plan["logicalGroup"] = plan.get("logical_group", f"{vg_mode}_stage_{int(plan.get('stage', 0))}")
        if self.virtual_topology:
            plan["nodeGpuMap"] = plan.get("node_gpu_map", json.dumps(self.virtual_topology["node_gpu_map"], sort_keys=True))

        if shadow_mode == "disabled_transfer":
            plan["fallbackCount"] = int(plan.get("fallback_count", 1))
        elif shadow_mode == "disabled_recompute":
            plan["fallbackCount"] = int(plan.get("fallback_count", 1))
        if isinstance(plan.get("moves"), list):
            if not allow_head_kv_migration:
                safe_moves = []
                rejected = int(plan.get("rejectedMoves", 0))
                for move in plan["moves"]:
                    if not isinstance(move, dict):
                        safe_moves.append(move)
                        continue
                    m = dict(move)
                    if int(m.get("headMove", 0) or 0) > 0:
                        rejected += 1
                        m["headMove"] = 0
                        if int(m.get("ffnMove", 0) or 0) > 0:
                            m["cmdKind"] = 2
                        else:
                            m["cmdKind"] = 0
                    safe_moves.append(m)
                plan["moves"] = [
                    m for m in safe_moves
                    if not (
                        isinstance(m, dict) and
                        int(m.get("headMove", 0) or 0) == 0 and
                        int(m.get("ffnMove", 0) or 0) == 0
                    )
                ]
                if rejected:
                    plan["rejectedMoves"] = rejected
                    plan["fallbackCount"] = max(int(plan.get("fallbackCount", 0)), 1)
                    plan["fallbackReason"] = "kv_head_move_requires_allow_head_kv_migration"
            else:
                plan["allowHeadKvMigration"] = True
        return plan

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        prompt = render_messages(messages)
        dynamic_plan = self._ablation_dynamic_plan(dynamic_plan)
        env = base_env(self.cuda_visible, self.vulkan_project)
        env.update(self.extra_env)
        procs: List[subprocess.Popen[Any]] = []
        proxies: List[TcpRateProxy] = []
        dynamic_events: List[Dict[str, Any]] = []
        port_base = self._choose_port_base()
        worker_actual_ports, root_worker_ports, proxies = self._make_worker_ports(port_base, out_dir, generation_name)
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
        ablation_log_path = out_dir / f"{generation_name}_ablation.jsonl"
        for idx, gpu in enumerate(self.worker_gpus):
            logs[f"worker{idx}"] = self._worker_log_path(out_dir, f"{generation_name}_edgevisor", idx, gpu)

        try:
            steps = estimate_edge_steps(prompt, max_tokens, self.ctx, self.steps)
            launch_stagger_s = float(self.virtual_topology.get("launch_stagger_s", 0.0)) if self.virtual_topology else 0.0
            for idx, gpu in enumerate(self.worker_gpus):
                worker_cmd = [
                    str(self.dllama_path),
                    "worker",
                    "--port",
                    str(worker_actual_ports[idx]),
                    "--nthreads",
                    "1",
                    "--gpu-index",
                    str(gpu),
                ]
                procs.append(self._launch_worker(worker_cmd, logs[f"worker{idx}"], env))
                if launch_stagger_s > 0.0 and idx + 1 < len(self.worker_gpus):
                    time.sleep(launch_stagger_s)
            if self.worker_gpus:
                time.sleep(3.0)

            root_cmd = [
                str(self.dllama_path),
                "inference",
                "--prompt",
                prompt,
                "--steps",
                str(steps),
                "--model",
                str(self.model_path),
                "--tokenizer",
                str(self.tokenizer_path),
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
                str(self.root_gpu),
            ]
            if self.enable_benchmark:
                root_cmd.append("--benchmark")
            if self.worker_gpus:
                root_cmd.extend(["--workers", *[f"127.0.0.1:{p}" for p in root_worker_ports]])
                if self.ratios:
                    root_cmd.extend(["--ratios", self.ratios])
            if socket_path:
                root_cmd.extend(self._plan_barrier_args())
            if self.ablation_config:
                cfg = self.ablation_config
                root_cmd.extend(["--ablation-log-path", str(ablation_log_path)])
                root_cmd.extend(["--experiment-id", str(cfg.get("experiment_id", generation_name))])
                root_cmd.extend(["--shadow-kv-mode", str(cfg.get("shadow_kv_mode", "enabled"))])
                root_cmd.extend(["--pointer-swizzling-mode", str(cfg.get("pointer_swizzling_mode", "enabled"))])
                root_cmd.extend(["--jit-mode", str(cfg.get("jit_mode", "enabled"))])
                root_cmd.extend(["--vg-mode", str(cfg.get("vg_mode", "enabled"))])
                root_cmd.extend(["--fallback-policy", str(cfg.get("fallback_policy", "disabled_unless_necessary"))])
                if cfg.get("config_path"):
                    root_cmd.extend(["--edgevisor-ablation-config", str(cfg["config_path"])])
            if self.extra_root_args:
                root_cmd.extend(self.extra_root_args)

            start = time.perf_counter()
            root_proc = popen_log(root_cmd, logs["root"], self.engine_dir, env)
            procs.append(root_proc)
            if not bool((self.ablation_config or {}).get("network_proxy_start_throttled", True)):
                for proxy in proxies:
                    proxy.activate()

            if socket_path:
                dynamic_events.extend(self._drive_dynamic_plan(socket_path, root_proc, dynamic_plan or {}, ablation_log_path))

            try:
                rc = root_proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                rc = 124
            wall_ms = (time.perf_counter() - start) * 1000.0
        finally:
            terminate_all(procs)
            close_logs(procs)
            for proxy in proxies:
                proxy.stop()
            if socket_path:
                try:
                    os.unlink(socket_path)
                except FileNotFoundError:
                    pass

        log_text = logs["root"].read_text(encoding="utf-8", errors="replace") if logs["root"].exists() else ""
        metrics = parse_edge_metrics(log_text)
        metrics["wall_ms"] = wall_ms
        metrics["valid"] = rc == 0 and "Critical" not in log_text and "error" not in log_text.lower()
        node_metrics: List[Dict[str, Any]] = []
        for label, path in logs.items():
            text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            node_metrics.append(self._node_metric_entry(label, path, text))
        metrics["node_metrics"] = node_metrics
        if self.ablation_config:
            metrics["ablation_config"] = dict(self.ablation_config)
            metrics["ablation_effective_topology"] = dict(self.ablation_effective_topology)
        ablation_records = _jsonl_records(ablation_log_path)
        if ablation_records:
            metrics["ablation_log_path"] = str(ablation_log_path)
            metrics["ablation_events"] = ablation_records
        if self.last_plan is not None:
            metrics["exo_plan"] = self.last_plan
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
        ablation_log_path: Optional[Path] = None,
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

        if isinstance(dynamic_plan.get("commands"), list):
            shared = {k: v for k, v in dynamic_plan.items() if k != "commands"}
            for idx, item in enumerate(dynamic_plan.get("commands", [])):
                if root_proc.poll() is not None:
                    events.append({"event": "root_exited_before_plan", "command_index": idx})
                    break
                if not isinstance(item, dict):
                    continue
                command_plan = dict(shared)
                command_plan.update(item)
                command_plan.pop("commands", None)
                command_plan.setdefault("delay_s", 0.0 if idx == 0 else float(shared.get("inter_command_delay_s", 0.0) or 0.0))
                events.extend(
                    self._drive_dynamic_plan_command(
                        socket_path,
                        root_proc,
                        command_plan,
                        command_index=idx,
                        ablation_log_path=ablation_log_path,
                    )
                )
            return events

        events.extend(
            self._drive_dynamic_plan_command(
                socket_path,
                root_proc,
                dynamic_plan,
                command_index=0,
                ablation_log_path=ablation_log_path,
            )
        )
        return events

    def _drive_dynamic_plan_command(
        self,
        socket_path: str,
        root_proc: subprocess.Popen[Any],
        dynamic_plan: Dict[str, Any],
        *,
        command_index: int = 0,
        ablation_log_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        delay_s = float(dynamic_plan.get("delay_s", 0.5))
        time.sleep(max(0.0, delay_s))
        dynamic_plan = dict(dynamic_plan)
        trigger_pos_strategy = str(dynamic_plan.get("trigger_pos_strategy", "")).strip().lower()
        if trigger_pos_strategy in {"status_offset", "current_offset", "future_status_offset"}:
            try:
                st = uds_request(socket_path, {"op": "status"}, timeout_s=1.0)
                events.append({"event": "pre_trigger_status", "response": st})
                cur_pos = int(st.get("position", 0) or 0)
                offset = max(1, int(dynamic_plan.get("trigger_pos_offset", 1) or 1))
                dynamic_plan["trigger_pos"] = cur_pos + offset
                dynamic_plan["triggerPos"] = cur_pos + offset
                dynamic_plan["resolved_trigger_pos"] = cur_pos + offset
            except Exception as exc:
                events.append({"event": "pre_trigger_status_error", "error": str(exc)})
        seq = int(dynamic_plan.get("seq", int(time.time()) % 100000))
        cmd: Dict[str, Any] = {
            "seq": seq,
            "mode": dynamic_plan.get("mode", "next_barrier"),
            "stageIndex": int(dynamic_plan.get("stage", 0)),
        }
        for key in (
            "candidateCount",
            "rejectedMoves",
            "fallbackCount",
            "layerCount",
            "tDecisionMs",
            "tStatePrepareMs",
            "tCommandMs",
            "tApplyMs",
            "tRecoverMs",
            "stallTimeMs",
            "fallbackReason",
            "fallback_reason",
            "triggerPos",
            "triggerLayer",
            "vgMappingBefore",
            "vgMappingAfter",
            "physicalDeviceGroup",
            "logicalGroup",
        ):
            if key in dynamic_plan:
                cmd[key] = dynamic_plan[key]
        if "trigger_pos" in dynamic_plan:
            cmd["triggerPos"] = dynamic_plan["trigger_pos"]
        if "trigger_layer" in dynamic_plan:
            cmd["triggerLayer"] = dynamic_plan["trigger_layer"]
        plan_op = str(dynamic_plan.get("plan_op", "set_plan"))
        if plan_op == "set_pp_migration":
            cmd.update(
                {
                    "fromNodeIndex": int(dynamic_plan.get("fromNodeIndex", dynamic_plan.get("from_node", 0))),
                    "toNodeIndex": int(dynamic_plan.get("toNodeIndex", dynamic_plan.get("to_node", 1))),
                    "layerCount": int(dynamic_plan.get("layerCount", dynamic_plan.get("layer_count", 1))),
                }
            )
        elif "moves" in dynamic_plan:
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
            set_resp = uds_request(socket_path, {"op": plan_op, "cmd": cmd}, timeout_s=2.0)
            events.append(
                {
                    "event": plan_op,
                    "command_index": command_index,
                    "request": cmd,
                    "response": set_resp,
                    "ablation_config": dict(self.ablation_config),
                    "ablation_effective_topology": dict(self.ablation_effective_topology),
                    "jit_decision_reason": dynamic_plan.get("jit_decision_reason"),
                    "stall_time_ms": float(dynamic_plan.get("stallTimeMs", dynamic_plan.get("tStatePrepareMs", 0.0)) or 0.0),
                    "recovery_latency_ms": float(dynamic_plan.get("tRecoverMs", dynamic_plan.get("stallTimeMs", 0.0)) or 0.0),
                    "fluctuation_type": dynamic_plan.get("fluctuation_type", "unspecified"),
                }
            )
        except Exception as exc:
            events.append({"event": "set_plan_error", "error": str(exc), "request": cmd})
            return events

        consume_deadline = time.time() + float(dynamic_plan.get("consume_timeout_s", 20.0))
        ablation_offset = ablation_log_path.stat().st_size if ablation_log_path is not None and ablation_log_path.exists() else 0
        expected_trigger_pos = int(cmd.get("triggerPos", -1) or -1)
        expected_trigger_layer = int(cmd.get("triggerLayer", -1) or -1)
        shadow_mode = str((self.ablation_config or {}).get("shadow_kv_mode", "enabled"))
        scope = str(dynamic_plan.get("shadow_scope", (self.ablation_config or {}).get("shadow_scope", "")))
        if shadow_mode == "enabled":
            observed_event_ids = {"pp_migration_recover"} if plan_op == "set_pp_migration" else {"plan_command_apply"}
        elif scope == "inter_stage_layers" or plan_op == "set_pp_migration":
            observed_event_ids = {"pp_migration_recover"}
        else:
            observed_event_ids = {"head_migration_recover"}
        while time.time() < consume_deadline and root_proc.poll() is None:
            try:
                st = uds_request(socket_path, {"op": "status"}, timeout_s=1.0)
                events.append({"event": "status", "response": st})
                cmd_state = st.get("cmd") if isinstance(st, dict) else None
                if isinstance(cmd_state, dict) and cmd_state.get("mode") == "none":
                    events.append({"event": "plan_consumed", "command_index": command_index})
                    break
            except Exception as exc:
                events.append({"event": "status_error", "error": str(exc)})
            if ablation_log_path is not None:
                records, ablation_offset = jsonl_records_since(ablation_log_path, ablation_offset)
                for rec in records:
                    event_id = str(rec.get("event_id", ""))
                    if event_id not in observed_event_ids:
                        continue
                    if not bool(rec.get("apply_success", False)):
                        continue
                    rec_pos = int(rec.get("trigger_pos", -2) or -2)
                    rec_layer = int(rec.get("trigger_layer", -2) or -2)
                    if expected_trigger_pos >= 0 and rec_pos != expected_trigger_pos:
                        continue
                    if expected_trigger_layer >= 0 and rec_layer < expected_trigger_layer:
                        continue
                    events.append(
                        {
                            "event": "plan_observed_applied",
                            "command_index": command_index,
                            "event_id": event_id,
                            "trigger_pos": rec_pos,
                            "trigger_layer": rec_layer,
                        }
                    )
                    return events
            time.sleep(0.2)
        return events


class DllamaBackend(EdgeVisorBackend):
    name = "dllama"

    def __init__(
        self,
        cuda_visible: str = "0,1,2",
        ctx: int = 2048,
        steps: int = 128,
        timeout_s: int = 240,
        ratios: str = "1:1",
        worker_gpus: Optional[List[int]] = None,
        root_gpu: int = 0,
    ):
        # The upstream distributed-llama runtime uses tensor-style splitting.
        # On the 3B Llama model used here, the 2-worker case has a kvDim
        # divisibility assertion; the previously verified stable setup is one
        # root plus one worker.
        super().__init__(
            cuda_visible=cuda_visible,
            ctx=ctx,
            steps=steps,
            timeout_s=timeout_s,
            ratios="",
            worker_gpus=worker_gpus if worker_gpus is not None else [1],
            backend_name=self.name,
            project_dir=DLLAMA_PROJECT,
            engine_dir=DLLAMA_ENGINE,
            dllama_path=DLLAMA_CLI,
            model_path=DLLAMA_MODEL,
            tokenizer_path=DLLAMA_TOKENIZER,
            vulkan_project=DLLAMA_PROJECT,
            root_gpu=root_gpu,
            enable_benchmark=False,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        return super().generate(messages, max_tokens, generation_name, out_dir, dynamic_plan=None)


class EdgeVisorExoBackend(EdgeVisorBackend):
    name = "edgevisor_exo"

    def __init__(
        self,
        cuda_visible: str = "0,1,2",
        ctx: int = 2048,
        steps: int = 128,
        timeout_s: int = 240,
        gpu_indices: Optional[List[int]] = None,
        total_layers: int = 28,
        memory_field: str = "total",
    ):
        ensure_paths([EDGE_EXO_DLLAMA, EDGE_EXO_MODEL, EDGE_EXO_TOKENIZER, EDGE_EXO_PLAN])
        self.gpu_indices = gpu_indices if gpu_indices is not None else [0, 1, 2]
        if not self.gpu_indices:
            raise BackendError("edgevisor_exo requires at least one GPU index")
        self.total_layers = total_layers
        self.memory_field = memory_field
        ratios, plan = self._make_exo_plan()
        super().__init__(
            cuda_visible=cuda_visible,
            ctx=ctx,
            steps=steps,
            timeout_s=timeout_s,
            ratios=ratios,
            worker_gpus=self.gpu_indices[1:],
            backend_name=self.name,
            project_dir=EDGE_EXO_PROJECT,
            engine_dir=EDGE_EXO_ENGINE,
            dllama_path=EDGE_EXO_DLLAMA,
            model_path=EDGE_EXO_MODEL,
            tokenizer_path=EDGE_EXO_TOKENIZER,
            vulkan_project=EDGE_EXO_PROJECT,
            root_gpu=self.gpu_indices[0],
        )
        self.last_plan = plan

    def _make_exo_plan(self) -> tuple[str, Dict[str, Any]]:
        cmd = [
            sys.executable,
            str(EDGE_EXO_PLAN),
            "--gpu-indices",
            ",".join(str(x) for x in self.gpu_indices),
            "--total-layers",
            str(self.total_layers),
            "--memory-field",
            self.memory_field,
            "--json",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(EDGE_EXO_ENGINE),
            env=base_env(",".join(str(x) for x in self.gpu_indices), EDGE_EXO_PROJECT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        if proc.returncode != 0:
            raise BackendError(f"edgevisor_exo plan failed:\n{proc.stderr.strip()}")
        plan = json.loads(proc.stdout)
        ratios = str(plan.get("ratios", ""))
        if not ratios:
            raise BackendError(f"edgevisor_exo plan did not produce ratios: {plan}")
        return ratios, plan


class EdgeVisorLinguaLinkedBackend(EdgeVisorBackend):
    name = "edgevisor_lingualinked"

    def __init__(
        self,
        cuda_visible: str = "0,1,2",
        ctx: int = 2048,
        steps: int = 128,
        timeout_s: int = 240,
        gpu_indices: Optional[List[int]] = None,
        total_layers: int = 28,
        overlap_layers: int = 2,
        memory_field: str = "total",
    ):
        ensure_paths([EDGE_LL_DLLAMA, EDGE_LL_MODEL, EDGE_LL_TOKENIZER, EDGE_LL_PLAN])
        self.gpu_indices = gpu_indices if gpu_indices is not None else [0, 1, 2]
        if not self.gpu_indices:
            raise BackendError("edgevisor_lingualinked requires at least one GPU index")
        self.total_layers = total_layers
        self.overlap_layers = max(0, overlap_layers)
        self.memory_field = memory_field
        self.current_plan = self._make_plan()
        self.current_layers = list(self.current_plan["layers"])
        ratios = str(self.current_plan["ratios"])
        super().__init__(
            cuda_visible=cuda_visible,
            ctx=ctx,
            steps=steps,
            timeout_s=timeout_s,
            ratios=ratios,
            worker_gpus=self.gpu_indices[1:],
            backend_name=self.name,
            project_dir=EDGE_LL_PROJECT,
            engine_dir=EDGE_LL_ENGINE,
            dllama_path=EDGE_LL_DLLAMA,
            model_path=EDGE_LL_MODEL,
            tokenizer_path=EDGE_LL_TOKENIZER,
            vulkan_project=EDGE_LL_PROJECT,
            root_gpu=self.gpu_indices[0],
            extra_root_args=[
                "--runtime-redundant-boundary-layers",
                str(self.overlap_layers),
                "--runtime-active-seg-enabled",
                "1",
                "--runtime-redundant-seg-enabled",
                "0",
                "--enable-kv-redundancy-during-migration",
                "0",
                "--kv-redundancy",
                "0",
            ],
            extra_env={
                "DLLAMA_LINGUALINKED_MODE": "1",
                "DLLAMA_RUNTIME_PLAN_PRINT": "1",
            },
        )
        self.last_plan = self.current_plan

    def _make_plan(self, stage_ms: Optional[List[float]] = None) -> Dict[str, Any]:
        cmd = [
            sys.executable,
            str(EDGE_LL_PLAN),
            "--gpu-indices",
            ",".join(str(x) for x in self.gpu_indices),
            "--total-layers",
            str(self.total_layers),
            "--overlap-layers",
            str(self.overlap_layers),
            "--memory-field",
            self.memory_field,
            "--json",
        ]
        if stage_ms is not None:
            cmd.extend(["--current-layers", ",".join(str(x) for x in self.current_layers)])
            cmd.extend(["--stage-ms", ",".join(f"{x:.6f}" for x in stage_ms)])
        proc = subprocess.run(
            cmd,
            cwd=str(EDGE_LL_ENGINE),
            env=base_env(",".join(str(x) for x in self.gpu_indices), EDGE_LL_PROJECT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        if proc.returncode != 0:
            raise BackendError(f"edgevisor_lingualinked plan failed:\n{proc.stderr.strip()}")
        plan = json.loads(proc.stdout)
        if not plan.get("ratios"):
            raise BackendError(f"edgevisor_lingualinked plan did not produce ratios: {plan}")
        return plan

    def _extract_stage_ms(self, result: GenerationResult) -> Optional[List[float]]:
        stage_profile = result.metrics.get("stage_profile")
        if isinstance(stage_profile, list) and len(stage_profile) >= len(self.gpu_indices):
            values: List[float] = []
            for entry in stage_profile[: len(self.gpu_indices)]:
                if not isinstance(entry, dict):
                    return None
                value = entry.get("per_tok_total_ms") or entry.get("per_fwd_total_ms")
                if value is None or float(value) <= 0.0:
                    return None
                values.append(float(value))
            return values

        node_metrics = result.metrics.get("node_metrics")
        if not isinstance(node_metrics, list) or len(node_metrics) < len(self.gpu_indices):
            return None
        values: List[float] = []
        for node in node_metrics[: len(self.gpu_indices)]:
            m = node.get("metrics", {}) if isinstance(node, dict) else {}
            if not isinstance(m, dict):
                return None
            value = m.get("prediction_ms_total")
            if value is None:
                eval_ms = m.get("eval_ms")
                pred_ms = m.get("prediction_ms_total")
                if eval_ms is not None or pred_ms is not None:
                    value = float(eval_ms or 0.0) + float(pred_ms or 0.0)
            if value is None or float(value) <= 0.0:
                return None
            values.append(float(value))
        return values

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        used_plan = dict(self.current_plan)
        self.ratios = str(used_plan["ratios"])
        self.last_plan = used_plan
        result = super().generate(messages, max_tokens, generation_name, out_dir, dynamic_plan=None)
        result.metrics["lingualinked_plan"] = used_plan
        stage_ms = self._extract_stage_ms(result)
        if stage_ms is not None:
            next_plan = self._make_plan(stage_ms=stage_ms)
            self.current_plan = next_plan
            self.current_layers = list(next_plan["layers"])
            result.metrics["lingualinked_stage_ms"] = stage_ms
            result.metrics["next_lingualinked_plan"] = next_plan
            result.dynamic_events.append(
                {
                    "event": "lingualinked_request_rebalance",
                    "stage_ms": stage_ms,
                    "from_layers": used_plan.get("layers"),
                    "to_layers": next_plan.get("layers"),
                    "from_ratios": used_plan.get("ratios"),
                    "to_ratios": next_plan.get("ratios"),
                }
            )
        else:
            result.metrics["lingualinked_stage_ms"] = None
            result.dynamic_events.append(
                {
                    "event": "lingualinked_request_rebalance_skipped",
                    "reason": "missing per-stage benchmark metrics",
                    "layers": used_plan.get("layers"),
                    "ratios": used_plan.get("ratios"),
                }
            )
        return result


class EdgeVisorAblationBackend(EdgeVisorBackend):
    name = "edgevisor_ablation"

    def __init__(
        self,
        cuda_visible: str = "0,1,2",
        ctx: int = 2048,
        steps: int = 128,
        timeout_s: int = 240,
        ratios: str = "1:1",
        worker_gpus: Optional[List[int]] = None,
        ablation_config: Optional[Dict[str, Any]] = None,
        persistent: bool = True,
        api_port: int = 0,
        virtual_topology: Optional[Dict[str, Any]] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ):
        config = dict(ablation_config or {})
        super().__init__(
            cuda_visible=cuda_visible,
            ctx=ctx,
            steps=steps,
            timeout_s=timeout_s,
            ratios=ratios,
            worker_gpus=worker_gpus if worker_gpus is not None else [1],
            backend_name=self.name,
            ablation_config=config,
            virtual_topology=virtual_topology,
            extra_env=extra_env,
            enable_benchmark=False,
        )
        self.persistent = persistent
        self.api_port = api_port
        self._persistent_started = False
        self._persistent_procs: List[subprocess.Popen[Any]] = []
        self._persistent_logs: Dict[str, Path] = {}
        self._persistent_log_offsets: Dict[str, int] = {}
        self._persistent_ablation_log_path: Optional[Path] = None
        self._persistent_ablation_offset = 0
        self._persistent_socket_path: Optional[str] = None
        self._persistent_root_cmd: List[str] = []
        self._persistent_worker_cmds: List[List[str]] = []
        self._persistent_proxies: List[TcpRateProxy] = []
        self._persistent_out_dir: Optional[Path] = None
        self._session_started_at_ms = 0.0
        self._session_generation_count = 0

    def close(self) -> None:
        for proxy in list(self._persistent_proxies):
            proxy.stop()
        self._persistent_proxies = []
        if not self._persistent_procs:
            return
        terminate_all(self._persistent_procs)
        close_logs(self._persistent_procs)
        self._persistent_procs = []
        self._persistent_started = False
        if self._persistent_socket_path:
            try:
                os.unlink(self._persistent_socket_path)
            except FileNotFoundError:
                pass
            self._persistent_socket_path = None

    def _persistent_env(self) -> Dict[str, str]:
        env = base_env(self.cuda_visible, self.vulkan_project)
        env.update(self.extra_env)
        return env

    def _choose_api_port(self) -> int:
        if self.api_port > 0:
            return self.api_port
        for _ in range(50):
            port = 35000 + random.randint(0, 2000)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise BackendError("could not allocate EdgeVisor API port")

    def _http_json(self, method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        if self.api_port <= 0:
            raise BackendError("persistent API port is not initialized")
        data = None
        headers: Dict[str, str] = {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}
        conn = http.client.HTTPConnection("127.0.0.1", self.api_port, timeout=timeout_s or self.timeout_s)
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
        finally:
            conn.close()
        if resp.status < 200 or resp.status >= 300:
            raise BackendError(f"EdgeVisor API HTTP {resp.status}: {raw[:500]}")
        return json.loads(raw) if raw.strip() else {}

    def _wait_api_ready(self, root_proc: subprocess.Popen[Any]) -> None:
        deadline = time.time() + 120.0
        last_error = ""
        while time.time() < deadline:
            if root_proc.poll() is not None:
                text = self._persistent_logs.get("root", Path("/nonexistent")).read_text(encoding="utf-8", errors="replace") if self._persistent_logs.get("root") else ""
                raise BackendError(f"EdgeVisor API exited while starting, rc={root_proc.returncode}\n{text[-2000:]}")
            try:
                self._http_json("GET", "/v1/models", timeout_s=2.0)
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.5)
        raise BackendError(f"EdgeVisor API did not become ready: {last_error}")

    def _start_persistent_session(self, out_dir: Path) -> None:
        if self._persistent_started:
            return
        ensure_paths([self.api_path, self.dllama_path, self.model_path, self.tokenizer_path])
        out_dir.mkdir(parents=True, exist_ok=True)
        self.api_port = self._choose_api_port()
        port_base = self._choose_port_base()
        worker_ports, root_worker_ports, proxies = self._make_worker_ports(port_base, out_dir, "persistent_edgevisor")
        socket_path = f"/tmp/edgevisor_agent_persistent_{os.getpid()}_{port_base}.sock"
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        self._persistent_socket_path = socket_path
        self._persistent_out_dir = out_dir
        self._persistent_ablation_log_path = out_dir / "persistent_ablation.jsonl"
        self._persistent_ablation_offset = 0
        self._persistent_logs = {"root": out_dir / "persistent_edgevisor_api_root.log"}
        for idx, gpu in enumerate(self.worker_gpus):
            self._persistent_logs[f"worker{idx}"] = self._worker_log_path(out_dir, "persistent_edgevisor", idx, gpu)
        self._persistent_log_offsets = {name: 0 for name in self._persistent_logs}

        env = self._persistent_env()
        env["DLLAMA_PLAN_CTRL_SOCKET"] = socket_path
        env["DLLAMA_ENABLE_PLAN_BARRIER"] = "1"
        layer_prof_path = out_dir / "persistent_layer_prof_stage0_root0.bin"
        env["DLLAMA_LAYER_PROF_PATH"] = str(layer_prof_path)
        env["DLLAMA_DYN_LAYER_PROF_PATH"] = str(layer_prof_path)

        procs: List[subprocess.Popen[Any]] = []
        worker_cmds: List[List[str]] = []
        try:
            launch_stagger_s = float(self.virtual_topology.get("launch_stagger_s", 0.0)) if self.virtual_topology else 0.0
            for idx, gpu in enumerate(self.worker_gpus):
                worker_cmd = [
                    str(self.dllama_path),
                    "worker",
                    "--port",
                    str(worker_ports[idx]),
                    "--nthreads",
                    "1",
                    "--gpu-index",
                    str(gpu),
                ]
                worker_cmds.append(worker_cmd)
                procs.append(self._launch_worker(worker_cmd, self._persistent_logs[f"worker{idx}"], env))
                if launch_stagger_s > 0.0 and idx + 1 < len(self.worker_gpus):
                    time.sleep(launch_stagger_s)
            if self.worker_gpus:
                time.sleep(3.0)

            root_cmd = [
                str(self.api_path),
                "--port",
                str(self.api_port),
                "--model",
                str(self.model_path),
                "--tokenizer",
                str(self.tokenizer_path),
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
                str(self.root_gpu),
            ]
            if self.enable_benchmark:
                root_cmd.append("--benchmark")
            if self.worker_gpus:
                root_cmd.extend(["--workers", *[f"127.0.0.1:{p}" for p in root_worker_ports]])
                if self.ratios:
                    root_cmd.extend(["--ratios", self.ratios])
            root_cmd.extend(self._plan_barrier_args())
            cfg = self.ablation_config
            root_cmd.extend(["--ablation-log-path", str(self._persistent_ablation_log_path)])
            root_cmd.extend(["--experiment-id", str(cfg.get("experiment_id", "persistent_episode"))])
            root_cmd.extend(["--shadow-kv-mode", str(cfg.get("shadow_kv_mode", "enabled"))])
            root_cmd.extend(["--pointer-swizzling-mode", str(cfg.get("pointer_swizzling_mode", "enabled"))])
            root_cmd.extend(["--jit-mode", str(cfg.get("jit_mode", "enabled"))])
            root_cmd.extend(["--vg-mode", str(cfg.get("vg_mode", "enabled"))])
            root_cmd.extend(["--fallback-policy", str(cfg.get("fallback_policy", "disabled_unless_necessary"))])
            if cfg.get("config_path"):
                root_cmd.extend(["--edgevisor-ablation-config", str(cfg["config_path"])])
            if self.extra_root_args:
                root_cmd.extend(self.extra_root_args)

            started = time.perf_counter()
            root_proc = popen_log(root_cmd, self._persistent_logs["root"], self.engine_dir, env)
            procs.append(root_proc)
            self.api_port = int(self.api_port)
            self._wait_api_ready(root_proc)
            if not bool((self.ablation_config or {}).get("network_proxy_start_throttled", True)):
                for proxy in proxies:
                    proxy.activate()
            self._session_started_at_ms = (time.perf_counter() - started) * 1000.0
            self._persistent_procs = procs
            self._persistent_worker_cmds = worker_cmds
            self._persistent_proxies = proxies
            self._persistent_root_cmd = root_cmd
            self._persistent_started = True
        except Exception:
            for proxy in proxies:
                proxy.stop()
            terminate_all(procs)
            close_logs(procs)
            if self._persistent_socket_path:
                try:
                    os.unlink(self._persistent_socket_path)
                except FileNotFoundError:
                    pass
                self._persistent_socket_path = None
            raise

    def _collect_persistent_metrics(self, wall_ms: float, rc: int) -> Dict[str, Any]:
        root_delta = ""
        node_metrics: List[Dict[str, Any]] = []
        cache_hit = False
        for label, path in self._persistent_logs.items():
            text, new_offset = read_text_since(path, self._persistent_log_offsets.get(label, 0))
            self._persistent_log_offsets[label] = new_offset
            if label == "root":
                root_delta = text
                cache_hit = "Found naive cache" in text
            node_metrics.append(self._node_metric_entry(label, path, text))
        metrics = parse_edge_metrics(root_delta)
        metrics.update(
            {
                "wall_ms": wall_ms,
                "valid": rc == 0,
                "persistent_session": True,
                "persistent_cache_hit": cache_hit,
                "persistent_session_start_ms": self._session_started_at_ms,
                "persistent_generation_index": self._session_generation_count,
                "api_port": self.api_port,
                "node_metrics": node_metrics,
                "ablation_config": dict(self.ablation_config),
                "ablation_effective_topology": dict(self.ablation_effective_topology),
            }
        )
        if self._persistent_ablation_log_path is not None:
            records, new_offset = jsonl_records_since(self._persistent_ablation_log_path, self._persistent_ablation_offset)
            self._persistent_ablation_offset = new_offset
            metrics["ablation_log_path"] = str(self._persistent_ablation_log_path)
            metrics["ablation_events"] = records
        return metrics

    def _generate_persistent(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        self._start_persistent_session(out_dir)
        self._session_generation_count += 1
        dynamic_plan = self._ablation_dynamic_plan(dynamic_plan)
        dynamic_events: List[Dict[str, Any]] = []
        plan_thread: Optional[threading.Thread] = None
        plan_box: Dict[str, Any] = {"events": []}
        if dynamic_plan and dynamic_plan.get("enabled", True) and self._persistent_socket_path and self._persistent_procs:
            root_proc = self._persistent_procs[-1]

            def drive_plan() -> None:
                plan_box["events"] = self._drive_dynamic_plan(
                    self._persistent_socket_path or "",
                    root_proc,
                    dynamic_plan or {},
                    self._persistent_ablation_log_path,
                )

            plan_thread = threading.Thread(target=drive_plan, daemon=True)
            plan_thread.start()

        payload = {
            "model": "edgevisor-ablation-persistent",
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": 0,
            "seed": 1,
            "stream": False,
        }
        start = time.perf_counter()
        rc = 0
        content = ""
        try:
            resp = self._http_json("POST", "/v1/chat/completions", payload, timeout_s=self.timeout_s)
            choices = resp.get("choices", []) if isinstance(resp, dict) else []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    content = str(message.get("content", ""))
        except Exception as exc:
            rc = 1
            content = f"[edgevisor persistent api error] {exc}"
        wall_ms = (time.perf_counter() - start) * 1000.0
        if plan_thread is not None:
            plan_thread.join(timeout=float((dynamic_plan or {}).get("consume_timeout_s", 20.0)) + 5.0)
            dynamic_events = list(plan_box.get("events", []))
        metrics = self._collect_persistent_metrics(wall_ms=wall_ms, rc=rc)
        if self.last_plan is not None:
            metrics["exo_plan"] = self.last_plan
        return GenerationResult(
            backend=self.name,
            content=content.strip(),
            metrics=metrics,
            command=list(self._persistent_root_cmd),
            log_path=str(self._persistent_logs.get("root", "")),
            dynamic_events=dynamic_events,
            rc=rc,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        generation_name: str,
        out_dir: Path,
        dynamic_plan: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        if not self.persistent:
            result = super().generate(messages, max_tokens, generation_name, out_dir, dynamic_plan=dynamic_plan)
            result.metrics["persistent_session"] = False
            return result
        return self._generate_persistent(messages, max_tokens, generation_name, out_dir, dynamic_plan=dynamic_plan)


def make_backend(name: str, **kwargs: Any) -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "prima":
        return PrimaBackend(**kwargs)
    if name == "edgevisor":
        return EdgeVisorBackend(**kwargs)
    if name == "dllama":
        return DllamaBackend(**kwargs)
    if name == "edgevisor_exo":
        return EdgeVisorExoBackend(**kwargs)
    if name == "edgevisor_lingualinked":
        return EdgeVisorLinguaLinkedBackend(**kwargs)
    if name == "edgevisor_ablation":
        return EdgeVisorAblationBackend(**kwargs)
    raise BackendError(f"unknown backend: {name}")

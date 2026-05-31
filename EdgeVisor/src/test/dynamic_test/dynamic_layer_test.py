
#!/usr/bin/env python3

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def uds_request(sock_path: str, req: dict, timeout_s: float = 2.0) -> dict:
	s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	s.settimeout(timeout_s)
	try:
		s.connect(sock_path)
		s.sendall((json.dumps(req) + "\n").encode("utf-8"))
		buf = bytearray()
		while True:
			chunk = s.recv(4096)
			if not chunk:
				break
			buf.extend(chunk)
			if b"\n" in chunk:
				break
		line = bytes(buf).split(b"\n", 1)[0].decode("utf-8", errors="replace")
		return json.loads(line)
	finally:
		try:
			s.close()
		except Exception:
			pass


def wait_until(deadline_s: float, interval_s: float, fn, desc: str):
	last_err = None
	while time.time() < deadline_s:
		try:
			v = fn()
			if v:
				return v
		except Exception as e:
			last_err = e
		time.sleep(interval_s)
	if last_err is not None:
		raise RuntimeError(f"timeout waiting for {desc}: last error={last_err}")
	raise RuntimeError(f"timeout waiting for {desc}")


def popen_safe(argv, env=None):
	return subprocess.Popen(
		argv,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		bufsize=1,
	)


def terminate_proc(p: subprocess.Popen, name: str, timeout_s: float = 5.0):
	if p is None:
		return
	if p.poll() is not None:
		return
	try:
		p.send_signal(signal.SIGINT)
	except Exception:
		try:
			p.terminate()
		except Exception:
			return
	t0 = time.time()
	while time.time() - t0 < timeout_s:
		if p.poll() is not None:
			return
		time.sleep(0.1)
	try:
		p.kill()
	except Exception:
		pass


def main() -> int:
	ap = argparse.ArgumentParser(
		description=(
			"Integration test for dynamic-layer background scheduler. "
			"It starts worker+inference, waits for layer_prof snapshots, "
			"and verifies set_plan is issued/consumed via UDS."
		)
	)
	ap.add_argument("--dllama", default="./dllama", help="Path to dllama binary")
	ap.add_argument("--socket", default="/tmp/dllama_plan.sock", help="UDS plan controller socket path")
	ap.add_argument("--worker-port", type=int, default=9999, help="Worker listen port")
	ap.add_argument("--nthreads", type=int, default=2, help="Threads for worker/inference")
	ap.add_argument("--steps", type=int, default=128, help="Inference steps")
	ap.add_argument("--prompt", default="The capital of France is", help="Prompt")
	ap.add_argument("--stage", type=int, default=0, help="Stage index")
	ap.add_argument("--root-node", type=int, default=0, help="Stage root node index")
	ap.add_argument("--model", default=os.environ.get("DLLAMA_TEST_MODEL", ""), help="Model path (or env DLLAMA_TEST_MODEL)")
	ap.add_argument(
		"--tokenizer",
		default=os.environ.get("DLLAMA_TEST_TOKENIZER", ""),
		help="Tokenizer path (or env DLLAMA_TEST_TOKENIZER)",
	)
	ap.add_argument("--timeout", type=float, default=60.0, help="Overall timeout seconds")
	args = ap.parse_args()

	dllama = str(Path(args.dllama))
	sock_path = args.socket

	if not args.model or not args.tokenizer:
		sys.stderr.write(
			"ERROR: missing --model/--tokenizer. Provide CLI flags or set env: DLLAMA_TEST_MODEL, DLLAMA_TEST_TOKENIZER\n"
		)
		return 2

	# Clean old socket
	try:
		os.unlink(sock_path)
	except FileNotFoundError:
		pass

	worker = None
	inf = None
	t_deadline = time.time() + float(args.timeout)

	try:
		worker_argv = [
			dllama,
			"worker",
			"--port",
			str(args.worker_port),
			"--nthreads",
			str(args.nthreads),
		]
		worker = popen_safe(worker_argv, env=os.environ.copy())

		env = os.environ.copy()
		env["DLLAMA_PLAN_CTRL_SOCKET"] = sock_path
		env["DLLAMA_ENABLE_PLAN_BARRIER"] = "1"

		# Enable internal scheduler thread and force one optimization.
		env["DLLAMA_DYNAMIC_LAYER_ENABLE"] = "1"
		env["DLLAMA_DYN_FORCE_TRIGGER"] = "1"
		env["DLLAMA_DYN_STAGE_INDEX"] = str(args.stage)
		env["DLLAMA_DYN_ROOT_NODE_INDEX"] = str(args.root_node)
		env.setdefault("DLLAMA_DYN_POLL_MS", "200")
		env.setdefault("DLLAMA_DYN_UDS_TIMEOUT_MS", "2000")
		env.setdefault("DLLAMA_DYN_DROP_RATIO", "0.30")

		inf_argv = [
			dllama,
			"inference",
			"--prompt",
			args.prompt,
			"--steps",
			str(args.steps),
			"--model",
			args.model,
			"--tokenizer",
			args.tokenizer,
			"--buffer-float-type",
			"q80",
			"--enable-plan-barrier",
			"--kv-redundancy",
			"2",
			"--nthreads",
			str(args.nthreads),
			"--max-seq-len",
			"256",
			"--benchmark",
			"--workers",
			f"127.0.0.1:{args.worker_port}",
			"--ratios",
			"1:1",
		]
		inf = popen_safe(inf_argv, env=env)

		# Wait for UDS ready
		def ping_ok():
			resp = uds_request(sock_path, {"op": "ping"}, timeout_s=1.0)
			return resp.get("ok") is True

		wait_until(t_deadline, 0.2, ping_ok, "UDS ping")

		# Read initial cacheSeq
		st0 = uds_request(sock_path, {"op": "status"}, timeout_s=1.0)
		if not st0.get("ok"):
			raise RuntimeError(f"status failed: {st0}")
		cache_seq0 = int(st0.get("cacheSeq", 0))

		# Wait for layer_prof epoch > 0 (snapshot file produced)
		def got_layer_prof():
			req = {
				"op": "layer_prof",
				"all": True,
				"stageIndex": int(args.stage),
				"rootNodeIndex": int(args.root_node),
			}
			resp = uds_request(sock_path, req, timeout_s=2.0)
			if not resp.get("ok"):
				return False
			lp = resp.get("layer_prof")
			if not isinstance(lp, dict):
				return False
			return int(lp.get("epoch", 0)) > 0

		wait_until(t_deadline, 0.2, got_layer_prof, "layer_prof epoch")

		# Wait for internal scheduler to issue at least one set_plan (cacheSeq increments)
		def cache_seq_bumped():
			st = uds_request(sock_path, {"op": "status"}, timeout_s=1.0)
			if not st.get("ok"):
				return False
			return int(st.get("cacheSeq", 0)) > cache_seq0

		wait_until(t_deadline, 0.2, cache_seq_bumped, "cacheSeq increment (set_plan issued)")

		# Best-effort: wait for command to be consumed (mode returns to none)
		def cmd_consumed():
			st = uds_request(sock_path, {"op": "status"}, timeout_s=1.0)
			if not st.get("ok"):
				return False
			cmd = st.get("cmd")
			if not isinstance(cmd, dict):
				return False
			return cmd.get("mode") == "none"

		wait_until(t_deadline, 0.2, cmd_consumed, "cmd consumption at next barrier")

		sys.stdout.write("PASS: dynamic-layer issued set_plan and it was consumed at barrier.\n")
		return 0

	finally:
		terminate_proc(inf, "inference")
		terminate_proc(worker, "worker")
		try:
			os.unlink(sock_path)
		except Exception:
			pass


if __name__ == "__main__":
	raise SystemExit(main())


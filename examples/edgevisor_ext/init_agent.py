#!/usr/bin/env python3
import argparse
import json
import os
import socket
import subprocess
import time
from urllib import request

from schemas import now_s, validate_agent_report


def detect_memory_bytes():
    # Linux: /proc/meminfo
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except Exception:
        pass

    # macOS fallback
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    except Exception:
        pass

    return 8 * 1024 * 1024 * 1024


def parse_links(spec):
    if not spec:
        return []
    links = []
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    for item in parts:
        if ":" not in item:
            raise ValueError(f"bad link spec: {item}; expect dst:gbps")
        dst, bw = item.split(":", 1)
        links.append({"dst_id": int(dst), "bandwidth_gbps": float(bw)})
    return links


def post_json(url, payload, timeout_s=3.0):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {"ok": True}


def main():
    ap = argparse.ArgumentParser(description="EdgeVisor init-agent: collect local profile and report to init-root.")
    ap.add_argument("--node-id", type=int, required=True, help="global node id")
    ap.add_argument("--root-url", required=True, help="init-root report endpoint, e.g. http://127.0.0.1:18080/report")
    ap.add_argument("--compute-flops", type=float, default=1e12, help="device compute FLOPS")
    ap.add_argument("--memory-bytes", type=int, default=0, help="device memory bytes (0=auto detect)")
    ap.add_argument("--host", default=socket.gethostname(), help="host label")
    ap.add_argument("--links", default="", help="comma list: dst:gbps,dst:gbps")
    ap.add_argument("--retries", type=int, default=20)
    ap.add_argument("--retry-interval-s", type=float, default=1.0)
    args = ap.parse_args()

    memory_bytes = args.memory_bytes if args.memory_bytes > 0 else detect_memory_bytes()
    report = {
        "node_id": int(args.node_id),
        "host": args.host,
        "compute_flops": float(args.compute_flops),
        "memory_bytes": int(memory_bytes),
        "links": parse_links(args.links),
        "timestamp": now_s(),
    }
    validate_agent_report(report)

    last_err = None
    for _ in range(max(1, args.retries)):
        try:
            resp = post_json(args.root_url, report)
            if resp.get("ok"):
                print(json.dumps({"ok": True, "reported_node": args.node_id, "resp": resp}, ensure_ascii=False))
                return 0
            last_err = RuntimeError(f"server rejected: {resp}")
        except Exception as e:
            last_err = e
        time.sleep(max(0.1, args.retry_interval_s))

    print(json.dumps({"ok": False, "error": str(last_err)}, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


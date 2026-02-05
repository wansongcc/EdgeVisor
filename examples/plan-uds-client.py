#!/usr/bin/env python3
import argparse
import json
import socket
import sys
from typing import Any, Dict


def uds_request(socket_path: str, req: Dict[str, Any], timeout_s: float = 2.0) -> Dict[str, Any]:
    data = (json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        s.connect(socket_path)
        s.sendall(data)
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                buf = buf.split(b"\n", 1)[0]
                break
    if not buf:
        raise RuntimeError("empty response")
    return json.loads(buf.decode("utf-8"))


def cmd_set_plan(args: argparse.Namespace) -> Dict[str, Any]:
    cmd: Dict[str, Any] = {
        "seq": args.seq,
        "mode": args.mode,
        "stageIndex": args.stage,
        "fromNodeIndex": args.from_node,
        "toNodeIndex": args.to_node,
        "cmdKind": args.kind,
        "nHeadsToMove": args.heads,
        "nFfnToMove": args.ffn,
    }
    if args.mode == "exact":
        if args.trigger_pos is None or args.trigger_layer is None:
            raise SystemExit("exact mode requires --trigger-pos and --trigger-layer")
        cmd["triggerPos"] = args.trigger_pos
        cmd["triggerLayer"] = args.trigger_layer
    return {"op": "set_plan", "cmd": cmd}


def main() -> int:
    p = argparse.ArgumentParser(description="dllama plan UDS controller client (JSON line protocol)")
    p.add_argument("socket", help="UDS path, e.g. /tmp/dllama_plan.sock")

    sub = p.add_subparsers(dest="op", required=True)

    sub.add_parser("ping")
    sub.add_parser("status")
    sub.add_parser("perf")
    sub.add_parser("clear")

    sp = sub.add_parser("set_plan")
    sp.add_argument("--seq", type=int, default=1)
    sp.add_argument("--mode", choices=["exact", "next_barrier", "next"], default="next_barrier")
    sp.add_argument("--stage", type=int, default=0)
    sp.add_argument("--from", dest="from_node", type=int, default=0)
    sp.add_argument("--to", dest="to_node", type=int, default=1)
    sp.add_argument("--kind", type=int, default=3, help="1=headSplit 2=ffnSplit 3=both")
    sp.add_argument("--heads", type=int, default=1)
    sp.add_argument("--ffn", type=int, default=256)
    sp.add_argument("--trigger-pos", type=int, default=None)
    sp.add_argument("--trigger-layer", type=int, default=None)

    raw = sub.add_parser("raw")
    raw.add_argument("json", help='raw request JSON, e.g. {"op":"ping"}')

    args = p.parse_args()

    if args.op == "raw":
        req = json.loads(args.json)
    elif args.op == "set_plan":
        req = cmd_set_plan(args)
    else:
        req = {"op": args.op}

    resp = uds_request(args.socket, req)
    json.dump(resp, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if resp.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

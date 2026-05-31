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
    }

    # v2 multi-move mode: --move FROM,TO,KIND,HEADS,FFN (repeatable)
    # Examples:
    #   --move 0,1,3,1,256 --move 2,1,3,1,256 --move 3,2,3,1,256
    if args.move:
        moves = []
        for spec in args.move:
            parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
            if len(parts) not in (2, 3, 4, 5):
                raise SystemExit("--move expects FROM,TO[,KIND[,HEADS[,FFN]]]")
            frm = int(parts[0])
            to = int(parts[1])
            kind = int(parts[2]) if len(parts) >= 3 else int(args.kind)
            heads = int(parts[3]) if len(parts) >= 4 else int(args.heads)
            ffn = int(parts[4]) if len(parts) >= 5 else int(args.ffn)
            moves.append({
                "fromNodeIndex": frm,
                "toNodeIndex": to,
                "cmdKind": kind,
                "headMove": heads,
                "ffnMove": ffn,
            })
        cmd["moves"] = moves
    else:
        # Legacy single-edge mode
        cmd["fromNodeIndex"] = args.from_node
        cmd["toNodeIndex"] = args.to_node
        cmd["cmdKind"] = args.kind
        cmd["nHeadsToMove"] = args.heads
        cmd["nFfnToMove"] = args.ffn
    if args.mode == "exact":
        if args.trigger_pos is None or args.trigger_layer is None:
            raise SystemExit("exact mode requires --trigger-pos and --trigger-layer")
        cmd["triggerPos"] = args.trigger_pos
        cmd["triggerLayer"] = args.trigger_layer
    return {"op": "set_plan", "cmd": cmd}


def cmd_set_pp_migration(args: argparse.Namespace) -> Dict[str, Any]:
    if args.layer_count < 1:
        raise SystemExit("--layer-count must be >= 1")
    cmd: Dict[str, Any] = {
        "seq": args.seq,
        "mode": args.mode,
        "stageIndex": args.stage,
        "fromNodeIndex": args.from_node,
        "toNodeIndex": args.to_node,
        "layerCount": args.layer_count,
    }
    if args.mode == "exact":
        if args.trigger_pos is None:
            raise SystemExit("exact mode requires --trigger-pos")
        cmd["triggerPos"] = args.trigger_pos
        if args.trigger_layer is not None:
            cmd["triggerLayer"] = args.trigger_layer
    return {"op": "set_pp_migration", "cmd": cmd}


def cmd_layer_prof(args: argparse.Namespace) -> Dict[str, Any]:
    req: Dict[str, Any] = {"op": "layer_prof"}
    if args.path:
        req["path"] = args.path
    if args.all:
        req["all"] = True
    else:
        req["layerIndex"] = args.layer
    if args.stage is not None:
        req["stageIndex"] = args.stage
    if args.root is not None:
        req["rootNodeIndex"] = args.root
    return req


def pick_min_max_nodes(nodes: Any) -> Any:
    # nodes is a list of dicts: {ok,nodeIndex,attnUs,ffnUs,...}
    ok_nodes = [n for n in nodes if isinstance(n, dict) and n.get("ok")]
    if not ok_nodes:
        return None
    # score = attn + ffn
    def score(n: Dict[str, Any]) -> int:
        return int(n.get("attnUs", 0)) + int(n.get("ffnUs", 0))
    slow = max(ok_nodes, key=score)
    fast = min(ok_nodes, key=score)
    return fast, slow, score(fast), score(slow)


def main() -> int:
    p = argparse.ArgumentParser(description="dllama plan UDS controller client (JSON line protocol)")
    p.add_argument("socket", help="UDS path, e.g. /tmp/dllama_plan.sock")

    sub = p.add_subparsers(dest="op", required=True)

    sub.add_parser("ping")
    sub.add_parser("status")
    sub.add_parser("perf")
    sub.add_parser("clear")
    lp = sub.add_parser("layer_prof", help="query layer-prof snapshot via UDS")
    lp.add_argument("--path", default=None, help="snapshot path (overrides env/default)")
    lp.add_argument("--all", action="store_true", help="return full table (can be large)")
    lp.add_argument("--layer", type=int, default=0, help="layer index (when --all is not set)")
    lp.add_argument("--stage", type=int, default=None, help="stage index for default path")
    lp.add_argument("--root", type=int, default=None, help="root node index for default path")

    w = sub.add_parser("watch_layer_prof", help="poll layer-prof and trigger set_plan on a simple condition")
    w.add_argument("--path", default=None, help="snapshot path (overrides env/default)")
    w.add_argument("--layer", type=int, default=0, help="watch this layer index")
    w.add_argument("--interval-ms", type=int, default=200)
    w.add_argument("--threshold-us", type=int, default=20000, help="trigger when (max attn+ffn) exceeds this (legacy absolute threshold)")
    w.add_argument("--delta-threshold-us", type=int, default=None, help="trigger when (slow-fast) score delta exceeds this (preferred)")
    w.add_argument("--stage", type=int, default=0)
    w.add_argument("--kind", type=int, default=1, help="1=headSplit 2=ffnSplit 3=both")
    w.add_argument("--heads", type=int, default=1)
    w.add_argument("--ffn", type=int, default=256)
    w.add_argument("--dry-run", action="store_true", help="print planned set_plan but do not send")

    sp = sub.add_parser("set_plan")
    sp.add_argument("--seq", type=int, default=1)
    sp.add_argument("--mode", choices=["exact", "next_barrier", "next"], default="next_barrier")
    sp.add_argument("--stage", type=int, default=0)
    sp.add_argument("--from", dest="from_node", type=int, default=0)
    sp.add_argument("--to", dest="to_node", type=int, default=1)
    sp.add_argument("--kind", type=int, default=3, help="1=headSplit 2=ffnSplit 3=both")
    sp.add_argument("--heads", type=int, default=1)
    sp.add_argument("--ffn", type=int, default=256)
    sp.add_argument(
        "--move",
        action="append",
        default=None,
        help="repeatable: FROM,TO[,KIND[,HEADS[,FFN]]] (uses v2 moves[]; overrides --from/--to)",
    )
    sp.add_argument("--trigger-pos", type=int, default=None)
    sp.add_argument("--trigger-layer", type=int, default=None)

    spp = sub.add_parser("set_pp_migration", help="set PP layer migration trigger/route (global root dispatch)")
    spp.add_argument("--seq", type=int, default=1)
    spp.add_argument("--mode", choices=["exact", "next_barrier", "next"], default="next_barrier")
    spp.add_argument("--stage", type=int, default=0xFFFFFFFF)
    spp.add_argument("--from", dest="from_node", type=int, default=0)
    spp.add_argument("--to", dest="to_node", type=int, default=1)
    spp.add_argument("--layer-count", type=int, default=1)
    spp.add_argument("--trigger-pos", type=int, default=None)
    spp.add_argument("--trigger-layer", type=int, default=None)

    rg = sub.add_parser("set_runtime_gate", help="toggle primary/redundant segment gate at runtime")
    rg.add_argument("--primary", type=int, choices=[0, 1], required=True, help="enable primary segments (0/1)")
    rg.add_argument("--redundant", type=int, choices=[0, 1], required=True, help="enable redundant segments (0/1)")

    pl = sub.add_parser("set_primary_layer", help="toggle a primary layer at runtime")
    pl.add_argument("--layer", type=int, required=True, help="layer index")
    pl.add_argument("--enabled", type=int, choices=[0, 1], required=True, help="enabled flag (0/1)")

    raw = sub.add_parser("raw")
    raw.add_argument("json", help='raw request JSON, e.g. {"op":"ping"}')

    args = p.parse_args()

    if args.op == "watch_layer_prof":
        import time

        last_epoch = None
        seq = 1
        try:
            while True:
                req = {
                    "op": "layer_prof",
                    "layerIndex": args.layer,
                    "stageIndex": args.stage,
                    "rootNodeIndex": 0,
                }
                if args.path:
                    req["path"] = args.path
                try:
                    resp = uds_request(args.socket, req)
                except Exception as e:
                    sys.stderr.write(f"[watch] uds_request failed: {e}\n")
                    return 2

                if not resp.get("ok"):
                    json.dump(resp, sys.stdout, indent=2, sort_keys=True)
                    sys.stdout.write("\n")
                    return 2

                lp = resp.get("layer_prof", {})
                epoch = lp.get("epoch")
                nodes = lp.get("nodes", [])

                if epoch is not None and epoch != last_epoch:
                    last_epoch = epoch
                    picked = pick_min_max_nodes(nodes)
                    if picked is not None:
                        fast, slow, fast_score, slow_score = picked
                        delta = int(slow_score) - int(fast_score)
                        print(f"[watch] epoch={epoch} layer={args.layer} slow=node{slow['nodeIndex']} score={slow_score} fast=node{fast['nodeIndex']} score={fast_score} delta={delta}")

                        if slow.get("nodeIndex") != fast.get("nodeIndex"):
                            if args.delta_threshold_us is not None:
                                should_trigger = delta >= int(args.delta_threshold_us)
                            else:
                                should_trigger = slow_score >= args.threshold_us

                        if slow.get("nodeIndex") != fast.get("nodeIndex") and should_trigger:
                            cmd = {
                                "seq": seq,
                                "mode": "next_barrier",
                                "stageIndex": args.stage,
                                "fromNodeIndex": int(slow["nodeIndex"]),
                                "toNodeIndex": int(fast["nodeIndex"]),
                                "cmdKind": int(args.kind),
                                "nHeadsToMove": int(args.heads),
                                "nFfnToMove": int(args.ffn),
                            }
                            req2 = {"op": "set_plan", "cmd": cmd}
                            if args.dry_run:
                                print("[watch] trigger set_plan (dry-run):", json.dumps(req2))
                            else:
                                resp2 = uds_request(args.socket, req2)
                                print("[watch] trigger set_plan resp:", json.dumps(resp2, sort_keys=True))
                                if not resp2.get("ok"):
                                    return 2
                            seq += 1

                time.sleep(max(1, args.interval_ms) / 1000.0)
        except KeyboardInterrupt:
            sys.stderr.write("[watch] stopped\n")
            return 0

    if args.op == "raw":
        req = json.loads(args.json)
    elif args.op == "set_plan":
        req = cmd_set_plan(args)
    elif args.op == "set_pp_migration":
        req = cmd_set_pp_migration(args)
    elif args.op == "set_runtime_gate":
        req = {
            "op": "set_runtime_gate",
            "primaryEnabled": int(args.primary),
            "redundantEnabled": int(args.redundant),
        }
    elif args.op == "set_primary_layer":
        req = {
            "op": "set_primary_layer",
            "layerIndex": int(args.layer),
            "enabled": int(args.enabled),
        }
    elif args.op == "layer_prof":
        req = cmd_layer_prof(args)
    else:
        req = {"op": args.op}

    resp = uds_request(args.socket, req)
    json.dump(resp, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if resp.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

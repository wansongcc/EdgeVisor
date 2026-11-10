#!/usr/bin/env python3
"""Drive repeated PP migrations into a running EdgeVisor root over its UDS.

Why a driver instead of a single env knob: `DLLAMA_PP_MIGRATION_POS` arms exactly
one trigger for the whole episode, so it cannot exercise
`resolvePpMigrationLayersByOwnership` -- the code path added in ed7e695 that is
supposed to walk the source stage's boundary inward when a second migration
names a layer the source already gave away. Every command here goes through the
same `set_pp_migration` admission gate the auto-runtime uses.

usage:
  uds_migrate.py --sock /tmp/dllama_plan.sock \
      --moves "0,1,7,2" "0,1,7,3" "0,1,7,4" [--wait 45] [--layer-count 1]
"""

import argparse
import json
import os
import socket
import sys
import time


def send(sock_path, payload, timeout=30.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock_path)
    try:
        s.sendall((json.dumps(payload) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    text = buf.decode(errors="replace").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default=os.environ.get("DLLAMA_PLAN_CTRL_SOCKET", "/tmp/dllama_plan.sock"))
    ap.add_argument("--moves", nargs="+", required=True,
                    help="one 'from,to,triggerLayer,triggerPos' per migration")
    ap.add_argument("--layer-count", type=int, default=1)
    ap.add_argument("--wait", type=float, default=45.0,
                    help="seconds to let the episode advance between commands")
    ap.add_argument("--settle", type=float, default=8.0,
                    help="seconds to wait for the socket to appear at startup")
    ap.add_argument("--jit-estimates", default="500,50,0.9",
                    help="predictedBenefitMs,predictedCostMs,survivalProbability; "
                         "empty string disables, to observe the gate rejecting")
    ap.add_argument("--mode", choices=["exact", "next_barrier"], default="next_barrier",
                    help="exact needs triggerPos to still be ahead of the pipeline; "
                         "next_barrier fires at the next barrier in the stage, which is "
                         "what the auto-runtime uses")
    ap.add_argument("--safety-margin", type=float, default=0.0)
    args = ap.parse_args()

    deadline = time.time() + args.settle
    while not os.path.exists(args.sock):
        if time.time() > deadline:
            print(f"FATAL: {args.sock} never appeared", file=sys.stderr)
            return 2
        time.sleep(0.5)
    print(f"[uds] talking to {args.sock}")

    # A migration is only admitted once the previous one has finished its KV
    # ack, so space the commands out instead of firing them in a burst.
    for idx, spec in enumerate(args.moves):
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 4:
            print(f"FATAL: bad move spec {spec!r}, want from,to,triggerLayer,triggerPos", file=sys.stderr)
            return 2
        frm, to, layer, pos = (int(p) for p in parts)
        cmd = {
            "mode": args.mode,
            "fromNodeIndex": frm,
            "toNodeIndex": to,
            "layerCount": args.layer_count,
            "seq": idx + 1,
        }
        if args.mode == "exact":
            cmd["triggerLayer"] = layer
            cmd["triggerPos"] = pos
        else:
            # next_barrier reuses triggerLayer as an exact firstLayer but leaves
            # triggerPos unset, so the command fires at whatever barrier comes
            # next instead of waiting for a position that may already be past.
            cmd["firstLayer"] = layer
        # The online planner gates every command on its own cost/benefit
        # estimates. A lab driver has no telemetry to derive them from, so
        # supply explicit, obviously-positive values rather than relying on a
        # different jitMode. Keep them in the request so the log records what
        # the gate actually saw.
        if args.jit_estimates:
            benefit, cost, survival = (float(x) for x in args.jit_estimates.split(","))
            cmd["predictedBenefitMs"] = benefit
            cmd["predictedCostMs"] = cost
            cmd["survivalProbability"] = survival
            cmd["safetyMarginMs"] = args.safety_margin
        if idx > 0:
            print(f"[uds] waiting {args.wait:.0f}s for the previous migration to settle")
            time.sleep(args.wait)
        print(f"[uds] move {idx + 1}/{len(args.moves)}: {frm}->{to} layer={layer} pos={pos}")
        resp = send(args.sock, {"op": "set_pp_migration", "cmd": cmd})
        verdict = "ACCEPTED" if resp.get("ok") else "REJECTED"
        print(f"[uds]   {verdict} {json.dumps(resp, sort_keys=True)}")
        if not resp.get("ok") and not resp.get("rejected"):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

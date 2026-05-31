#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class Request:
    layer: int
    pos: Optional[int] = None
    stage_nodes: Optional[List[int]] = None
    head_lens: Optional[List[int]] = None

    def to_file_line(self) -> str:
        parts = [f"layer={self.layer}"]
        if self.pos is not None:
            parts.append(f"pos={self.pos}")
        if self.stage_nodes is not None and self.head_lens is not None:
            parts.append("stage_nodes=" + ",".join(str(x) for x in self.stage_nodes))
            parts.append("head_lens=" + ",".join(str(x) for x in self.head_lens))
        return " ".join(parts) + "\n"


def _parse_int(s: str) -> int:
    s = s.strip()
    if s == "":
        raise ValueError("empty")
    v = int(s, 10)
    if v < 0:
        raise ValueError("negative")
    return v


def parse_plan(plan: str) -> List[Request]:
    """
    Supported formats:
      - "21"                 -> layer=21
      - "21 4"               -> layer=21 pos=4
      - "layer=21"           -> layer=21
      - "layer=21 pos=4"     -> layer=21 pos=4
      - Multiple steps separated by comma:
          "21, 22, 23 4, layer=24 pos=5"
    """
    steps: List[Request] = []
    for raw in plan.split(","):
        raw = raw.strip()
        if not raw:
            continue

        # Key=Value style
        if any(k in raw for k in ("layer=", "pos=", "stage_nodes=", "head_lens=")):
            layer_val: Optional[int] = None
            pos_val: Optional[int] = None
            stage_nodes: Optional[List[int]] = None
            head_lens: Optional[List[int]] = None
            for tok in raw.replace("\t", " ").split():
                tok = tok.strip()
                if tok.startswith("layer="):
                    layer_val = _parse_int(tok[len("layer="):])
                elif tok.startswith("pos="):
                    pos_val = _parse_int(tok[len("pos="):])
                elif tok.startswith("stage_nodes="):
                    stage_nodes = [_parse_int(x) for x in tok[len("stage_nodes="):].split(",") if x.strip()]
                elif tok.startswith("head_lens="):
                    head_lens = [_parse_int(x) for x in tok[len("head_lens="):].split(",") if x.strip()]
            if layer_val is None:
                raise ValueError(f"missing layer in step: {raw!r}")
            if (stage_nodes is None) ^ (head_lens is None):
                raise ValueError(f"stage_nodes/head_lens must be provided together: {raw!r}")
            if stage_nodes is not None and head_lens is not None and len(stage_nodes) != len(head_lens):
                raise ValueError(f"stage_nodes/head_lens length mismatch: {raw!r}")
            steps.append(Request(layer=layer_val, pos=pos_val, stage_nodes=stage_nodes, head_lens=head_lens))
            continue

        # Positional style: "<layer> [pos]"
        parts = raw.split()
        if len(parts) == 1:
            steps.append(Request(layer=_parse_int(parts[0]), pos=None))
        elif len(parts) == 2:
            steps.append(Request(layer=_parse_int(parts[0]), pos=_parse_int(parts[1])))
        else:
            raise ValueError(f"unrecognized step: {raw!r}")

    if not steps:
        raise ValueError("empty plan")
    return steps


def _tail_lines(path: str, sleep_s: float = 0.05) -> Iterator[str]:
    """Simple tail -f generator (yields lines as they appear)."""
    while True:
        try:
            with open(path, "r", encoding="utf-8") as f:
                while True:
                    line = f.readline()
                    if line:
                        yield line
                        continue
                    time.sleep(sleep_s)
        except FileNotFoundError:
            time.sleep(0.2)


def _alloc_ints_by_weights(total: int, weights: List[float]) -> List[int]:
    """Allocate 'total' into len(weights) non-negative ints proportional to weights."""
    if total <= 0:
        return [0 for _ in weights]
    if not weights or any(w < 0 for w in weights):
        raise ValueError("bad weights")
    s = sum(weights)
    if s <= 0:
        # fallback: uniform
        base = total // len(weights)
        rem = total - base * len(weights)
        out = [base] * len(weights)
        for i in range(rem):
            out[i] += 1
        return out

    raw = [w / s * total for w in weights]
    floors = [int(x) for x in raw]
    rem = total - sum(floors)
    frac = [raw[i] - floors[i] for i in range(len(weights))]
    # Largest remainder
    order = sorted(range(len(weights)), key=lambda i: frac[i], reverse=True)
    out = floors[:]
    for i in range(rem):
        out[order[i % len(order)]] += 1
    return out


def run_auto_perf(
    req_file: str,
    perf_file: str,
    layer: int,
    stage_index: int,
    stage_nodes: List[int],
    total_heads: int,
    alpha: float,
    imbalance_ratio: float,
    min_write_interval_s: float,
    pin_pos: bool,
) -> int:
    stage_nodes = list(stage_nodes)
    stage_nodes_set = set(stage_nodes)
    if len(stage_nodes) == 0:
        raise ValueError("empty stage_nodes")
    if total_heads <= 0:
        raise ValueError("total_heads must be > 0")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1")

    ewma_total_us: Dict[int, float] = {}
    last_write_t = 0.0
    last_written: Optional[Tuple[int, ...]] = None

    print(f"[auto] perf_file: {perf_file}")
    print(f"[auto] req_file:  {req_file}")
    print(f"[auto] stage={stage_index} nodes={stage_nodes} total_heads={total_heads} layer={layer}")
    print(f"[auto] alpha={alpha} imbalance_ratio={imbalance_ratio} min_write_interval_s={min_write_interval_s}")

    for line in _tail_lines(perf_file):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue

        pos = int(ev.get("pos", 0))
        nodes = ev.get("nodes", [])
        # Update EWMA for relevant nodes in this stage.
        seen_any = False
        for p in nodes:
            try:
                node = int(p.get("node"))
                stg = int(p.get("stage"))
                if stg != stage_index:
                    continue
                if node not in stage_nodes_set:
                    continue
                total_us = float(int(p.get("execUs", 0)) + int(p.get("syncUs", 0)))
            except Exception:
                continue
            prev = ewma_total_us.get(node)
            ewma_total_us[node] = total_us if prev is None else (alpha * total_us + (1.0 - alpha) * prev)
            seen_any = True

        if not seen_any:
            continue
        if any(n not in ewma_total_us for n in stage_nodes):
            continue

        vals = [ewma_total_us[n] for n in stage_nodes]
        vmin = min(vals)
        vmax = max(vals)
        if vmin <= 0:
            continue

        ratio = vmax / vmin
        if ratio < imbalance_ratio:
            continue

        now = time.time()
        if now - last_write_t < min_write_interval_s:
            continue

        # Faster nodes (smaller us) get more heads.
        weights = [1.0 / ewma_total_us[n] for n in stage_nodes]
        head_lens = _alloc_ints_by_weights(total_heads, weights)
        key = tuple(head_lens)
        if last_written == key:
            continue

        req = Request(
            layer=layer,
            pos=pos if pin_pos else None,
            stage_nodes=stage_nodes,
            head_lens=head_lens,
        )
        atomic_write(req_file, req.to_file_line())
        last_write_t = now
        last_written = key
        print(f"[auto] pos={pos} ewmaUs={{{', '.join(f'{n}:{ewma_total_us[n]:.0f}' for n in stage_nodes)}}} -> head_lens={head_lens}")

    return 0


def atomic_write(path: str, content: str) -> None:
    # Write via temp file then rename to ensure readers never see partial data.
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="sharding_scheduler",
        description=(
            "Root-side scheduler that updates DLLAMA_SHARDING_UPDATE_FILE to trigger runtime sharding updates. "
            "Run this on the root machine while inference is running."
        ),
    )
    ap.add_argument(
        "--file",
        default=os.environ.get("DLLAMA_SHARDING_UPDATE_FILE", "/tmp/dllama_sharding_req.txt"),
        help="Request file path (default: env DLLAMA_SHARDING_UPDATE_FILE or /tmp/dllama_sharding_req.txt)",
    )
    ap.add_argument(
        "--plan",
        default=None,
        help=(
            "Comma-separated plan. Examples: '21', '21 4', 'layer=21 pos=4', '21,22,23 4'. "
            "Each step overwrites the file once."
        ),
    )
    ap.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        help="Sleep between steps (default: 500ms)",
    )
    ap.add_argument(
        "--loop",
        action="store_true",
        help="Loop the plan forever",
    )
    ap.add_argument(
        "--stdin",
        action="store_true",
        help=(
            "Interactive mode: read lines from stdin and write them verbatim to the request file. "
            "Useful for manual scheduling: e.g. type 'layer=21 pos=4'."
        ),
    )

    ap.add_argument(
        "--auto-perf-file",
        default=None,
        help=(
            "Auto mode: read perf JSONL generated by DLLAMA_PERF_STREAM_FILE and write sharding requests. "
            "Example: /tmp/dllama_perf.jsonl"
        ),
    )
    ap.add_argument("--auto-layer", type=int, default=None, help="Auto mode: target layer to update")
    ap.add_argument("--auto-stage", type=int, default=None, help="Auto mode: stageIndex to consider")
    ap.add_argument(
        "--auto-stage-nodes",
        default=None,
        help="Auto mode: comma-separated node indices in this stage, e.g. '2,3'",
    )
    ap.add_argument("--auto-total-heads", type=int, default=None, help="Auto mode: total heads to distribute across stage nodes")
    ap.add_argument("--auto-alpha", type=float, default=0.3, help="Auto mode: EWMA alpha (default 0.3)")
    ap.add_argument("--auto-imbalance-ratio", type=float, default=1.25, help="Auto mode: trigger when max/min EWMA >= this (default 1.25)")
    ap.add_argument("--auto-min-write-interval-ms", type=int, default=500, help="Auto mode: minimum interval between writes (default 500ms)")
    ap.add_argument("--auto-pin-pos", action="store_true", help="Auto mode: include pos=<current> in request file")

    args = ap.parse_args(argv)

    if args.auto_perf_file is not None:
        if args.auto_layer is None or args.auto_stage is None or args.auto_stage_nodes is None or args.auto_total_heads is None:
            ap.error("auto mode requires: --auto-perf-file --auto-layer --auto-stage --auto-stage-nodes --auto-total-heads")
        try:
            stage_nodes = [_parse_int(x) for x in args.auto_stage_nodes.split(",") if x.strip()]
        except Exception as e:
            ap.error(f"invalid --auto-stage-nodes: {e}")
        return run_auto_perf(
            req_file=args.file,
            perf_file=args.auto_perf_file,
            layer=int(args.auto_layer),
            stage_index=int(args.auto_stage),
            stage_nodes=stage_nodes,
            total_heads=int(args.auto_total_heads),
            alpha=float(args.auto_alpha),
            imbalance_ratio=float(args.auto_imbalance_ratio),
            min_write_interval_s=max(0.0, float(args.auto_min_write_interval_ms) / 1000.0),
            pin_pos=bool(args.auto_pin_pos),
        )

    if args.stdin:
        path = args.file
        print(f"[sched] writing requests to: {path}")
        print("[sched] enter requests like: 'layer=21 pos=4' or '21 4' or '21'. Ctrl-D to exit.")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                # Normalize supported shorthand to canonical file line.
                reqs = parse_plan(line)
                # In stdin mode, only take first request from the line.
                req = reqs[0]
                content = req.to_file_line()
            except Exception:
                # If parse fails, just write the raw line (still may be accepted by runtime).
                content = line + "\n"
            atomic_write(path, content)
            print(f"[sched] wrote: {content.strip()}")
        return 0

    if args.plan is None:
        ap.error("need --plan or --stdin")

    try:
        steps = parse_plan(args.plan)
    except Exception as e:
        ap.error(f"invalid --plan: {e}")

    interval_s = max(0.0, args.interval_ms / 1000.0)
    path = args.file

    print(f"[sched] file: {path}")
    print(f"[sched] steps: {len(steps)} | interval: {args.interval_ms}ms | loop: {args.loop}")

    while True:
        for req in steps:
            content = req.to_file_line()
            atomic_write(path, content)
            print(f"[sched] wrote: {content.strip()}")
            if interval_s > 0:
                time.sleep(interval_s)
        if not args.loop:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

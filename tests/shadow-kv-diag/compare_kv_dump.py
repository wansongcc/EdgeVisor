#!/usr/bin/env python3
"""Compare dumped shadow (red_k/red_v) KV rows against main KV rows.

Usage: compare_kv_dump.py <dump_dir>

Pairs: for every 'red' dump of layer L on node N, find the 'main' dump of the
same layer L (same kind k/v, same batch row b, same position p) on a DIFFERENT
node (the stage that owns layer L actively). Reports max/mean abs diff, and
also the best-matching position on the reference node (staleness detection).
"""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

META_RE = re.compile(r"(\w+)=([^\s]+)")


def load_entries(dump_dir):
    entries = []
    for meta_path in glob.glob(os.path.join(dump_dir, "*.meta")):
        meta = {}
        with open(meta_path) as f:
            for k, v in META_RE.findall(f.read()):
                meta[k] = v
        f32_path = meta_path[:-5] + ".f32"
        if not os.path.exists(f32_path):
            continue
        meta["path"] = f32_path
        for key in ("node", "seg", "layer", "batch", "pos", "slot", "rowWidth",
                    "dstColStart", "dstRowStride", "kvSlotStride"):
            meta[key] = int(meta[key])
        entries.append(meta)
    return entries


def main():
    dump_dir = sys.argv[1]
    entries = load_entries(dump_dir)
    if not entries:
        print("no dump entries found")
        return 1

    main_idx = defaultdict(list)  # (layer, kind, batch, pos) -> [entry]
    red_entries = []
    for e in entries:
        if e["role"] == "red":
            red_entries.append(e)
        elif e["role"] == "main":
            main_idx[(e["layer"], e["kind"], e["batch"], e["pos"])].append(e)

    # reference positions per (layer, kind, batch) on each node for staleness check
    main_by_lkb = defaultdict(list)
    for (layer, kind, batch, pos), es in main_idx.items():
        for e in es:
            main_by_lkb[(layer, kind, batch)].append(e)

    print(f"total entries={len(entries)} red={len(red_entries)} main={sum(len(v) for v in main_idx.values())}")
    print()
    header = f"{'layer':>5} {'kind':>4} {'b':>2} {'pos':>4} {'redNode':>7} {'refNode':>7} {'maxAbsDiff':>12} {'meanAbsDiff':>12} {'refMaxAbs':>10} {'bestPos':>8} {'bestDiff':>12}"
    print(header)
    print("-" * len(header))
    n_ok = n_bad = 0
    rows = []
    for e in sorted(red_entries, key=lambda x: (x["layer"], x["kind"], x["batch"], x["pos"])):
        key = (e["layer"], e["kind"], e["batch"], e["pos"])
        refs = [r for r in main_idx.get(key, []) if r["node"] != e["node"] and r["rowWidth"] == e["rowWidth"]]
        if not refs:
            continue
        red = np.fromfile(e["path"], dtype=np.float32)
        for r in refs:
            ref = np.fromfile(r["path"], dtype=np.float32)
            if ref.shape != red.shape:
                continue
            diff = np.abs(red - ref)
            max_d = float(diff.max())
            mean_d = float(diff.mean())
            ref_max = float(np.abs(ref).max()) + 1e-12
            # staleness: best matching position among all positions of same layer/kind/batch on ref node
            best_pos, best_d = None, None
            for r2 in main_by_lkb.get((e["layer"], e["kind"], e["batch"]), []):
                if r2["node"] != r["node"] or r2["rowWidth"] != e["rowWidth"]:
                    continue
                ref2 = np.fromfile(r2["path"], dtype=np.float32)
                if ref2.shape != red.shape:
                    continue
                d2 = float(np.abs(red - ref2).max())
                if best_d is None or d2 < best_d:
                    best_d, best_pos = d2, r2["pos"]
            ok = max_d < 1e-3 * max(ref_max, 1.0)
            n_ok += ok
            n_bad += (not ok)
            rows.append((e["layer"], e["kind"], e["batch"], e["pos"], e["node"], r["node"],
                         max_d, mean_d, ref_max, best_pos, best_d, ok))
            print(f"{e['layer']:>5} {e['kind']:>4} {e['batch']:>2} {e['pos']:>4} {e['node']:>7} {r['node']:>7} "
                  f"{max_d:>12.6g} {mean_d:>12.6g} {ref_max:>10.4g} {str(best_pos):>8} {best_d:>12.6g}"
                  f"{'' if ok else '  <-- MISMATCH'}")
    print()
    print(f"pairs compared: {n_ok + n_bad}, ok={n_ok}, mismatch={n_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

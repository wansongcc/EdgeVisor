from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from agent_bench.backends import make_backend
from agent_bench.runtime import run_episode


def load_episode(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a LangGraph agent episode with a swappable LLM backend.")
    parser.add_argument("--backend", choices=["mock", "prima", "edgevisor"], required=True)
    parser.add_argument("--episode", type=Path, default=Path(__file__).resolve().parent / "episodes" / "calculator_episode.json")
    parser.add_argument("--out-root", type=Path, default=Path("/home/byh/B01/agent_bench_results"))
    parser.add_argument("--cuda-visible", default="0,1,2")
    parser.add_argument("--edge-worker-gpus", default="1", help="Comma-separated worker GPU indices for EdgeVisor.")
    parser.add_argument("--edge-ratios", default="1:1")
    parser.add_argument("--edge-steps", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=512)
    args = parser.parse_args()

    episode = load_episode(args.episode)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"{episode['id']}_{args.backend}_{stamp}"

    backend_kwargs: Dict[str, Any] = {}
    if args.backend == "prima":
        backend_kwargs = {"cuda_visible": args.cuda_visible, "ctx": args.ctx}
    elif args.backend == "edgevisor":
        worker_gpus = [int(x) for x in args.edge_worker_gpus.split(",") if x.strip()]
        backend_kwargs = {
            "cuda_visible": args.cuda_visible,
            "ctx": args.ctx,
            "steps": args.edge_steps,
            "ratios": args.edge_ratios,
            "worker_gpus": worker_gpus,
        }
    backend = make_backend(args.backend, **backend_kwargs)
    trace = run_episode(episode, backend, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "trace": trace}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


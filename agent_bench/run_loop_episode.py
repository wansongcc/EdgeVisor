from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from agent_bench.backends import make_backend
from agent_bench.runtime_loop import run_loop_episode


def load_episode(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_gpu_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a looping LangGraph agent with a swappable LLM backend.")
    parser.add_argument("--backend", choices=["mock", "prima", "edgevisor", "dllama", "edgevisor_exo"], required=True)
    parser.add_argument(
        "--episode",
        type=Path,
        default=Path(__file__).resolve().parent / "episodes" / "multi_tool_loop_episode.json",
    )
    parser.add_argument("--out-root", type=Path, default=Path("/home/byh/B01/agent_bench_results"))
    parser.add_argument("--cuda-visible", default="0,1,2")
    parser.add_argument("--edge-worker-gpus", default="1", help="Comma-separated worker GPU indices for EdgeVisor.")
    parser.add_argument("--edge-ratios", default="1:1")
    parser.add_argument("--edge-steps", type=int, default=256)
    parser.add_argument("--dllama-worker-gpus", default="1", help="Comma-separated worker GPU indices for Dllama.")
    parser.add_argument("--dllama-ratios", default="1:1")
    parser.add_argument("--edge-exo-gpus", default="0,1,2", help="Comma-separated GPU indices for EdgeVisor-EXO.")
    parser.add_argument("--edge-exo-total-layers", type=int, default=28)
    parser.add_argument("--edge-exo-memory-field", choices=["total", "free"], default="total")
    parser.add_argument("--ctx", type=int, default=2048)
    args = parser.parse_args()

    episode = load_episode(args.episode)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"{episode['id']}_{args.backend}_{stamp}"

    backend_kwargs: Dict[str, Any] = {}
    if args.backend == "prima":
        backend_kwargs = {"cuda_visible": args.cuda_visible, "ctx": args.ctx}
    elif args.backend == "edgevisor":
        backend_kwargs = {
            "cuda_visible": args.cuda_visible,
            "ctx": args.ctx,
            "steps": args.edge_steps,
            "ratios": args.edge_ratios,
            "worker_gpus": parse_gpu_list(args.edge_worker_gpus),
        }
    elif args.backend == "dllama":
        backend_kwargs = {
            "cuda_visible": args.cuda_visible,
            "ctx": args.ctx,
            "steps": args.edge_steps,
            "ratios": args.dllama_ratios,
            "worker_gpus": parse_gpu_list(args.dllama_worker_gpus),
        }
    elif args.backend == "edgevisor_exo":
        backend_kwargs = {
            "cuda_visible": args.cuda_visible,
            "ctx": args.ctx,
            "steps": args.edge_steps,
            "gpu_indices": parse_gpu_list(args.edge_exo_gpus),
            "total_layers": args.edge_exo_total_layers,
            "memory_field": args.edge_exo_memory_field,
        }
    backend = make_backend(args.backend, **backend_kwargs)
    trace = run_loop_episode(episode, backend, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "trace": trace}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    parser.add_argument(
        "--backend",
        choices=["mock", "prima", "edgevisor", "dllama", "edgevisor_exo", "edgevisor_lingualinked", "edgevisor_ablation"],
        required=True,
    )
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
    parser.add_argument("--lingualinked-gpus", default="0,1,2", help="Comma-separated GPU indices for EdgeVisor-LinguaLinked.")
    parser.add_argument("--lingualinked-total-layers", type=int, default=28)
    parser.add_argument("--lingualinked-overlap-layers", type=int, default=2)
    parser.add_argument("--lingualinked-memory-field", choices=["total", "free"], default="total")
    parser.add_argument("--shadow-kv-mode", choices=["enabled", "disabled_transfer", "disabled_recompute"], default="enabled")
    parser.add_argument(
        "--pointer-swizzling-mode",
        choices=["enabled", "operator_rebuild", "weight_rematerialize"],
        default="enabled",
    )
    parser.add_argument("--jit-mode", choices=["enabled", "static", "greedy", "oracle"], default="enabled")
    parser.add_argument("--vg-mode", choices=["enabled", "flat", "random", "pure_pp", "no_elastic_vg"], default="enabled")
    parser.add_argument("--fallback-policy", default="disabled_unless_necessary")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--edgevisor-ablation-config", type=Path, default=None)
    parser.add_argument("--edge-cold-start", action="store_true", help="Disable persistent EdgeVisor API session for cold-start comparison.")
    parser.add_argument("--edge-api-port", type=int, default=0, help="Optional fixed port for the persistent EdgeVisor API session.")
    parser.add_argument(
        "--allow-head-kv-migration",
        action="store_true",
        help="Opt in to experimental online KV/head migration; disabled by default for agentic success runs.",
    )
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
    elif args.backend == "edgevisor_lingualinked":
        backend_kwargs = {
            "cuda_visible": args.cuda_visible,
            "ctx": args.ctx,
            "steps": args.edge_steps,
            "gpu_indices": parse_gpu_list(args.lingualinked_gpus),
            "total_layers": args.lingualinked_total_layers,
            "overlap_layers": args.lingualinked_overlap_layers,
            "memory_field": args.lingualinked_memory_field,
        }
    elif args.backend == "edgevisor_ablation":
        backend_kwargs = {
            "cuda_visible": args.cuda_visible,
            "ctx": args.ctx,
            "steps": args.edge_steps,
            "ratios": args.edge_ratios,
            "worker_gpus": parse_gpu_list(args.edge_worker_gpus),
            "ablation_config": {
                "shadow_kv_mode": args.shadow_kv_mode,
                "pointer_swizzling_mode": args.pointer_swizzling_mode,
                "jit_mode": args.jit_mode,
                "vg_mode": args.vg_mode,
                "fallback_policy": args.fallback_policy,
                "allow_head_kv_migration": args.allow_head_kv_migration,
                "experiment_id": args.experiment_id or f"{episode['id']}_{args.backend}",
                "config_path": str(args.edgevisor_ablation_config) if args.edgevisor_ablation_config else "",
            },
            "persistent": not args.edge_cold_start,
            "api_port": args.edge_api_port,
        }
    backend = make_backend(args.backend, **backend_kwargs)
    trace = run_loop_episode(episode, backend, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "trace": trace}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

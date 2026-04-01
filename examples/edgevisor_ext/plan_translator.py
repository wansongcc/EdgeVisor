import math


def _alphas_to_int_ratios(alphas, scale=1000):
    if not alphas:
        return [1]
    vals = []
    for a in alphas:
        v = int(round(float(a) * scale))
        if v <= 0:
            v = 1
        vals.append(v)
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
    if g > 1:
        vals = [max(1, v // g) for v in vals]
    return vals


def translate_init_result_to_launch(
    init_result,
    n_nodes,
    default_shadow_heads=2,
    runtime_redundant_boundary_layers=1,
):
    rragc = init_result["rragc"]
    intravg = init_result["intravg"]
    inter_layers = init_result["intervg_layers"]
    pipeline_order = rragc["pipeline_order"]

    if len(inter_layers) != len(pipeline_order):
        raise ValueError("intervg layer allocation size mismatch")

    stages = []
    layer_cursor = 0
    ratios_parts = []
    for idx, vg_id in enumerate(pipeline_order):
        key = str(vg_id)
        alloc = intravg[key]
        node_ids = list(alloc["worker_ids"])
        alphas = list(alloc["alphas"])
        if len(node_ids) != len(alphas):
            raise ValueError(f"intravg size mismatch for vg={vg_id}")
        int_ratios = _alphas_to_int_ratios(alphas)
        n_layers = int(inter_layers[idx])
        start_layer = layer_cursor
        end_layer = layer_cursor + n_layers
        layer_cursor = end_layer

        stage = {
            "stage_index": idx,
            "vg_id": vg_id,
            "root_node_id": int(rragc["vg_roots"][vg_id]),
            "node_ids": node_ids,
            "tp_alphas": alphas,
            "tp_ratios": int_ratios,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "n_layers": n_layers,
            "shadow_layers": {
                "boundary_count": runtime_redundant_boundary_layers,
                "at_start": max(0, start_layer - runtime_redundant_boundary_layers),
                "at_end": end_layer + runtime_redundant_boundary_layers,
            },
        }
        stages.append(stage)
        tp = ":".join(str(x) for x in int_ratios)
        ratios_parts.append(f"{tp}@{n_layers}")

    all_nodes = set()
    for st in stages:
        for nid in st["node_ids"]:
            if nid in all_nodes:
                raise ValueError(f"duplicated node assignment: {nid}")
            all_nodes.add(nid)
    if len(all_nodes) != n_nodes:
        raise ValueError(f"node assignment mismatch: assigned={len(all_nodes)} expected={n_nodes}")

    kv_redundancy = [int(default_shadow_heads) for _ in range(n_nodes)]
    ratios = "*".join(ratios_parts)
    kv_str = ",".join(str(x) for x in kv_redundancy)

    required_flags = [
        "--enable-plan-barrier",
        "--enable-stage-full-weights",
        "--enable-kv-redundancy-during-migration 1",
    ]

    inference_args = [
        "--ratios",
        ratios,
        "--kv-redundancy",
        kv_str,
        "--runtime-redundant-boundary-layers",
        str(int(runtime_redundant_boundary_layers)),
        "--enable-plan-barrier",
        "--enable-stage-full-weights",
        "--enable-kv-redundancy-during-migration",
        "1",
    ]

    cmd = (
        "./dllama inference "
        "--model <MODEL_PATH> --tokenizer <TOKENIZER_PATH> "
        "--workers <WORKER0:PORT> <WORKER1:PORT> ... "
        + " ".join(inference_args)
    )

    init_plan = {
        "pipeline_order": pipeline_order,
        "stages": stages,
        "shadow_heads_per_device": kv_redundancy,
        "runtime_redundant_boundary_layers": int(runtime_redundant_boundary_layers),
    }

    launch_plan = {
        "ratios": ratios,
        "kv_redundancy": kv_redundancy,
        "runtime_redundant_boundary_layers": int(runtime_redundant_boundary_layers),
        "required_flags": required_flags,
        "inference_args": inference_args,
        "inference_command_template": cmd,
    }
    return init_plan, launch_plan


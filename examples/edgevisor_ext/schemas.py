import time


class SchemaError(ValueError):
    pass


def _require_keys(obj, keys, name):
    if not isinstance(obj, dict):
        raise SchemaError(f"{name} must be an object")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise SchemaError(f"{name} missing keys: {missing}")


def validate_agent_report(report):
    _require_keys(
        report,
        ["node_id", "host", "compute_flops", "memory_bytes", "links", "timestamp"],
        "agent_report",
    )
    if not isinstance(report["node_id"], int) or report["node_id"] < 0:
        raise SchemaError("node_id must be non-negative int")
    if not isinstance(report["host"], str) or not report["host"]:
        raise SchemaError("host must be non-empty string")
    if float(report["compute_flops"]) <= 0:
        raise SchemaError("compute_flops must be > 0")
    if int(report["memory_bytes"]) <= 0:
        raise SchemaError("memory_bytes must be > 0")
    if not isinstance(report["links"], list):
        raise SchemaError("links must be a list")
    for idx, link in enumerate(report["links"]):
        _require_keys(link, ["dst_id", "bandwidth_gbps"], f"links[{idx}]")
        if int(link["dst_id"]) < 0:
            raise SchemaError("dst_id must be non-negative")
        if float(link["bandwidth_gbps"]) < 0:
            raise SchemaError("bandwidth_gbps must be >= 0")
    if float(report["timestamp"]) <= 0:
        raise SchemaError("timestamp must be positive epoch seconds")
    return report


def validate_model_meta(meta):
    _require_keys(
        meta,
        [
            "total_layers",
            "activation_size_gb",
            "layer_total_flops",
            "layer_input_bytes",
            "layer_output_bytes",
        ],
        "model_meta",
    )
    if int(meta["total_layers"]) <= 0:
        raise SchemaError("total_layers must be > 0")
    if float(meta["activation_size_gb"]) <= 0:
        raise SchemaError("activation_size_gb must be > 0")
    if float(meta["layer_total_flops"]) <= 0:
        raise SchemaError("layer_total_flops must be > 0")
    if float(meta["layer_input_bytes"]) <= 0:
        raise SchemaError("layer_input_bytes must be > 0")
    if float(meta["layer_output_bytes"]) <= 0:
        raise SchemaError("layer_output_bytes must be > 0")
    return meta


def validate_launch_plan(plan):
    _require_keys(
        plan,
        [
            "ratios",
            "kv_redundancy",
            "runtime_redundant_boundary_layers",
            "required_flags",
            "inference_args",
        ],
        "launch_plan",
    )
    if not isinstance(plan["ratios"], str) or not plan["ratios"]:
        raise SchemaError("ratios must be non-empty string")
    if not isinstance(plan["kv_redundancy"], list) or not plan["kv_redundancy"]:
        raise SchemaError("kv_redundancy must be non-empty list")
    if int(plan["runtime_redundant_boundary_layers"]) < 0:
        raise SchemaError("runtime_redundant_boundary_layers must be >= 0")
    if not isinstance(plan["required_flags"], list):
        raise SchemaError("required_flags must be list")
    if not isinstance(plan["inference_args"], list):
        raise SchemaError("inference_args must be list")
    return plan


def validate_perf_response(resp):
    _require_keys(resp, ["ok", "perf"], "perf_response")
    if not resp["ok"]:
        raise SchemaError(f"perf response not ok: {resp}")
    if not isinstance(resp["perf"], list):
        raise SchemaError("perf must be list")
    return resp


def validate_status_response(resp):
    _require_keys(resp, ["ok", "enablePlanBarrier"], "status_response")
    if not resp["ok"]:
        raise SchemaError(f"status response not ok: {resp}")
    return resp


def validate_plan_snapshot_response(resp):
    _require_keys(resp, ["ok", "plan_snapshot"], "plan_snapshot_response")
    if not resp["ok"]:
        raise SchemaError(f"plan_snapshot response not ok: {resp}")
    snap = resp["plan_snapshot"]
    _require_keys(snap, ["nNodes", "nStages", "stages", "splits"], "plan_snapshot")
    return resp


def validate_layer_prof_response(resp):
    _require_keys(resp, ["ok", "layer_prof"], "layer_prof_response")
    if not resp["ok"]:
        raise SchemaError(f"layer_prof response not ok: {resp}")
    lp = resp["layer_prof"]
    _require_keys(lp, ["header", "layers", "epoch"], "layer_prof")
    return resp


def now_s():
    return time.time()


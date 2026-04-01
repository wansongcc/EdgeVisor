#!/usr/bin/env python3
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from init_algorithms import run_initialization
from plan_translator import translate_init_result_to_launch
from schemas import (
    SchemaError,
    validate_agent_report,
    validate_launch_plan,
    validate_model_meta,
)


class ReportCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._reports = {}

    def upsert(self, report):
        with self._lock:
            self._reports[int(report["node_id"])] = report
            return len(self._reports)

    def snapshot(self):
        with self._lock:
            return dict(self._reports)


def make_handler(collector):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            if self.path != "/report":
                self._json(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                report = json.loads(raw)
                validate_agent_report(report)
                count = collector.upsert(report)
                self._json(200, {"ok": True, "received_nodes": count})
            except (ValueError, SchemaError) as e:
                self._json(400, {"ok": False, "error": str(e)})
            except Exception as e:
                self._json(500, {"ok": False, "error": str(e)})

        def do_GET(self):  # noqa: N802
            if self.path != "/status":
                self._json(404, {"ok": False, "error": "not found"})
                return
            reports = collector.snapshot()
            self._json(200, {"ok": True, "received_nodes": len(reports), "node_ids": sorted(reports.keys())})

        def log_message(self, fmt, *args):  # noqa: A003
            return

    return Handler


def _build_inputs_from_reports(reports):
    devices = []
    links = []
    for node_id, report in sorted(reports.items(), key=lambda x: x[0]):
        devices.append(
            {
                "id": int(node_id),
                "compute": float(report["compute_flops"]),
                "memory": float(report["memory_bytes"]),
            }
        )
        for link in report["links"]:
            links.append(
                {
                    "src_id": int(node_id),
                    "dst_id": int(link["dst_id"]),
                    "bandwidth": float(link["bandwidth_gbps"]),
                }
            )
    return devices, links


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description="EdgeVisor init-root: gather agent reports and output init/launch plans.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--expected-nodes", type=int, required=True)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--output-dir", default="examples/edgevisor_ext/out")

    ap.add_argument("--rragc-k", type=int, default=2)
    ap.add_argument("--rragc-p-min", type=float, default=0.0, help="0 means auto from reports")
    ap.add_argument("--rragc-m-min", type=float, default=0.0, help="0 means auto from reports")
    ap.add_argument("--rragc-alpha", type=float, default=0.7)
    ap.add_argument("--rragc-beta", type=float, default=0.3)

    ap.add_argument("--total-layers", type=int, required=True)
    ap.add_argument("--activation-size-gb", type=float, required=True)
    ap.add_argument("--layer-total-flops", type=float, required=True)
    ap.add_argument("--layer-input-bytes", type=float, required=True)
    ap.add_argument("--layer-output-bytes", type=float, required=True)

    ap.add_argument("--default-shadow-heads", type=int, default=2)
    ap.add_argument("--runtime-redundant-boundary-layers", type=int, default=1)
    args = ap.parse_args()

    model_meta = {
        "total_layers": args.total_layers,
        "activation_size_gb": args.activation_size_gb,
        "layer_total_flops": args.layer_total_flops,
        "layer_input_bytes": args.layer_input_bytes,
        "layer_output_bytes": args.layer_output_bytes,
    }
    validate_model_meta(model_meta)

    collector = ReportCollector()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(collector))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(json.dumps({"ok": True, "listening": f"http://{args.host}:{args.port}/report"}, ensure_ascii=False))

    deadline = time.time() + max(1.0, args.timeout_s)
    while time.time() < deadline:
        reports = collector.snapshot()
        if len(reports) >= args.expected_nodes:
            break
        time.sleep(0.2)

    reports = collector.snapshot()
    server.shutdown()
    server.server_close()

    if len(reports) < args.expected_nodes:
        print(json.dumps({"ok": False, "error": "timeout waiting reports", "received": len(reports)}, ensure_ascii=False))
        return 2

    devices, links = _build_inputs_from_reports(reports)
    total_compute = sum(float(d["compute"]) for d in devices)
    total_memory = sum(float(d["memory"]) for d in devices)
    k = max(1, min(args.rragc_k, len(devices)))
    p_min = args.rragc_p_min if args.rragc_p_min > 0 else (total_compute / k) * 0.4
    m_min = args.rragc_m_min if args.rragc_m_min > 0 else (total_memory / k) * 0.4

    rragc_config = {
        "K": k,
        "P_min": p_min,
        "M_min": m_min,
        "alpha": args.rragc_alpha,
        "beta": args.rragc_beta,
    }
    layer_task = {
        "total_flops": model_meta["layer_total_flops"],
        "input_bytes": model_meta["layer_input_bytes"],
        "output_bytes": model_meta["layer_output_bytes"],
    }
    model_config = {
        "total_layers": model_meta["total_layers"],
        "activation_size_gb": model_meta["activation_size_gb"],
    }

    init_result = run_initialization(devices, links, rragc_config, layer_task, model_config)
    init_plan, launch_plan = translate_init_result_to_launch(
        init_result=init_result,
        n_nodes=len(devices),
        default_shadow_heads=args.default_shadow_heads,
        runtime_redundant_boundary_layers=args.runtime_redundant_boundary_layers,
    )
    validate_launch_plan(launch_plan)

    os.makedirs(args.output_dir, exist_ok=True)
    init_plan_path = os.path.join(args.output_dir, "init_plan.json")
    launch_plan_path = os.path.join(args.output_dir, "launch_plan.json")
    _dump_json(init_plan_path, init_plan)
    _dump_json(launch_plan_path, launch_plan)

    print(json.dumps({"ok": True, "init_plan": init_plan_path, "launch_plan": launch_plan_path}, ensure_ascii=False))
    print("Suggested inference command:")
    print(launch_plan["inference_command_template"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


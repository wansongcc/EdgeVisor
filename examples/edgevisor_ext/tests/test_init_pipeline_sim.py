import os
import sys
import unittest


THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from init_algorithms import run_initialization  # noqa: E402
from plan_translator import translate_init_result_to_launch  # noqa: E402


def _build_synthetic_topology(n):
    devices = []
    for i in range(n):
        devices.append(
            {
                "id": i,
                "compute": 2e12 + (n - i) * 3e11,
                "memory": 8e9 + ((i % 3) * 2e9),
            }
        )

    links = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = abs(i - j)
            bw = 12.0 / (1.0 + dist)
            links.append({"src_id": i, "dst_id": j, "bandwidth": bw})
    return devices, links


class TestInitPipelineSim(unittest.TestCase):
    def _run_case(self, n_nodes):
        devices, links = _build_synthetic_topology(n_nodes)
        total_layers = 24 if n_nodes <= 4 else 32

        init_result = run_initialization(
            devices=devices,
            links=links,
            rragc_config={
                "K": 2 if n_nodes < 8 else 3,
                "P_min": 2e12,
                "M_min": 8e9,
                "alpha": 0.7,
                "beta": 0.3,
            },
            layer_task={
                "total_flops": 1e12,
                "input_bytes": 5e7,
                "output_bytes": 5e7,
            },
            model_config={
                "total_layers": total_layers,
                "activation_size_gb": 0.1,
            },
        )

        layers = init_result["intervg_layers"]
        order = init_result["rragc"]["pipeline_order"]
        self.assertEqual(len(layers), len(order))
        self.assertEqual(sum(layers), total_layers)
        self.assertTrue(all(x >= 0 for x in layers))

        init_plan, launch_plan = translate_init_result_to_launch(
            init_result=init_result,
            n_nodes=n_nodes,
            default_shadow_heads=2,
            runtime_redundant_boundary_layers=1,
        )

        self.assertEqual(len(launch_plan["kv_redundancy"]), n_nodes)
        self.assertEqual(sum(x["n_layers"] for x in init_plan["stages"]), total_layers)

        ratios = launch_plan["ratios"].split("*")
        self.assertEqual(len(ratios), len(order))
        parsed_layers = []
        for seg in ratios:
            self.assertIn("@", seg)
            _, layer_str = seg.split("@", 1)
            parsed_layers.append(int(layer_str))
        self.assertEqual(sum(parsed_layers), total_layers)

        assigned = set()
        for st in init_plan["stages"]:
            for nid in st["node_ids"]:
                self.assertNotIn(nid, assigned)
                assigned.add(nid)
        self.assertEqual(len(assigned), n_nodes)

    def test_4_nodes(self):
        self._run_case(4)

    def test_6_nodes(self):
        self._run_case(6)

    def test_8_nodes(self):
        self._run_case(8)


if __name__ == "__main__":
    unittest.main()


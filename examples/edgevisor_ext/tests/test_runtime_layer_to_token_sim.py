import os
import sys
import unittest


THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runtime_optimizer import OptimizerConfig, RuntimeOptimizer  # noqa: E402


class FakeUdsClient:
    def __init__(self):
        self.step = 0
        self.pp_commands = []
        self.token_commands = []
        self.clear_calls = 0

    def request(self, payload):
        op = payload["op"]
        if op == "status":
            self.step += 1
            return {"ok": True, "enablePlanBarrier": True, "cacheSeq": self.step, "cmd": {"mode": "none"}}
        if op == "last_apply":
            return {"ok": True, "last_apply": {"valid": False, "tsMs": 0}}
        if op == "perf":
            return {
                "ok": True,
                "perf": [
                    {"stageIndex": 0, "nodeIndex": 0, "execUs": 90000, "syncUs": 10000},
                    {"stageIndex": 0, "nodeIndex": 1, "execUs": 85000, "syncUs": 8000},
                    {"stageIndex": 1, "nodeIndex": 2, "execUs": 60000, "syncUs": 5000},
                    {"stageIndex": 1, "nodeIndex": 3, "execUs": 55000, "syncUs": 6000},
                ],
            }
        if op == "plan_snapshot":
            return {
                "ok": True,
                "plan_snapshot": {
                    "nNodes": 4,
                    "nStages": 2,
                    "stages": [
                        {"stageIndex": 0, "rootNodeIndex": 0, "startLayer": 0, "endLayer": 16, "nodes": [0, 1]},
                        {"stageIndex": 1, "rootNodeIndex": 2, "startLayer": 16, "endLayer": 32, "nodes": [2, 3]},
                    ],
                    "splits": {
                        "kvHeadSplit": {"starts": [0, 4, 0, 0], "lengths": [4, 4, 1, 1]},
                        "kvHeadComputeSplit": {"starts": [0, 2, 0, 0], "lengths": [6, 6, 1, 1]},
                        "headSplit": {"starts": [0, 8, 0, 0], "lengths": [8, 8, 2, 2]},
                        "ffnSplit": {"starts": [0, 2048, 0, 0], "lengths": [2048, 2048, 512, 512]},
                    },
                },
            }
        if op == "layer_prof":
            return {
                "ok": True,
                "layer_prof": {
                    "epoch": self.step,
                    "header": {"nLayers": 2, "nStageNodes": 2},
                    "layers": [
                        [
                            {"ok": True, "nodeIndex": 0, "attnUs": 30000, "ffnUs": 10000},
                            {"ok": True, "nodeIndex": 1, "attnUs": 7000, "ffnUs": 3000},
                        ],
                        [
                            {"ok": True, "nodeIndex": 0, "attnUs": 12000, "ffnUs": 8000},
                            {"ok": True, "nodeIndex": 1, "attnUs": 9000, "ffnUs": 5000},
                        ],
                    ],
                },
            }
        if op == "set_pp_migration":
            self.pp_commands.append(payload)
            return {"ok": True, "cacheSeq": 100 + len(self.pp_commands)}
        if op == "set_plan":
            self.token_commands.append(payload)
            return {"ok": True, "cacheSeq": 200 + len(self.token_commands)}
        if op == "clear":
            self.clear_calls += 1
            return {"ok": True}
        raise AssertionError(f"unexpected op: {op}")


class TestRuntimeLayerToTokenSim(unittest.TestCase):
    def test_layer_fallback_to_token(self):
        fake = FakeUdsClient()
        cfg = OptimizerConfig(
            min_improve_ratio=0.05,
            layer_no_improve_epochs=3,
            cooldown_steps=1,
            timeout_steps=5,
            backoff_base_steps=1,
            max_backoff_steps=4,
            seq_start=1,
        )
        opt = RuntimeOptimizer(fake, cfg)

        for _ in range(8):
            opt.step_once()

        self.assertGreaterEqual(len(fake.pp_commands), 3)
        self.assertGreaterEqual(len(fake.token_commands), 1)
        self.assertEqual(opt.mode, RuntimeOptimizer.MODE_TOKEN)
        self.assertTrue(fake.token_commands[0]["cmd"]["moves"])


if __name__ == "__main__":
    unittest.main()

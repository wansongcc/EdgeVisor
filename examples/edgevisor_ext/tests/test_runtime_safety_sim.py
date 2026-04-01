import os
import sys
import unittest


THIS_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runtime_optimizer import OptimizerConfig, RuntimeOptimizer  # noqa: E402


class TimeoutFakeUdsClient:
    def __init__(self):
        self.step = 0
        self.sent = 0
        self.clear_calls = 0

    def request(self, payload):
        op = payload["op"]
        if op == "status":
            self.step += 1
            return {"ok": True, "enablePlanBarrier": True, "cacheSeq": self.step, "cmd": {"mode": "next_barrier"}}
        if op == "last_apply":
            return {"ok": True, "last_apply": {"valid": False, "tsMs": 0}}
        if op == "perf":
            return {
                "ok": True,
                "perf": [
                    {"stageIndex": 0, "nodeIndex": 0, "execUs": 100000, "syncUs": 10000},
                    {"stageIndex": 1, "nodeIndex": 1, "execUs": 50000, "syncUs": 4000},
                ],
            }
        if op == "plan_snapshot":
            return {
                "ok": True,
                "plan_snapshot": {
                    "nNodes": 2,
                    "nStages": 2,
                    "stages": [
                        {"stageIndex": 0, "rootNodeIndex": 0, "startLayer": 0, "endLayer": 16, "nodes": [0]},
                        {"stageIndex": 1, "rootNodeIndex": 1, "startLayer": 16, "endLayer": 32, "nodes": [1]},
                    ],
                    "splits": {
                        "kvHeadSplit": {"starts": [0, 0], "lengths": [8, 8]},
                        "kvHeadComputeSplit": {"starts": [0, 0], "lengths": [8, 8]},
                        "headSplit": {"starts": [0, 0], "lengths": [8, 8]},
                        "ffnSplit": {"starts": [0, 0], "lengths": [2048, 2048]},
                    },
                },
            }
        if op == "set_pp_migration":
            self.sent += 1
            return {"ok": True, "cacheSeq": 123}
        if op == "clear":
            self.clear_calls += 1
            return {"ok": True}
        raise AssertionError(f"unexpected op: {op}")


class TestRuntimeSafetySim(unittest.TestCase):
    def test_inflight_timeout_clear_and_backoff(self):
        fake = TimeoutFakeUdsClient()
        cfg = OptimizerConfig(
            min_improve_ratio=0.05,
            layer_no_improve_epochs=3,
            cooldown_steps=10,  # force timeout before evaluation
            timeout_steps=2,
            backoff_base_steps=2,
            max_backoff_steps=8,
            seq_start=10,
        )
        opt = RuntimeOptimizer(fake, cfg)

        for _ in range(4):
            opt.step_once()

        self.assertEqual(fake.sent, 1)
        self.assertGreaterEqual(fake.clear_calls, 1)
        self.assertGreater(opt.backoff_steps, 0)
        self.assertIsNone(opt.in_flight)


if __name__ == "__main__":
    unittest.main()

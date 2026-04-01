# Phase-1 Local Simulation Test Report

Date: 2026-04-01  
Workspace: `EdgeVisor`

## 1. Command

```bash
python3 -m unittest discover -s examples/edgevisor_ext/tests -p "test_*.py" -v
```

## 2. Result Summary

- Total: 5
- Passed: 5
- Failed: 0

## 3. Case Details

1. `test_init_pipeline_sim.py::test_4_nodes`  
   Result: PASS  
   Checkpoint: RRAGC/CCWF/OLP 可生成有效层划分；`ratios` 与 `kv_redundancy` 输出合法。

2. `test_init_pipeline_sim.py::test_6_nodes`  
   Result: PASS  
   Checkpoint: 中规模拓扑下层数守恒、节点唯一归属保持成立。

3. `test_init_pipeline_sim.py::test_8_nodes`  
   Result: PASS  
   Checkpoint: 更大拓扑下初始化结果稳定，`launch_plan` 可落地。

4. `test_runtime_layer_to_token_sim.py::test_layer_fallback_to_token`  
   Result: PASS  
   Checkpoint: Layer 优化连续无效 3 次后切换到 Token；`set_pp_migration` 后可发 `set_plan(moves[])`。

5. `test_runtime_safety_sim.py::test_inflight_timeout_clear_and_backoff`  
   Result: PASS  
   Checkpoint: in-flight 超时触发 `clear`，并进入退避，避免重复下发风暴。

## 4. Conclusion

本地模拟数据测试已通过，可进入真实设备联调阶段。


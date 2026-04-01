# EdgeVisor Phase-1 External Controllers

本目录提供第一阶段外挂实现：

- `init_agent.py`: 设备侧上报 profile 到 `init_root`.
- `init_root.py`: 汇总 profile，执行 `RRAGC -> CCWF -> OLP`，产出 `init_plan.json` 和 `launch_plan.json`.
- `runtime_optimizer.py`: 运行期 Layer 优先优化，连续无效后切换到 Token 优化。
- `schemas.py`: 输入/输出的轻量 schema 校验。
- `init_algorithms.py`: 初始化算法实现。
- `plan_translator.py`: 论文语义计划 -> 当前工程启动参数。

## 1. 初始化阶段（HTTP JSON）

先启动 root：

```bash
python3 examples/edgevisor_ext/init_root.py \
  --expected-nodes 4 \
  --total-layers 32 \
  --activation-size-gb 0.1 \
  --layer-total-flops 1e12 \
  --layer-input-bytes 5e7 \
  --layer-output-bytes 5e7
```

然后每个节点运行 agent（示例）：

```bash
python3 examples/edgevisor_ext/init_agent.py \
  --node-id 0 \
  --root-url http://127.0.0.1:18080/report \
  --compute-flops 8e12 \
  --links "1:10,2:5"
```

输出目录默认 `examples/edgevisor_ext/out`：

- `init_plan.json`
- `launch_plan.json`

## 2. 运行时动态优化（UDS）

确保推理端已开启：

- `DLLAMA_PLAN_CTRL_SOCKET=/tmp/dllama_plan.sock`
- `--enable-plan-barrier`
- 并且不要启用 `DLLAMA_DYNAMIC_LAYER_ENABLE`（避免双控制器）

运行：

```bash
python3 examples/edgevisor_ext/runtime_optimizer.py /tmp/dllama_plan.sock
```

## 3. 本地模拟测试

```bash
python3 -m unittest discover -s examples/edgevisor_ext/tests -p "test_*.py" -v
```


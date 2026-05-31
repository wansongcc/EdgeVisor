# 02 GPU 单机测试

- 测试日志目录：`/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/02_gpu_single`
- 项目根目录：`/home/cc/yhbian/B01_Copy_API_Refactor`
- 本次测试参数：已加入 `--benchmark`

## 1. 运行的代码输入

```bash
cd /home/cc/yhbian/B01_Copy_API_Refactor/EdgeVisor
./dllama inference \
  --prompt "What is 2+2? Answer with only the number." \
  --steps 32 \
  --model "/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m" \
  --tokenizer "/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --temperature 0 \
  --seed 1 \
  --gpu-index 0 \
  --benchmark >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/02_gpu_single/root.log" 2>&1
```

## 2. 程序输出

原始输出已过滤 executor 初始化与 logits 诊断，仅保留停止符、chat template、Eval/Pred token 行和 benchmark 汇总。

```text
RC=0
🛑 Stop: <|end_of_text|>
🛑 Stop: <|eot_id|>
⭐ Chat template: llama3
🔷️ Eval  666 ms Sync    0 ms | Sent     0 kB Recv     0 kB | (21 tokens)
🔶 Pred   30 ms Sync    0 ms | pos=21 | Sent     0 kB Recv     0 kB | 4
🔶 Pred   26 ms Sync    0 ms | pos=22 | Sent     0 kB Recv     0 kB | [EOS]
Evaluation
   nBatches: 32
    nTokens: 21
   tokens/s: 31.48 (31.76 ms/tok)
Prediction
    nTokens: 2
   tokens/s: 35.27 (28.35 ms/tok)
⏱️  [Stage/Node Profile Summary]
  • Stage 0 Node 0: per-fwd total=241.23 ms (exec=241.23 sync=  0.00) | per-tok total= 31.46 ms (exec= 31.46 sync=  0.00) | fwd=3 tok=23
Hint: prompt eval uses batchSize>1, so per-token is usually the meaningful metric for rebalancing.
```

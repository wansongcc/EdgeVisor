# 04 GPU 非均匀静态测试

- 测试日志目录：`/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/04_gpu_uneven_static`
- 项目根目录：`/home/cc/yhbian/B01_Copy_API_Refactor`
- 本次测试参数：已加入 `--benchmark`

## 1. 运行的代码输入

```bash
cd /home/cc/yhbian/B01_Copy_API_Refactor/EdgeVisor
PORT1=19501
PORT2=19502
./dllama worker --port "${PORT1}" --nthreads 1 --gpu-index 1 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/04_gpu_uneven_static/worker1.log" 2>&1 &
./dllama worker --port "${PORT2}" --nthreads 1 --gpu-index 2 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/04_gpu_uneven_static/worker2.log" 2>&1 &
sleep 4
./dllama inference \
  --prompt "What is 2+2? Answer with only the number." \
  --steps 64 \
  --model "/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m" \
  --tokenizer "/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --temperature 0 \
  --seed 1 \
  --gpu-index 0 \
  --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
  --benchmark \
  --ratios "2:3:3" >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/04_gpu_uneven_static/root.log" 2>&1
```

## 2. 程序输出

原始输出已过滤 executor 初始化与 logits 诊断，仅保留停止符、chat template、Eval/Pred token 行和 benchmark 汇总。

```text
RC=0
🛑 Stop: <|end_of_text|>
🛑 Stop: <|eot_id|>
⭐ Chat template: llama3
🔷️ Eval  443 ms Sync  623 ms | Sent  8001 kB Recv 15389 kB | (21 tokens)
🔶 Pred   23 ms Sync   16 ms | pos=21 | Sent   381 kB Recv   734 kB | 4
🔶 Pred   22 ms Sync   10 ms | pos=22 | Sent   381 kB Recv   734 kB | [EOS]
Evaluation
   nBatches: 32
    nTokens: 21
   tokens/s: 19.67 (50.83 ms/tok)
Prediction
    nTokens: 2
   tokens/s: 27.44 (36.45 ms/tok)
⏱️  [Stage/Node Profile Summary]
  • Stage 0 Node 0: per-fwd total=380.09 ms (exec=163.33 sync=216.75) | per-tok total= 49.58 ms (exec= 21.30 sync= 28.27) | fwd=3 tok=23
  • Stage 0 Node 1: per-fwd total=276.92 ms (exec=214.05 sync= 62.87) | per-tok total= 36.12 ms (exec= 27.92 sync=  8.20) | fwd=3 tok=23
  • Stage 0 Node 2: per-fwd total=307.43 ms (exec=207.39 sync=100.04) | per-tok total= 40.10 ms (exec= 27.05 sync= 13.05) | fwd=3 tok=23
Hint: prompt eval uses batchSize>1, so per-token is usually the meaningful metric for rebalancing.
```

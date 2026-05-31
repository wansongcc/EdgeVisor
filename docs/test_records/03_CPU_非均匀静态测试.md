# 03 CPU 非均匀静态测试

- 测试日志目录：`/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/03_cpu_uneven_static`
- 项目根目录：`/home/cc/yhbian/B01_Copy_API_Refactor`
- 本次测试参数：已加入 `--benchmark`

## 1. 运行的代码输入

```bash
cd /home/cc/yhbian/B01_Copy_API_Refactor/EdgeVisor
PORT1=19601
PORT2=19602
./dllama worker --port "${PORT1}" --nthreads 2 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/03_cpu_uneven_static/worker1.log" 2>&1 &
./dllama worker --port "${PORT2}" --nthreads 2 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/03_cpu_uneven_static/worker2.log" 2>&1 &
sleep 4
./dllama inference \
  --prompt "What is 2+2? Answer with only the number." \
  --steps 32 \
  --model "/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m" \
  --tokenizer "/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t" \
  --buffer-float-type q80 \
  --nthreads 2 \
  --max-seq-len 512 \
  --temperature 0 \
  --seed 1 \
  --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
  --benchmark \
  --ratios "2:3:3" >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/03_cpu_uneven_static/root.log" 2>&1
```

## 2. 程序输出

原始输出已过滤 executor 初始化与 logits 诊断，仅保留停止符、chat template、Eval/Pred token 行和 benchmark 汇总。

```text
RC=0
🛑 Stop: <|end_of_text|>
🛑 Stop: <|eot_id|>
⭐ Chat template: llama3
🔷️ Eval 1315 ms Sync  709 ms | Sent  8001 kB Recv 15389 kB | (21 tokens)
🔶 Pred  304 ms Sync   75 ms | pos=21 | Sent   381 kB Recv   734 kB | 4
🔶 Pred  108 ms Sync   47 ms | pos=22 | Sent   381 kB Recv   734 kB | [EOS]
Evaluation
   nBatches: 32
    nTokens: 21
   tokens/s: 10.37 (96.40 ms/tok)
Prediction
    nTokens: 2
   tokens/s: 3.74 (267.72 ms/tok)
⏱️  [Stage/Node Profile Summary]
  • Stage 0 Node 0: per-fwd total=853.31 ms (exec=575.91 sync=277.40) | per-tok total=111.30 ms (exec= 75.12 sync= 36.18) | fwd=3 tok=23
  • Stage 0 Node 1: per-fwd total=852.29 ms (exec=746.60 sync=105.69) | per-tok total=111.17 ms (exec= 97.38 sync= 13.79) | fwd=3 tok=23
  • Stage 0 Node 2: per-fwd total=853.25 ms (exec=808.53 sync= 44.72) | per-tok total=111.29 ms (exec=105.46 sync=  5.83) | fwd=3 tok=23
Hint: prompt eval uses batchSize>1, so per-token is usually the meaningful metric for rebalancing.
```

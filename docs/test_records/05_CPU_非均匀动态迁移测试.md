# 05 CPU 非均匀动态迁移测试

- 测试日志目录：`/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/05_cpu_uneven_dynamic`
- 项目根目录：`/home/cc/yhbian/B01_Copy_API_Refactor`
- 本次测试参数：已加入 `--benchmark`

## 1. 运行的代码输入

### 1.1 init 输入

```bash
cd /home/cc/yhbian/B01_Copy_API_Refactor/EdgeVisor
PORT1=19711
PORT2=19712
SOCK=/tmp/dllama_bench_cpu_heads.sock
PROMPT="Write a comma-separated list of the numbers from 1 to 20."
rm -f "${SOCK}"
./dllama worker --port "${PORT1}" --nthreads 1 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/05_cpu_uneven_dynamic/worker_cpu1.log" 2>&1 &
./dllama worker --port "${PORT2}" --nthreads 1 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/05_cpu_uneven_dynamic/worker_cpu2.log" 2>&1 &
sleep 4
DLLAMA_PLAN_CTRL_SOCKET="${SOCK}" ./dllama inference \
  --prompt "${PROMPT}" \
  --steps 64 \
  --model "/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m" \
  --tokenizer "/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t" \
  --buffer-float-type q80 \
  --nthreads 1 \
  --max-seq-len 512 \
  --temperature 0 \
  --seed 1 \
  --workers "127.0.0.1:${PORT1}" "127.0.0.1:${PORT2}" \
  --benchmark \
  --ratios "2:3:3" \
  --enable-stage-full-weights \
  --enable-plan-barrier \
  --enable-kv-redundancy-during-migration 1 \
  --kv-redundancy 2 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/05_cpu_uneven_dynamic/root_cpu0.log" 2>&1 &
```

### 1.2 中间动态迁移 UDS 输入

```bash
python3 examples/plan-uds-client.py "${SOCK}" ping
python3 examples/plan-uds-client.py "${SOCK}" set_plan \
  --seq 601 \
  --mode next_barrier \
  --stage 0 \
  --from 1 \
  --to 2 \
  --kind 1 \
  --heads 1 \
  --ffn 0
```

### 1.3 UDS 返回

ping 返回：

```json
{
  "ok": true
}
```

set_plan 返回：

```json
{
  "cacheSeq": 2,
  "cmd": {
    "cmdKind": 1,
    "fromNodeIndex": 1,
    "mode": "next_barrier",
    "nFfnToMove": 0,
    "nHeadsToMove": 1,
    "reserved0": 0,
    "seq": 601,
    "stageIndex": 0,
    "toNodeIndex": 2,
    "triggerLayer": 4294967295,
    "triggerPos": 4294967295
  },
  "ok": true
}
```

## 2. 程序输出

原始输出已过滤 executor 初始化与 logits 诊断，仅保留 UDS 监听、plan emit/apply/work、每个 token 行和 benchmark 汇总。

```text
RC=0
[plan-uds] listening on /tmp/dllama_bench_cpu_heads.sock
🛑 Stop: <|end_of_text|>
🛑 Stop: <|eot_id|>
⭐ Chat template: llama3
🧭 [plan][emit] node=0 stage=0 layer=0 pos=0 epoch=1 kind=head move=1 from=1 to=2
🧭 [plan][apply] node=0 stage=0 epoch=1 layer=0 pos=0 kind=head headMove=1 ffnMove=0 from=1 to=2 (gqaGroup=3)
📊 [plan][work] node=0 stage=0 epoch=1 layer=0 pos=0 kvHeads=2 qHeads=6 ffnDim=2048 reason=legacy
🔷️ Eval 6528 ms Sync 3993 ms | Sent  8803 kB Recv 16854 kB | (23 tokens)
🔶 Pred  804 ms Sync   48 ms | pos=23 | Sent   385 kB Recv   734 kB | 1
🔶 Pred  301 ms Sync  139 ms | pos=24 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  281 ms Sync  157 ms | pos=25 | Sent   382 kB Recv   734 kB |  
🔶 Pred  312 ms Sync  132 ms | pos=26 | Sent   382 kB Recv   734 kB | 2
🔶 Pred  281 ms Sync  158 ms | pos=27 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  282 ms Sync  155 ms | pos=28 | Sent   382 kB Recv   734 kB |  
🔶 Pred  293 ms Sync  147 ms | pos=29 | Sent   382 kB Recv   734 kB | 3
🔶 Pred  283 ms Sync  155 ms | pos=30 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  281 ms Sync  155 ms | pos=31 | Sent   382 kB Recv   734 kB |  
🔶 Pred  311 ms Sync  165 ms | pos=32 | Sent   382 kB Recv   734 kB | 4
🔶 Pred  296 ms Sync  139 ms | pos=33 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  283 ms Sync  153 ms | pos=34 | Sent   382 kB Recv   734 kB |  
🔶 Pred  283 ms Sync  151 ms | pos=35 | Sent   382 kB Recv   734 kB | 5
🔶 Pred  283 ms Sync  151 ms | pos=36 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  315 ms Sync  126 ms | pos=37 | Sent   382 kB Recv   734 kB |  
🔶 Pred  283 ms Sync  151 ms | pos=38 | Sent   382 kB Recv   734 kB | 6
🔶 Pred  283 ms Sync  151 ms | pos=39 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  283 ms Sync  151 ms | pos=40 | Sent   382 kB Recv   734 kB |  
🔶 Pred  283 ms Sync  151 ms | pos=41 | Sent   382 kB Recv   734 kB | 7
🔶 Pred  315 ms Sync  126 ms | pos=42 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  283 ms Sync  151 ms | pos=43 | Sent   382 kB Recv   734 kB |  
🔶 Pred  283 ms Sync  151 ms | pos=44 | Sent   382 kB Recv   734 kB | 8
🔶 Pred  283 ms Sync  151 ms | pos=45 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  283 ms Sync  151 ms | pos=46 | Sent   382 kB Recv   734 kB |  
🔶 Pred  316 ms Sync  125 ms | pos=47 | Sent   382 kB Recv   734 kB | 9
🔶 Pred  283 ms Sync  151 ms | pos=48 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  283 ms Sync  151 ms | pos=49 | Sent   382 kB Recv   734 kB |  
🔶 Pred  282 ms Sync  152 ms | pos=50 | Sent   382 kB Recv   734 kB | 10
🔶 Pred  281 ms Sync  154 ms | pos=51 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  308 ms Sync  133 ms | pos=52 | Sent   382 kB Recv   734 kB |  
🔶 Pred  286 ms Sync  149 ms | pos=53 | Sent   382 kB Recv   734 kB | 11
🔶 Pred  281 ms Sync  154 ms | pos=54 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  281 ms Sync  154 ms | pos=55 | Sent   382 kB Recv   734 kB |  
🔶 Pred  281 ms Sync  154 ms | pos=56 | Sent   382 kB Recv   734 kB | 12
🔶 Pred  309 ms Sync  132 ms | pos=57 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  286 ms Sync  150 ms | pos=58 | Sent   382 kB Recv   734 kB |  
🔶 Pred  281 ms Sync  154 ms | pos=59 | Sent   382 kB Recv   734 kB | 13
🔶 Pred  281 ms Sync  154 ms | pos=60 | Sent   382 kB Recv   734 kB | ,
🔶 Pred  281 ms Sync  154 ms | pos=61 | Sent   382 kB Recv   734 kB |  
🔶 Pred  297 ms Sync  142 ms | pos=62 | Sent   382 kB Recv   734 kB | 14
🔶 Pred  281 ms Sync  155 ms | pos=63 | Sent   382 kB Recv   734 kB | ,
Evaluation
   nBatches: 32
    nTokens: 23
   tokens/s: 2.19 (457.49 ms/tok)
Prediction
    nTokens: 41
   tokens/s: 2.23 (448.68 ms/tok)
⏱️  [Stage/Node Profile Summary]
  • Stage 0 Node 0: per-fwd total=688.53 ms (exec=450.48 sync=238.04) | per-tok total=451.84 ms (exec=295.63 sync=156.21) | fwd=42 tok=64
  • Stage 0 Node 1: per-fwd total=687.98 ms (exec=599.15 sync= 88.83) | per-tok total=451.49 ms (exec=393.19 sync= 58.29) | fwd=42 tok=64
  • Stage 0 Node 2: per-fwd total=688.33 ms (exec=675.39 sync= 12.95) | per-tok total=451.72 ms (exec=443.22 sync=  8.50) | fwd=42 tok=64
Hint: prompt eval uses batchSize>1, so per-token is usually the meaningful metric for rebalancing.
```

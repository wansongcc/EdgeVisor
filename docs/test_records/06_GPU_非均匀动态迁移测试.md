# 06 GPU 非均匀动态迁移测试

- 测试日志目录：`/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/06_gpu_uneven_dynamic`
- 项目根目录：`/home/cc/yhbian/B01_Copy_API_Refactor`
- 本次测试参数：已加入 `--benchmark`

## 1. 运行的代码输入

### 1.1 init 输入

```bash
cd /home/cc/yhbian/B01_Copy_API_Refactor/EdgeVisor
PORT1=19701
PORT2=19702
SOCK=/tmp/dllama_bench_gpu_heads.sock
PROMPT="Write a comma-separated list of the numbers from 1 to 20."
rm -f "${SOCK}"
./dllama worker --port "${PORT1}" --nthreads 1 --gpu-index 1 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/06_gpu_uneven_dynamic/worker_gpu1.log" 2>&1 &
./dllama worker --port "${PORT2}" --nthreads 1 --gpu-index 2 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/06_gpu_uneven_dynamic/worker_gpu2.log" 2>&1 &
sleep 4
DLLAMA_PLAN_CTRL_SOCKET="${SOCK}" ./dllama inference \
  --prompt "${PROMPT}" \
  --steps 96 \
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
  --ratios "2:3:3" \
  --enable-stage-full-weights \
  --enable-plan-barrier \
  --enable-kv-redundancy-during-migration 1 \
  --kv-redundancy 2 >"/home/cc/yhbian/B01_Copy_API_Refactor/runtime_logs/benchmark_docs_20260531_071456/06_gpu_uneven_dynamic/root_gpu0.log" 2>&1 &
```

### 1.2 中间动态迁移 UDS 输入

```bash
python3 examples/plan-uds-client.py "${SOCK}" ping
python3 examples/plan-uds-client.py "${SOCK}" set_plan \
  --seq 501 \
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
    "seq": 501,
    "stageIndex": 0,
    "toNodeIndex": 2,
    "triggerLayer": 4294967295,
    "triggerPos": 4294967295
  },
  "ok": true
}
```

## 2. 程序输出

原始输出已过滤 executor 初始化与 logits 诊断，仅保留 UDS 监听、plan emit/apply、每个 token 行和 benchmark 汇总。

```text
RC=0
[plan-uds] listening on /tmp/dllama_bench_gpu_heads.sock
🛑 Stop: <|end_of_text|>
🛑 Stop: <|eot_id|>
⭐ Chat template: llama3
🧭 [plan][emit][gpu] node=0 stage=0 layer=0 pos=0 epoch=1 kind=1 headMove=1 ffnMove=0 from=1 to=2
🧭 [plan][apply][gpu] node=0 stage=0 epoch=1 layer=0 pos=0 cmd=1 kvHeads=2 qHeads=6 ffnDim=2048
🔷️ Eval11895 ms Sync 1572 ms | Sent  8803 kB Recv 16854 kB | (23 tokens)
🔶 Pred  562 ms Sync   28 ms | pos=23 | Sent   385 kB Recv   734 kB | 1
🔶 Pred   90 ms Sync   53 ms | pos=24 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   27 ms Sync   18 ms | pos=25 | Sent   382 kB Recv   734 kB |  
🔶 Pred   26 ms Sync    6 ms | pos=26 | Sent   382 kB Recv   734 kB | 2
🔶 Pred   26 ms Sync    6 ms | pos=27 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   26 ms Sync    6 ms | pos=28 | Sent   382 kB Recv   734 kB |  
🔶 Pred   26 ms Sync    9 ms | pos=29 | Sent   382 kB Recv   734 kB | 3
🔶 Pred   25 ms Sync    8 ms | pos=30 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   26 ms Sync    8 ms | pos=31 | Sent   382 kB Recv   734 kB |  
🔶 Pred   25 ms Sync    7 ms | pos=32 | Sent   382 kB Recv   734 kB | 4
🔶 Pred   25 ms Sync    7 ms | pos=33 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   24 ms Sync    8 ms | pos=34 | Sent   382 kB Recv   734 kB |  
🔶 Pred   21 ms Sync    9 ms | pos=35 | Sent   382 kB Recv   734 kB | 5
🔶 Pred   21 ms Sync    7 ms | pos=36 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   21 ms Sync    8 ms | pos=37 | Sent   382 kB Recv   734 kB |  
🔶 Pred   20 ms Sync    8 ms | pos=38 | Sent   382 kB Recv   734 kB | 6
🔶 Pred   20 ms Sync    7 ms | pos=39 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   20 ms Sync    9 ms | pos=40 | Sent   382 kB Recv   734 kB |  
🔶 Pred   20 ms Sync    9 ms | pos=41 | Sent   382 kB Recv   734 kB | 7
🔶 Pred   19 ms Sync    8 ms | pos=42 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   20 ms Sync   11 ms | pos=43 | Sent   382 kB Recv   734 kB |  
🔶 Pred   19 ms Sync    8 ms | pos=44 | Sent   382 kB Recv   734 kB | 8
🔶 Pred   19 ms Sync    8 ms | pos=45 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   18 ms Sync    9 ms | pos=46 | Sent   382 kB Recv   734 kB |  
🔶 Pred   19 ms Sync    9 ms | pos=47 | Sent   382 kB Recv   734 kB | 9
🔶 Pred   17 ms Sync   10 ms | pos=48 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   19 ms Sync   12 ms | pos=49 | Sent   382 kB Recv   734 kB |  
🔶 Pred   17 ms Sync   10 ms | pos=50 | Sent   382 kB Recv   734 kB | 10
🔶 Pred   17 ms Sync   10 ms | pos=51 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   18 ms Sync   10 ms | pos=52 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   10 ms | pos=53 | Sent   382 kB Recv   734 kB | 11
🔶 Pred   17 ms Sync   10 ms | pos=54 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   19 ms Sync   12 ms | pos=55 | Sent   382 kB Recv   734 kB |  
🔶 Pred   17 ms Sync   11 ms | pos=56 | Sent   382 kB Recv   734 kB | 12
🔶 Pred   17 ms Sync   11 ms | pos=57 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   18 ms Sync   11 ms | pos=58 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   10 ms | pos=59 | Sent   382 kB Recv   734 kB | 13
🔶 Pred   18 ms Sync   11 ms | pos=60 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   19 ms Sync   12 ms | pos=61 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   11 ms | pos=62 | Sent   382 kB Recv   734 kB | 14
🔶 Pred   18 ms Sync   11 ms | pos=63 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   17 ms Sync   11 ms | pos=64 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   11 ms | pos=65 | Sent   382 kB Recv   734 kB | 15
🔶 Pred   18 ms Sync   11 ms | pos=66 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   19 ms Sync   11 ms | pos=67 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   13 ms | pos=68 | Sent   382 kB Recv   734 kB | 16
🔶 Pred   18 ms Sync   11 ms | pos=69 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   18 ms Sync   11 ms | pos=70 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   11 ms | pos=71 | Sent   382 kB Recv   734 kB | 17
🔶 Pred   18 ms Sync   11 ms | pos=72 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   19 ms Sync   11 ms | pos=73 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   13 ms | pos=74 | Sent   382 kB Recv   734 kB | 18
🔶 Pred   18 ms Sync   11 ms | pos=75 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   18 ms Sync   11 ms | pos=76 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   10 ms | pos=77 | Sent   382 kB Recv   734 kB | 19
🔶 Pred   18 ms Sync   11 ms | pos=78 | Sent   382 kB Recv   734 kB | ,
🔶 Pred   19 ms Sync   11 ms | pos=79 | Sent   382 kB Recv   734 kB |  
🔶 Pred   18 ms Sync   12 ms | pos=80 | Sent   382 kB Recv   734 kB | 20
🔶 Pred   18 ms Sync   11 ms | pos=81 | Sent   382 kB Recv   734 kB | [EOS]
Evaluation
   nBatches: 32
    nTokens: 23
   tokens/s: 1.71 (585.59 ms/tok)
Prediction
    nTokens: 59
   tokens/s: 23.81 (41.99 ms/tok)
⏱️  [Stage/Node Profile Summary]
  • Stage 0 Node 0: per-fwd total=265.77 ms (exec=228.32 sync= 37.45) | per-tok total=194.46 ms (exec=167.06 sync= 27.40) | fwd=60 tok=82
  • Stage 0 Node 1: per-fwd total=255.08 ms (exec=229.25 sync= 25.82) | per-tok total=186.64 ms (exec=167.75 sync= 18.90) | fwd=60 tok=82
  • Stage 0 Node 2: per-fwd total=251.01 ms (exec=233.20 sync= 17.81) | per-tok total=183.67 ms (exec=170.64 sync= 13.03) | fwd=60 tok=82
Hint: prompt eval uses batchSize>1, so per-token is usually the meaningful metric for rebalancing.
```

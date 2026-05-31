#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

KEEP_PREFIXES = ('🔷', '🔶', '🛑', '⭐', '🧭', '📊', '[plan-uds]', '⏱️', 'Hint:')
KEEP_EXACT = {'Evaluation', 'Prediction'}


def read(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def filtered_output(log_path: Path) -> str:
    lines: list[str] = []
    for raw in read(log_path).splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped in KEEP_EXACT:
            lines.append(raw)
        elif stripped.startswith(KEEP_PREFIXES):
            lines.append(raw)
        elif stripped.startswith(('nBatches:', 'nTokens:', 'tokens/s:')):
            lines.append(raw)
        elif stripped.startswith('• Stage') or stripped.startswith('Stage 0 Node') or stripped.startswith('  • Stage'):
            lines.append(raw)
    return '\n'.join(lines)


def block(text: str, lang: str = 'bash') -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def rc(logs: Path, case: str) -> str:
    return read(logs / case / 'rc.txt').strip()


def header(title: str, logs: Path, case: str, root: Path) -> str:
    return (
        f"# {title}\n\n"
        f"- 测试日志目录：`{logs / case}`\n"
        f"- 项目根目录：`{root}`\n"
        f"- 本次测试参数：已加入 `--benchmark`\n\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate six EdgeVisor semantic benchmark test records.')
    parser.add_argument('--logs', required=True, type=Path, help='Output directory from run_six_benchmark_tests.sh')
    parser.add_argument('--out', required=True, type=Path, help='Directory for Markdown records')
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--model', default='/home/cc/dllama/distributed-llama/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m')
    parser.add_argument('--tokenizer', default='/home/cc/dllama/distributed-llama/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t')
    args = parser.parse_args()

    logs = args.logs.resolve()
    out = args.out.resolve()
    root = args.project_root.resolve()
    engine = root / 'EdgeVisor'
    out.mkdir(parents=True, exist_ok=True)

    model = args.model
    tok = args.tokenizer

    cpu_single_cmd = f'''cd {engine}
./dllama inference \\
  --prompt "What is 2+2? Answer with only the number." \\
  --steps 32 \\
  --model "{model}" \\
  --tokenizer "{tok}" \\
  --buffer-float-type q80 \\
  --nthreads 2 \\
  --max-seq-len 512 \\
  --temperature 0 \\
  --seed 1 \\
  --benchmark >"{logs}/01_cpu_single/root.log" 2>&1'''

    gpu_single_cmd = f'''cd {engine}
./dllama inference \\
  --prompt "What is 2+2? Answer with only the number." \\
  --steps 32 \\
  --model "{model}" \\
  --tokenizer "{tok}" \\
  --buffer-float-type q80 \\
  --nthreads 1 \\
  --max-seq-len 512 \\
  --temperature 0 \\
  --seed 1 \\
  --gpu-index 0 \\
  --benchmark >"{logs}/02_gpu_single/root.log" 2>&1'''

    cpu_static_cmd = f'''cd {engine}
PORT1=19601
PORT2=19602
./dllama worker --port "${{PORT1}}" --nthreads 2 >"{logs}/03_cpu_uneven_static/worker1.log" 2>&1 &
./dllama worker --port "${{PORT2}}" --nthreads 2 >"{logs}/03_cpu_uneven_static/worker2.log" 2>&1 &
sleep 4
./dllama inference \\
  --prompt "What is 2+2? Answer with only the number." \\
  --steps 32 \\
  --model "{model}" \\
  --tokenizer "{tok}" \\
  --buffer-float-type q80 \\
  --nthreads 2 \\
  --max-seq-len 512 \\
  --temperature 0 \\
  --seed 1 \\
  --workers "127.0.0.1:${{PORT1}}" "127.0.0.1:${{PORT2}}" \\
  --benchmark \\
  --ratios "2:3:3" >"{logs}/03_cpu_uneven_static/root.log" 2>&1'''

    gpu_static_cmd = f'''cd {engine}
PORT1=19501
PORT2=19502
./dllama worker --port "${{PORT1}}" --nthreads 1 --gpu-index 1 >"{logs}/04_gpu_uneven_static/worker1.log" 2>&1 &
./dllama worker --port "${{PORT2}}" --nthreads 1 --gpu-index 2 >"{logs}/04_gpu_uneven_static/worker2.log" 2>&1 &
sleep 4
./dllama inference \\
  --prompt "What is 2+2? Answer with only the number." \\
  --steps 64 \\
  --model "{model}" \\
  --tokenizer "{tok}" \\
  --buffer-float-type q80 \\
  --nthreads 1 \\
  --max-seq-len 512 \\
  --temperature 0 \\
  --seed 1 \\
  --gpu-index 0 \\
  --workers "127.0.0.1:${{PORT1}}" "127.0.0.1:${{PORT2}}" \\
  --benchmark \\
  --ratios "2:3:3" >"{logs}/04_gpu_uneven_static/root.log" 2>&1'''

    cpu_dyn_init = f'''cd {engine}
PORT1=19711
PORT2=19712
SOCK=/tmp/dllama_bench_cpu_heads.sock
PROMPT="Write a comma-separated list of the numbers from 1 to 20."
rm -f "${{SOCK}}"
./dllama worker --port "${{PORT1}}" --nthreads 1 >"{logs}/05_cpu_uneven_dynamic/worker_cpu1.log" 2>&1 &
./dllama worker --port "${{PORT2}}" --nthreads 1 >"{logs}/05_cpu_uneven_dynamic/worker_cpu2.log" 2>&1 &
sleep 4
DLLAMA_PLAN_CTRL_SOCKET="${{SOCK}}" ./dllama inference \\
  --prompt "${{PROMPT}}" \\
  --steps 64 \\
  --model "{model}" \\
  --tokenizer "{tok}" \\
  --buffer-float-type q80 \\
  --nthreads 1 \\
  --max-seq-len 512 \\
  --temperature 0 \\
  --seed 1 \\
  --workers "127.0.0.1:${{PORT1}}" "127.0.0.1:${{PORT2}}" \\
  --benchmark \\
  --ratios "2:3:3" \\
  --enable-stage-full-weights \\
  --enable-plan-barrier \\
  --enable-kv-redundancy-during-migration 1 \\
  --kv-redundancy 2 >"{logs}/05_cpu_uneven_dynamic/root_cpu0.log" 2>&1 &'''

    gpu_dyn_init = f'''cd {engine}
PORT1=19701
PORT2=19702
SOCK=/tmp/dllama_bench_gpu_heads.sock
PROMPT="Write a comma-separated list of the numbers from 1 to 20."
rm -f "${{SOCK}}"
./dllama worker --port "${{PORT1}}" --nthreads 1 --gpu-index 1 >"{logs}/06_gpu_uneven_dynamic/worker_gpu1.log" 2>&1 &
./dllama worker --port "${{PORT2}}" --nthreads 1 --gpu-index 2 >"{logs}/06_gpu_uneven_dynamic/worker_gpu2.log" 2>&1 &
sleep 4
DLLAMA_PLAN_CTRL_SOCKET="${{SOCK}}" ./dllama inference \\
  --prompt "${{PROMPT}}" \\
  --steps 96 \\
  --model "{model}" \\
  --tokenizer "{tok}" \\
  --buffer-float-type q80 \\
  --nthreads 1 \\
  --max-seq-len 512 \\
  --temperature 0 \\
  --seed 1 \\
  --gpu-index 0 \\
  --workers "127.0.0.1:${{PORT1}}" "127.0.0.1:${{PORT2}}" \\
  --benchmark \\
  --ratios "2:3:3" \\
  --enable-stage-full-weights \\
  --enable-plan-barrier \\
  --enable-kv-redundancy-during-migration 1 \\
  --kv-redundancy 2 >"{logs}/06_gpu_uneven_dynamic/root_gpu0.log" 2>&1 &'''

    cpu_dyn_uds = '''python3 examples/plan-uds-client.py "${SOCK}" ping
python3 examples/plan-uds-client.py "${SOCK}" set_plan \\
  --seq 601 \\
  --mode next_barrier \\
  --stage 0 \\
  --from 1 \\
  --to 2 \\
  --kind 1 \\
  --heads 1 \\
  --ffn 0'''
    gpu_dyn_uds = cpu_dyn_uds.replace('--seq 601', '--seq 501')

    docs: dict[str, str] = {}
    docs['01_CPU_单机测试.md'] = (
        header('01 CPU 单机测试', logs, '01_cpu_single', root)
        + '## 1. 运行的代码输入\n\n' + block(cpu_single_cmd) + '\n\n'
        + '## 2. 程序输出\n\n'
        + '原始输出已过滤 executor 初始化与 logits 诊断，仅保留停止符、chat template、Eval/Pred token 行和 benchmark 汇总。\n\n'
        + block('RC=' + rc(logs, '01_cpu_single') + '\n' + filtered_output(logs / '01_cpu_single' / 'root.log'), 'text') + '\n'
    )
    docs['02_GPU_单机测试.md'] = (
        header('02 GPU 单机测试', logs, '02_gpu_single', root)
        + '## 1. 运行的代码输入\n\n' + block(gpu_single_cmd) + '\n\n'
        + '## 2. 程序输出\n\n'
        + '原始输出已过滤 executor 初始化与 logits 诊断，仅保留停止符、chat template、Eval/Pred token 行和 benchmark 汇总。\n\n'
        + block('RC=' + rc(logs, '02_gpu_single') + '\n' + filtered_output(logs / '02_gpu_single' / 'root.log'), 'text') + '\n'
    )
    docs['03_CPU_非均匀静态测试.md'] = (
        header('03 CPU 非均匀静态测试', logs, '03_cpu_uneven_static', root)
        + '## 1. 运行的代码输入\n\n' + block(cpu_static_cmd) + '\n\n'
        + '## 2. 程序输出\n\n'
        + '原始输出已过滤 executor 初始化与 logits 诊断，仅保留停止符、chat template、Eval/Pred token 行和 benchmark 汇总。\n\n'
        + block('RC=' + rc(logs, '03_cpu_uneven_static') + '\n' + filtered_output(logs / '03_cpu_uneven_static' / 'root.log'), 'text') + '\n'
    )
    docs['04_GPU_非均匀静态测试.md'] = (
        header('04 GPU 非均匀静态测试', logs, '04_gpu_uneven_static', root)
        + '## 1. 运行的代码输入\n\n' + block(gpu_static_cmd) + '\n\n'
        + '## 2. 程序输出\n\n'
        + '原始输出已过滤 executor 初始化与 logits 诊断，仅保留停止符、chat template、Eval/Pred token 行和 benchmark 汇总。\n\n'
        + block('RC=' + rc(logs, '04_gpu_uneven_static') + '\n' + filtered_output(logs / '04_gpu_uneven_static' / 'root.log'), 'text') + '\n'
    )
    docs['05_CPU_非均匀动态迁移测试.md'] = (
        header('05 CPU 非均匀动态迁移测试', logs, '05_cpu_uneven_dynamic', root)
        + '## 1. 运行的代码输入\n\n### 1.1 init 输入\n\n' + block(cpu_dyn_init) + '\n\n'
        + '### 1.2 中间动态迁移 UDS 输入\n\n' + block(cpu_dyn_uds) + '\n\n'
        + '### 1.3 UDS 返回\n\nping 返回：\n\n' + block(read(logs / '05_cpu_uneven_dynamic' / 'uds_ping.json').strip(), 'json') + '\n\n'
        + 'set_plan 返回：\n\n' + block(read(logs / '05_cpu_uneven_dynamic' / 'uds_set_plan.json').strip(), 'json') + '\n\n'
        + '## 2. 程序输出\n\n'
        + '原始输出已过滤 executor 初始化与 logits 诊断，仅保留 UDS 监听、plan emit/apply/work、每个 token 行和 benchmark 汇总。\n\n'
        + block('RC=' + rc(logs, '05_cpu_uneven_dynamic') + '\n' + filtered_output(logs / '05_cpu_uneven_dynamic' / 'root_cpu0.log'), 'text') + '\n'
    )
    docs['06_GPU_非均匀动态迁移测试.md'] = (
        header('06 GPU 非均匀动态迁移测试', logs, '06_gpu_uneven_dynamic', root)
        + '## 1. 运行的代码输入\n\n### 1.1 init 输入\n\n' + block(gpu_dyn_init) + '\n\n'
        + '### 1.2 中间动态迁移 UDS 输入\n\n' + block(gpu_dyn_uds) + '\n\n'
        + '### 1.3 UDS 返回\n\nping 返回：\n\n' + block(read(logs / '06_gpu_uneven_dynamic' / 'uds_ping.json').strip(), 'json') + '\n\n'
        + 'set_plan 返回：\n\n' + block(read(logs / '06_gpu_uneven_dynamic' / 'uds_set_plan.json').strip(), 'json') + '\n\n'
        + '## 2. 程序输出\n\n'
        + '原始输出已过滤 executor 初始化与 logits 诊断，仅保留 UDS 监听、plan emit/apply、每个 token 行和 benchmark 汇总。\n\n'
        + block('RC=' + rc(logs, '06_gpu_uneven_dynamic') + '\n' + filtered_output(logs / '06_gpu_uneven_dynamic' / 'root_gpu0.log'), 'text') + '\n'
    )

    for name, content in docs.items():
        (out / name).write_text(content, encoding='utf-8')
        print(out / name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

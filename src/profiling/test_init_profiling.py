import subprocess
import sys
import json
import time
import signal
import os
import glob

# 配置参数
NUM_DEVICES = 8
# 假定 init_profiling.py 在当前脚本的同一目录下
SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_profiling.py")

processes = []

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号，确保清理子进程"""
    print("\nCtrl+C captured, terminating processes...")
    for p in processes:
        if p.poll() is None: # 仍在运行
            try:
                p.terminate() 
            except Exception:
                pass
    
    # 给一点时间让进程退出
    time.sleep(1)
    
    for p in processes:
        if p.poll() is None:
            try:
                p.kill() # 强制 kill
            except Exception:
                pass
    sys.exit(0)

def run_simulation():
    # 清理之前残留的结果文件
    for f in glob.glob("profile_result_id_*.json"):
        try:
            os.remove(f)
        except OSError:
            pass

    # 1. 启动子进程（顺序执行，避免同时跑大矩阵乘法导致机器过载）
    print(f"Launching {NUM_DEVICES} processes (sequential)...")
    try:
        for i in range(1, NUM_DEVICES + 1):
            cmd = [
                sys.executable, 
                SCRIPT_PATH,
                "--total", str(NUM_DEVICES),
                "--id", str(i),
            ]
            
            # 启动进程
            p = subprocess.Popen(cmd)
            processes.append(p)
            print(f"Started process {i} (PID: {p.pid})")
            p.wait()
        print("All processes completed.")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        signal_handler(None, None)

    # 2. 结果汇总验证
    print("\n" + "="*90)
    print("SIMULATION RESULTS SUMMARY")
    print("="*90)
    
    # 表头
    # ID | Memory (GB) | Compute (GFLOPS) | Attention Timing
    headers = ["ID", "Mem(GB)", "GFLOPS", "T_kv(us)", "T_full(us)", "Gamma"]
    print(
        f"{headers[0]:<4} | {headers[1]:<8} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<11} | {headers[5]}"
    )
    print("-" * 90)

    results = []
    # 读取预期生成的 8 个文件
    for i in range(1, NUM_DEVICES + 1):
        fname = f"profile_result_id_{i}.json"
        
        # 因为脚本在当前目录运行，结果文件应该也在当前目录
        # 如果不是，可能需要调整路径查找
        if os.path.exists(fname):
            try:
                with open(fname, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except json.JSONDecodeError:
                 print(f"Warning: Result file {fname} is broken.")
        else:
            print(f"Warning: Result file {fname} not found.")

    # 按 Device ID 排序
    results.sort(key=lambda x: x['id'])

    for r in results:
        dev_id = r.get('id')
        mem_gb = r.get('memory_bytes', 0) / (1024 ** 3)
        comp = r.get('compute_gflops', 0.0)
        t_kv = r.get('t_kv_us', 0.0)
        t_full = r.get('t_full_us', 0.0)
        gamma = r.get('gamma_ratio', 0.0)

        print(f"{dev_id:<4} | {mem_gb:<8.2f} | {comp:<10.2f} | {t_kv:<10.4f} | {t_full:<11.4f} | {gamma:.4f}")

    print("="*90)
    print("Simulation Finished.")

if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    run_simulation()

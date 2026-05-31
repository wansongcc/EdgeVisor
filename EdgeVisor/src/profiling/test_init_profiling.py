import subprocess
import sys
import json
import time
import signal
import os
import glob
import psutil

# 配置参数
NUM_DEVICES = 8
BASE_PORT = 8001
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
    # 1. 生成配置
    config = {}
    for i in range(1, NUM_DEVICES + 1):
        # 端口从 8001 到 8008
        config[str(i)] = f"127.0.0.1:{BASE_PORT + i - 1}"
    
    config_json = json.dumps(config)
    print(f"Generated Config: {config_json}")
    
    # 清理之前残留的结果文件
    for f in glob.glob("profile_result_*.json"):
        try:
            os.remove(f)
        except OSError:
            pass

    # 2. 并发启动子进程
    print(f"Launching {NUM_DEVICES} processes...")
    try:
        for i in range(1, NUM_DEVICES + 1):
            cmd = [
                sys.executable, 
                SCRIPT_PATH,
                "--total_devices", str(NUM_DEVICES),
                "--id", str(i),
                "--ip_list", config_json
            ]
            
            # 启动进程
            p = subprocess.Popen(cmd)
            processes.append(p)
            print(f"Started process {i} (PID: {p.pid})")
            
        # 3. 进程管理：等待所有子进程结束
        print("Waiting for completion (this may take a while)...")
        for p in processes:
            p.wait()
        print("All processes completed.")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        signal_handler(None, None)

    # 4. 结果汇总验证
    print("\n" + "="*90)
    print("SIMULATION RESULTS SUMMARY")
    print("="*90)
    
    # 表头
    # ID | Memory (GB) | Compute (GFLOPS) | Bandwidth to Peers (MB/s)
    headers = ["ID", "Mem(GB)", "GFLOPS", "Bandwidth (MB/s) -> [TargetID:Speed]"]
    print(f"{headers[0]:<4} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]}")
    print("-" * 90)

    results = []
    # 读取预期生成的 8 个文件
    for i in range(1, NUM_DEVICES + 1):
        fname = f"profile_result_{i}.json"
        
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
    results.sort(key=lambda x: x['device_id'])

    for r in results:
        dev_id = r['device_id']
        mem = r['memory_available_gb']
        comp = r['compute_gflops']
        
        # 格式化带宽信息
        bw_str_parts = []
        bw_map = r['bandwidth_mbps'] # 这里存的是 Bytes/s
        
        # 按 Target ID 排序展示
        sorted_targets = sorted(bw_map.keys(), key=lambda x: int(x))
        for t_id in sorted_targets:
            speed_bps = bw_map[t_id]
            speed_mbps = speed_bps / (1024 * 1024)
            bw_str_parts.append(f"{t_id}:{speed_mbps:.1f}")
            
        bw_str = ", ".join(bw_str_parts)
        
        print(f"{dev_id:<4} | {mem:<8.2f} | {comp:<8.2f} | {bw_str}")

    print("="*90)
    print("Simulation Finished.")

if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    run_simulation()

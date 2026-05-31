import socket
import threading
import time
import random
import json
import argparse
import numpy as np
import psutil
import os

# ================= 配置区域 =================
# 所有设备统一监听的端口
PORT = 9999
# 测试带宽用的数据包大小 (10MB)
TEST_DATA_SIZE = 10 * 1024 * 1024 
# IP 前缀模板 (假设 ID=8 -> 192.168.1.8)
IP_TEMPLATE = "192.168.1.{}"
# 算力测试矩阵大小 (2048 x 2048 float32)
# 如果是极弱的设备(如树莓派Zero)，建议改为 1024
MATRIX_SIZE = 2048
# ===========================================

class BandwidthServer(threading.Thread):
    """
    后台守护线程:负责接收其他设备的连接,消耗数据,充当测速靶子.
    """
    def __init__(self, port):
        super().__init__()
        self.port = port
        self.daemon = True # 主程序退出时自动退出
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 允许端口复用，防止重启脚本时报 Address already in use
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def run(self):
        try:
            # 绑定 0.0.0.0 以允许外部连接
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(50) 
            print(f"[Server] 后台接收服务已启动 (Port {self.port})")
            
            while True:
                client_sock, _ = self.server_socket.accept()
                # 开启新线程处理该连接，实现并发接收，避免阻塞
                threading.Thread(target=self.handle_client, args=(client_sock,)).start()
        except Exception as e:
            print(f"[Server Error] {e}")

    def handle_client(self, conn):
        try:
            while True:
                # 循环读取数据并丢弃 (Buffer 设大一点提高处理效率)
                data = conn.recv(65536)
                if not data:
                    break
        except:
            pass
        finally:
            conn.close()

def get_memory_info():
    """
    获取系统可用内存 (Bytes)
    """
    try:
        # psutil 在 Linux/Mac 上通用
        mem = psutil.virtual_memory()
        return mem.available
    except Exception as e:
        print(f"[Error] 获取内存失败: {e}")
        return 0

def measure_compute_power():
    """
    执行矩阵乘法测试算力 (GFLOPS)
    """
    print(f"[Compute] 开始算力测试 (Matrix Size: {MATRIX_SIZE})...")
    
    # 1. 预热 (Warm-up)
    dummy_n = 512
    A = np.random.rand(dummy_n, dummy_n).astype(np.float32)
    B = np.random.rand(dummy_n, dummy_n).astype(np.float32)
    np.dot(A, B)

    # 2. 正式测试
    N = MATRIX_SIZE
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    start_time = time.time()
    np.dot(A, B) 
    end_time = time.time()

    duration = end_time - start_time
    # 浮点运算次数: 2 * N^3
    ops = 2.0 * (N ** 3)
    gflops = (ops / duration) / 1e9
    
    print(f"[Compute] 耗时: {duration:.4f}s, 算力: {gflops:.4f} GFLOPS")
    return gflops

def measure_attention_cost_ratio():
    """
    针对 Qwen-3-8B 模型测算 Attention Head 完整计算与 KV Cache 维护的时间比。
    Matrix Shapes (Single Token):
    - Head Dim (D) = 128 
    - Seq Len (S) = 128 (模拟推理阶段)
    - FFN Per Head (H_ffn) = 688
    
    Process A (T_kv): Maintains K, V projection
    Process B (T_full): Q, K, V, Attention, Output, FFN (SwiGLU)
    """
    print(f"[Attention] 开始 Attention 算力比例测试 (Qwen-3-8B Head Params)...")
    
    # --- 参数定义 ---
    D = 128
    S = 128
    H_ffn = 688
    
    # --- 模拟数据初始化 (float32) ---
    # 输入: Batch=1
    x = np.random.randn(1, D).astype(np.float32)

    # 权重 (Process A: KV)
    W_k = np.random.randn(D, D).astype(np.float32)
    W_v = np.random.randn(D, D).astype(np.float32)
    
    # 权重 (Process B: Others)
    W_q = np.random.randn(D, D).astype(np.float32)
    W_o = np.random.randn(D, D).astype(np.float32)
    
    # FFN Weights (SwiGLU)
    W_gate = np.random.randn(D, H_ffn).astype(np.float32) 
    W_up   = np.random.randn(D, H_ffn).astype(np.float32)
    W_down = np.random.randn(H_ffn, D).astype(np.float32)
    
    # Attention Cache (Transposed K for efficient dot product)
    K_cache_T = np.random.randn(D, S).astype(np.float32)
    V_cache   = np.random.randn(S, D).astype(np.float32)

    warmup_steps = 10
    test_steps = 200

    # ==========================================
    # 1. 测量 T_kv (仅 KV Projection)
    # 逻辑: 计算当前 Token 的 K 和 V
    # ==========================================
    
    # Warm-up
    for _ in range(warmup_steps):
        np.dot(x, W_k)
        np.dot(x, W_v)
        
    start_kv = time.perf_counter()
    for _ in range(test_steps):
        np.dot(x, W_k)
        np.dot(x, W_v)
    end_kv = time.perf_counter()
    
    t_kv_avg_s = (end_kv - start_kv) / test_steps
    t_kv_us = t_kv_avg_s * 1e6
    
    # ==========================================
    # 2. 测量 T_full (Full Head + FFN)
    # 逻辑: 包含 KV Proj, Q Proj, Attn, Output Proj, FFN(3 matmuls)
    # ==========================================

    # Warm-up
    for _ in range(warmup_steps):
        # A parts
        np.dot(x, W_k)
        np.dot(x, W_v)
        # B parts
        q = np.dot(x, W_q)
        score = np.dot(q, K_cache_T) # Score: (1, D) @ (D, S) -> (1, S)
        context = np.dot(score, V_cache) # Context: (1, S) @ (S, D) -> (1, D)
        out = np.dot(context, W_o) # Out Proj
        
        # FFN (SwiGLU)
        g = np.dot(out, W_gate)
        u = np.dot(out, W_up)
        # Simulating data dependency for Down Proj
        np.dot(u, W_down)

    start_full = time.perf_counter()
    for _ in range(test_steps):
        # 1. KV Proj (Process A is part of process B)
        np.dot(x, W_k)
        np.dot(x, W_v)
        
        # 2. Q Proj
        q = np.dot(x, W_q)
        
        # 3. Attention
        score = np.dot(q, K_cache_T)
        context = np.dot(score, V_cache)
        
        # 4. Output Proj
        # Input to FFN
        out = np.dot(context, W_o)
        
        # 5. FFN (SwiGLU)
        # Gate & Up
        g = np.dot(out, W_gate)
        u = np.dot(out, W_up)
        # Down (ignoring element-wise activation cost, focusing on matrix ops)
        np.dot(u, W_down)
        
    end_full = time.perf_counter()
    
    t_full_avg_s = (end_full - start_full) / test_steps
    t_full_us = t_full_avg_s * 1e6

    # ==========================================
    # 结果输出
    # ==========================================
    gamma = t_kv_us / t_full_us if t_full_us > 0 else 0.0
    
    print(f"[Attention] T_kv: {t_kv_us:.4f} us")
    print(f"[Attention] T_full: {t_full_us:.4f} us")
    print(f"[Attention] Gamma Ratio: {gamma:.4f}")
    
    return {
        "t_kv_us": round(t_kv_us, 4),
        "t_full_us": round(t_full_us, 4),
        "gamma_ratio": round(gamma, 4)
    }

def measure_bandwidth_to(target_ip, target_port):
    """
    测量到目标 IP 的 TCP 带宽 (Bytes/s)
    包含重试机制
    """
    dummy_data = b'x' * TEST_DATA_SIZE
    max_retries = 5 
    
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3) # 3秒连接超时
            
            # 尝试连接
            sock.connect((target_ip, target_port))
            
            # 开始测速
            start_time = time.time()
            sock.sendall(dummy_data)
            # 发送完毕确保缓冲区刷出
            sock.shutdown(socket.SHUT_WR)
            sock.close()
            end_time = time.time()
            
            duration = end_time - start_time
            if duration <= 0: duration = 0.0001
            
            bw = TEST_DATA_SIZE / duration
            return bw

        except (ConnectionRefusedError, socket.timeout, OSError):
            sock.close()
            # 对方可能没启动或正在忙，随机休眠后重试
            time.sleep(random.uniform(0.5, 1.5))
            continue
            
    # 重试多次仍失败，认为不可达
    return 0.0

def run_profiling(total_devices, my_id):
    # 1. 启动接收服务
    server = BandwidthServer(PORT)
    server.start()
    
    # 随机等待，避免所有设备完全同时开始测速造成拥塞
    # 同时给其他设备一点时间启动 Server
    print("[System] 等待集群同步...", end="", flush=True)
    time.sleep(random.uniform(2.0, 5.0))
    print(" 开始。")

    results = {
        "id": my_id,
        "ip": IP_TEMPLATE.format(my_id),
        "memory_bytes": 0,
        "compute_gflops": 0.0,
        "bandwidth_vector": [] # 索引对齐 ID
    }

    # 2. 测量本机硬件指标
    results["memory_bytes"] = get_memory_info()
    results["compute_gflops"] = measure_compute_power()

    # --- 新增: 测量 Attention 成本比例 ---
    attn_results = measure_attention_cost_ratio()
    results.update(attn_results)
    # -----------------------------------

    # 3. 测量带宽
    print("[Network] 开始全网带宽扫描...")
    # 初始化向量，长度为 total_devices + 1 (让 ID 直接对应索引，索引0空置)
    bw_list = [0.0] * (total_devices + 1)

    for target_id in range(1, total_devices + 1):
        if target_id == my_id:
            bw_list[target_id] = 0.0 # 自己到自己带宽为 0
            continue
        
        target_ip = IP_TEMPLATE.format(target_id)
        
        print(f"  -> 正在探测 ID {target_id} ({target_ip})...", end="", flush=True)
        
        # 随机抖动 (Jitter)，避免网络风暴
        time.sleep(random.uniform(0.2, 0.8))
        
        bw = measure_bandwidth_to(target_ip, PORT)
        bw_list[target_id] = bw
        
        if bw > 0:
            print(f" {bw / 1024 / 1024:.2f} MB/s")
        else:
            print(f" 连接失败 (0 MB/s)")

    results["bandwidth_vector"] = bw_list

    # 4. 输出结果
    filename = f"profile_result_id_{my_id}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[Done] 测评完成！结果已保存至 {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Environment Edge Profiler")
    parser.add_argument("--total", type=int, required=True, help="总设备数量")
    parser.add_argument("--id", type=int, required=True, help="本设备 ID (对应 192.168.1.ID)")
    
    args = parser.parse_args()
    
    run_profiling(args.total, args.id)
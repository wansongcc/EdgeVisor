import time
import json
import argparse
import numpy as np
import psutil

# ================= 配置区域 =================
# IP 前缀模板 (假设 ID=8 -> 192.168.1.8)
IP_TEMPLATE = "192.168.1.{}"
# 算力测试矩阵大小 (2048 x 2048 float32)
# 如果是极弱的设备(如树莓派Zero)，建议改为 1024
MATRIX_SIZE = 2048
# ===========================================

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
    针对 Qwen-3-8B 模型测算单个 Head 的若干计算阶段耗时（CPU / NumPy）。

    说明（均为单 token、batch=1 的矩阵乘为主的粗粒度近似）：
    - T_kv: 仅 K/V 投影（维护 KV cache 的新增计算）
    - T_attn: Attention block（Q 投影 + QK + (Score)V + 输出投影），不含 KV，不含 FFN
    - T_ffn: 仅 FFN（SwiGLU 的 3 次 matmul），不含 Attention/KV
    - T_attn_ffn: Attention block + FFN，不含 KV
    - T_full: KV + Attention block + FFN

    注意：这里刻意忽略 softmax、激活/逐元素门控等逐元素算子开销，主要聚焦 matmul 成本。
    Matrix Shapes (Single Token):
    - Head Dim (D) = 128 
    - Seq Len (S) = 128 (模拟推理阶段)
    - FFN Per Head (H_ffn) = 688
    
    Process A (T_kv): Maintains K, V projection
    Process B (T_full): KV + (Q, Attention, Output) + FFN (SwiGLU)
    """
    print(f"[Attention] 开始 Attention/FFN 耗时拆分测试 (Qwen-3-8B Head Params)...")
    
    # --- 参数定义 ---
    D = 128
    S = 128
    H_ffn = 688
    
    # --- 模拟数据初始化 (float32) ---
    # 输入: Batch=1
    x = np.random.randn(1, D).astype(np.float32)
    # FFN-only 的输入（与 Attention 解耦，避免依赖链影响测量）
    ffn_in = np.random.randn(1, D).astype(np.float32)

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
    # 1) 测量 T_kv (仅 KV Projection)
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
    # 2) 测量 T_attn (Attention block, 不含 KV / FFN)
    #    Q Proj + QK + (Score)V + Out Proj
    # ==========================================

    for _ in range(warmup_steps):
        q = np.dot(x, W_q)
        score = np.dot(q, K_cache_T)
        context = np.dot(score, V_cache)
        np.dot(context, W_o)

    start_attn = time.perf_counter()
    for _ in range(test_steps):
        q = np.dot(x, W_q)
        score = np.dot(q, K_cache_T)
        context = np.dot(score, V_cache)
        np.dot(context, W_o)
    end_attn = time.perf_counter()

    t_attn_avg_s = (end_attn - start_attn) / test_steps
    t_attn_us = t_attn_avg_s * 1e6

    # ==========================================
    # 3) 测量 T_ffn (仅 FFN, 不含 Attention / KV)
    #    Gate/Up/Down 三次 matmul
    # ==========================================

    for _ in range(warmup_steps):
        _g = np.dot(ffn_in, W_gate)
        u = np.dot(ffn_in, W_up)
        np.dot(u, W_down)

    start_ffn = time.perf_counter()
    for _ in range(test_steps):
        _g = np.dot(ffn_in, W_gate)
        u = np.dot(ffn_in, W_up)
        np.dot(u, W_down)
    end_ffn = time.perf_counter()

    t_ffn_avg_s = (end_ffn - start_ffn) / test_steps
    t_ffn_us = t_ffn_avg_s * 1e6

    # ==========================================
    # 4) 测量 T_attn_ffn (Attention block + FFN, 不含 KV)
    # ==========================================

    for _ in range(warmup_steps):
        q = np.dot(x, W_q)
        score = np.dot(q, K_cache_T)
        context = np.dot(score, V_cache)
        out = np.dot(context, W_o)
        _g = np.dot(out, W_gate)
        u = np.dot(out, W_up)
        np.dot(u, W_down)

    start_attn_ffn = time.perf_counter()
    for _ in range(test_steps):
        q = np.dot(x, W_q)
        score = np.dot(q, K_cache_T)
        context = np.dot(score, V_cache)
        out = np.dot(context, W_o)
        _g = np.dot(out, W_gate)
        u = np.dot(out, W_up)
        np.dot(u, W_down)
    end_attn_ffn = time.perf_counter()

    t_attn_ffn_avg_s = (end_attn_ffn - start_attn_ffn) / test_steps
    t_attn_ffn_us = t_attn_ffn_avg_s * 1e6
    
    # ==========================================
    # 5) 测量 T_full (KV + Attention block + FFN)
    # ==========================================

    # Warm-up
    for _ in range(warmup_steps):
        np.dot(x, W_k)
        np.dot(x, W_v)
        q = np.dot(x, W_q)
        score = np.dot(q, K_cache_T) # (1, D) @ (D, S) -> (1, S)
        context = np.dot(score, V_cache) # (1, S) @ (S, D) -> (1, D)
        out = np.dot(context, W_o)
        _g = np.dot(out, W_gate)
        u = np.dot(out, W_up)
        np.dot(u, W_down)

    start_full = time.perf_counter()
    for _ in range(test_steps):
        np.dot(x, W_k)
        np.dot(x, W_v)
        q = np.dot(x, W_q)
        score = np.dot(q, K_cache_T)
        context = np.dot(score, V_cache)
        out = np.dot(context, W_o)
        _g = np.dot(out, W_gate)
        u = np.dot(out, W_up)
        np.dot(u, W_down)
        
    end_full = time.perf_counter()
    
    t_full_avg_s = (end_full - start_full) / test_steps
    t_full_us = t_full_avg_s * 1e6

    # ==========================================
    # 结果输出
    # ==========================================
    gamma = t_kv_us / t_full_us if t_full_us > 0 else 0.0

    print(f"[Attention] T_kv: {t_kv_us:.4f} us")
    print(f"[Attention] T_attn: {t_attn_us:.4f} us")
    print(f"[Attention] T_ffn: {t_ffn_us:.4f} us")
    print(f"[Attention] T_attn+ffn: {t_attn_ffn_us:.4f} us")
    print(f"[Attention] T_full: {t_full_us:.4f} us")
    print(f"[Attention] Gamma Ratio (kv/full): {gamma:.4f}")

    return {
        "t_kv_us": round(t_kv_us, 4),
        "t_attn_us": round(t_attn_us, 4),
        "t_ffn_us": round(t_ffn_us, 4),
        "t_attn_ffn_us": round(t_attn_ffn_us, 4),
        "t_full_us": round(t_full_us, 4),
        "gamma_ratio": round(gamma, 4)
    }

def run_profiling(_total_devices, my_id):
    results = {
        "id": my_id,
        "ip": IP_TEMPLATE.format(my_id),
        "memory_bytes": 0,
        "compute_gflops": 0.0,
    }

    # 2. 测量本机硬件指标
    results["memory_bytes"] = get_memory_info()
    results["compute_gflops"] = measure_compute_power()

    # --- 新增: 测量 Attention 成本比例 ---
    attn_results = measure_attention_cost_ratio()
    results.update(attn_results)
    # -----------------------------------

    # 输出结果
    filename = f"profile_result_id_{my_id}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[Done] 测评完成！结果已保存至 {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Environment Edge Profiler")
    parser.add_argument(
        "--total",
        type=int,
        required=False,
        default=0,
        help="(已弃用) 总设备数量：网络测速已移除，此参数保留用于兼容旧命令",
    )
    parser.add_argument("--id", type=int, required=True, help="本设备 ID (对应 192.168.1.ID)")
    
    args = parser.parse_args()
    
    run_profiling(args.total, args.id)
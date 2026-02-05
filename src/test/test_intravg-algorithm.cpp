#include "../nn/nn-intravg-algorithm.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

void print_result(const std::vector<WorkerProfile>& workers, const AllocationResult& result) {
    std::cout << "Estimated Latency (T*): " << result.estimated_latency << " s" << std::endl;
    std::cout << "Worker Allocations:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    double total_alpha = 0.0;
    for (size_t i = 0; i < workers.size(); ++i) {
        std::cout << "  Worker " << workers[i].id 
                  << ": Alpha = " << result.alphas[i] 
                  << ", MaxMem = " << workers[i].max_alpha_mem << std::endl;
        total_alpha += result.alphas[i];
    }
    std::cout << "  Total Alpha Sum: " << total_alpha << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

int main() {
    // 场景演示：高算力低带宽 vs 低算力高带宽
    // 假设：
    // Worker 1 (GPU Server): 高算力 (100 TFLOPS), 低带宽 (例如 1 GB/s - 模拟远程/拥塞)
    // Worker 2 (Edge Device): 低算力 (10 TFLOPS), 高带宽 (10 GB/s - 模拟本地/快速链路)
    
    std::vector<WorkerProfile> workers = {
        {1, 100e12, 1e9, 1.0},  // 100 TFLOPS, 1 GB/s
        {2, 10e12,  10e9, 1.0}  // 10 TFLOPS, 10 GB/s
    };

    // 定义层任务
    // 假设是一个典型的 LLM 层
    // Total FLOPs: 10 TFLOPS 任务 (10e12)
    // Input Bytes: 100 MB (100e6)
    // Output Bytes: 100 MB (100e6)
    
    LayerTask task;
    task.total_flops = 10e12; 
    task.input_bytes = 100e6;
    task.output_bytes = 100e6; 

    std::cout << "Test Case 1: Standard Complementary Scenario" << std::endl;
    AllocationResult result = SolveCCWF(workers, task);
    print_result(workers, result);

    // 验证逻辑：
    // Worker 1: 算力强，但受限于带宽，接收数据慢。
    // Worker 2: 算力弱，但带宽快，能迅速开始计算。
    // 预期：Worker 2 会分担一部分任务，尽管它算力弱，因为 Worker 1 还在收数据的时候 Worker 2 已经可以跑了。
    
    // 场景 2：显存受限
    // Worker 1 虽然强，但只能存 20% 的 KV Cache / Weights
    std::vector<WorkerProfile> workers_mem_limited = {
        {1, 100e12, 1e9, 0.2},  // Max alpha 0.2
        {2, 10e12,  10e9, 1.0}
    };
    
    std::cout << "Test Case 2: Memory Constraint on Worker 1 (Max Alpha 0.2)" << std::endl;
    result = SolveCCWF(workers_mem_limited, task);
    print_result(workers_mem_limited, result);

    return 0;
}

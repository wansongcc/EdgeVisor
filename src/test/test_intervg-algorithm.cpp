#include <iostream>
#include <vector>
#include <numeric>
#include "../nn/nn-intervg-algorithm.hpp"

// 辅助打印函数
void print_allocation(const std::vector<VGProfile>& vgs, const std::vector<int>& allocation, const ModelConfig& model) {
    if (allocation.empty()) {
        std::cout << "Allocation Failed!" << std::endl;
        return;
    }

    std::cout << "Allocation Result:" << std::endl;
    double max_stage_time = 0.0;

    for (size_t i = 0; i < vgs.size(); ++i) {
        int count = allocation[i];
        double t_calc = count * vgs[i].unit_time_ms;
        double t_comm = 0.0;
        if (count > 0 && vgs[i].next_link_bw_gbps > 1e-9) {
            t_comm = (model.activation_size_gb * 8.0 / vgs[i].next_link_bw_gbps) * 1000.0;
        }
        double total = t_calc + t_comm;
        if (total > max_stage_time) max_stage_time = total;

        std::cout << "VG " << vgs[i].id << ": Layers = " << count 
                  << ", Calc = " << t_calc << " ms"
                  << ", Comm = " << t_comm << " ms"
                  << ", Total = " << total << " ms" << std::endl;
    }
    std::cout << "Pipeline Bottleneck: " << max_stage_time << " ms" << std::endl;
}

int main() {
    // 模拟场景：
    // Model: 24 Layers, 0.5 GB activation size
    ModelConfig model;
    model.total_layers = 24;
    model.activation_size_gb = 0.5; // 500MB

    // VG1: High Compute, Mid BW
    // VG2: Low Compute, High BW
    // VG3: Mid Compute, Last (BW=0/End)
    std::vector<VGProfile> vgs = {
        // id, unit_time_ms, max_layers, next_bw_gbps
        {1, 100.0, 32, 10.0}, // VG1: Strong Compute (100ms/layer), BW 10Gbps
                              // Comm overhead: 0.5*8/10 * 1000 = 400ms
        
        {2, 300.0, 32, 100.0},// VG2: Weak Compute (300ms/layer), Fast Link (100Gbps)
                              // Comm overhead: 0.5*8/100 * 1000 = 40ms

        {3, 150.0, 32, 0.0}   // VG3: Mid Compute, Last node (0 overhead)
    };

    std::cout << "Test Case: 3-Stage Pipeline Partitioning" << std::endl;
    std::cout << "Model: " << model.total_layers << " layers, " << model.activation_size_gb << " GB activations." << std::endl;
    
    std::vector<int> allocation = SolveLayerPartition(vgs, model);
    print_allocation(vgs, allocation, model);

    // 预期：
    // VG2 虽然算力弱 (300ms vs 100ms)，但通信开销极小 (40ms vs 400ms)。
    // 算法应该会给 VG2 分配较少层数，防止成为瓶颈。
    // VG1 虽然通信慢，但算力强，如果层数太少，通信占比高；层数多计算占比高。
    
    return 0;
}

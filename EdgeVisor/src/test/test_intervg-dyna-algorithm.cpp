#include <iostream>
#include <vector>
#include <iomanip>
#include "../nn/nn-intervg-dynamic-algorithm.hpp"

// 辅助打印函数
void print_state(const std::vector<int>& layers, const std::vector<double>& times) {
    if (times.empty()) {
        std::cout << "Layers: [ ";
        for (int l : layers) std::cout << l << " ";
        std::cout << "]" << std::endl;
        return;
    }
    
    std::cout << "State:" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "  VG " << i << ": Layers=" << layers[i] << ", Time=" << times[i] << " ms";
        if (layers[i] > 0) {
            std::cout << " (avg " << times[i]/layers[i] << " ms/layer)";
        }
        std::cout << std::endl;
    }
}

int main() {
    // 场景构造:
    // VG=[0, 1, 2]
    // Time=[100, 200, 110]
    // Layer=[10, 10, 10]
    
    // 物理含义：
    // VG1 是明显的瓶颈 (200ms)，比左右邻居都慢。
    // VG0 (100ms) 和 VG2 (110ms) 都比较空闲。
    // VG0 更快 (100 < 110)，所以算法应该优先选择 VG0 作为救援目标。
    // Unit time:
    // VG1: 200/10 = 20 ms/layer
    // VG0: 100/10 = 10 ms/layer
    
    // 模拟移动 1 layer:
    // VG1 new time = 200 - 20 = 180 ms
    // VG0 new time = 100 + 10 = 110 ms
    // New Bottleneck between them = 180 ms (依然是 VG1，但降低了)
    // Gain = 200 - 180 = 20 ms.
    
    // Threshold check:
    // Threshold ratio = 0.05 (5%)
    // Required gain = 200 * 0.05 = 10 ms.
    // 20 > 10, should trigger rebalance.

    std::vector<int> layers = {10, 10, 10};
    std::vector<double> times = {100.0, 200.0, 110.0};
    double threshold = 0.05;

    std::cout << "--- Initial State ---" << std::endl;
    print_state(layers, times);

    std::cout << "\nRunning IBSA..." << std::endl;
    RebalanceResult result = SolveIBSA(layers, times, threshold);

    if (result.is_changed) {
        std::cout << "Rebalance Triggered!" << std::endl;
        std::cout << "Moved 1 layer from VG " << result.source_vg_idx 
                  << " to VG " << result.target_vg_idx << std::endl;
        
        std::cout << "New Layer Allocation: [ ";
        for (int l : result.new_layers) std::cout << l << " ";
        std::cout << "]" << std::endl;
        
        // 验证计算
        double t_old = 200.0;
        double t_src_new = 200.0 - (200.0/10.0); // 180
        double t_dst_old = 100.0;
        double t_dst_new = 100.0 + (100.0/10.0); // 110
        double max_new = std::max(t_src_new, t_dst_new);
        double gain = t_old - max_new; // 20
        
        std::cout << "Predicted Gain: " << gain << " ms" << std::endl;
        std::cout << "Threshold: " << t_old * threshold << " ms" << std::endl;
    } else {
        std::cout << "No Change Triggered (Gain below threshold or no optimization possible)." << std::endl;
    }

    return 0;
}

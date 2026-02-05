#include "nn-intervg-dynamic-algorithm.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

RebalanceResult SolveIBSA(
    const std::vector<int>& current_layers,
    const std::vector<double>& execution_times,
    double threshold_ratio
) {
    // 1. 输入校验
    if (current_layers.empty() || execution_times.empty()) {
        return {current_layers, false, -1, -1};
    }
    if (current_layers.size() != execution_times.size()) {
        std::cerr << "Sizes of layers and times mismatch." << std::endl;
        return {current_layers, false, -1, -1};
    }

    size_t n = current_layers.size();
    
    // 2. 计算 Unit Cost (mu)
    std::vector<double> mu(n);
    for (size_t i = 0; i < n; ++i) {
        if (current_layers[i] > 0) {
            mu[i] = execution_times[i] / current_layers[i];
        } else {
            // 防御性编程：空闲节点处理边际成本很低 (这里简单设为 0，实际上应该是一个基于硬件的预估值)
            mu[i] = 0.0; 
        }
    }

    // 3. 定位瓶颈 (Bottleneck)
    int max_idx = -1;
    double t_max = -1.0;

    for (size_t i = 0; i < n; ++i) {
        if (execution_times[i] > t_max) {
            t_max = execution_times[i];
            max_idx = (int)i;
        }
    }
    
    // 如果全都是 0 或者没找到，或者是空闲的瓶颈（不可能发生，但防御）
    if (max_idx == -1 || current_layers[max_idx] == 0) {
        return {current_layers, false, -1, -1};
    }

    // 4. 寻找救援邻居 (Neighbor Search)
    int target_idx = -1;
    double min_neighbor_time = std::numeric_limits<double>::max();

    // 检查左邻居
    if (max_idx > 0) {
        int left_idx = max_idx - 1;
        if (execution_times[left_idx] < min_neighbor_time) {
            min_neighbor_time = execution_times[left_idx];
            target_idx = left_idx;
        }
    }

    // 检查右邻居
    if (max_idx < (int)n - 1) {
        int right_idx = max_idx + 1;
        // 优先选择更空闲的
        if (execution_times[right_idx] < min_neighbor_time) {
            min_neighbor_time = execution_times[right_idx];
            target_idx = right_idx;
        }
    }

    // 如果没有可用的邻居 (e.g., 单节点) 或者邻居并不比瓶颈快 (无法优化)
    if (target_idx == -1 || min_neighbor_time >= t_max) {
        return {current_layers, false, -1, -1};
    }

    // 5. 模拟移动与收益评估 (Simulation)
    // 尝试移动 1 Layer: max_idx -> target_idx
    double t_src_predicted = t_max - mu[max_idx];
    // 目标节点增加一层，成本增加对应的 mu (注意：成本取决于接收者的能力)
    double t_target_predicted = execution_times[target_idx] + mu[target_idx];
    
    // 如果 target 原来没有层数，mu_target 为 0。这其实不准。
    // 在真实系统中应该有该节点的各个 profile。
    // 这里为了严格遵守算法描述： "mu_i = 0 ... t'_dst = t_target + mu_target"
    // 如果 mu_target 为 0，意味着我们假设给空闲节点加任务不需要额外时间，这有点激进，但符合题目描述。
    // 在实际中，如果 current_layers[i] == 0, mu_i 应该 fallback 到一个预设值。
    // 但题目明确说： "如果 current_layers[i] == 0，则设 mu_i = 0"
    
    // 新的瓶颈时间 (局部视角：只看这两个节点的变化情况)
    double t_bottleneck_predicted = std::max(t_src_predicted, t_target_predicted);
    double gain = t_max - t_bottleneck_predicted;

    // 6. 惯性决策 (Hysteresis Check)
    // 只有收益显著才变动，避免抖动
    double required_gain = t_max * threshold_ratio;
    
    if (gain > required_gain) {
        std::vector<int> new_layers = current_layers;
        new_layers[max_idx]--;
        new_layers[target_idx]++;
        
        return {new_layers, true, max_idx, target_idx};
    }

    return {current_layers, false, -1, -1};
}

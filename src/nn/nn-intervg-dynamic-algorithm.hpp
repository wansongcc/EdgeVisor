#pragma once
#include <vector>

struct RebalanceResult {
    std::vector<int> new_layers; // 调整后的 Layer 分配方案
    bool is_changed;             // 是否发生了变动
    int source_vg_idx;           // 移出 Layer 的 VG 索引 (-1 表示无)
    int target_vg_idx;           // 移入 Layer 的 VG 索引 (-1 表示无)
};

/**
 * 惯性边界平滑算法 (Inertial Boundary Smoothing Algorithm, IBSA)
 * 
 * 基于运行时反馈，对 VG 间的 Layer 划分进行微调。
 * 
 * @param current_layers 当前每个 VG 持有的 Layer 数量
 * @param execution_times 上一轮观测到的每个 VG 的总执行时间 (ms)
 * @param threshold_ratio 惯性阈值比例 (e.g. 0.05 代表 5% improvement needed)
 * @return RebalanceResult 包含调整结果
 */
RebalanceResult SolveIBSA(
    const std::vector<int>& current_layers,
    const std::vector<double>& execution_times,
    double threshold_ratio
);

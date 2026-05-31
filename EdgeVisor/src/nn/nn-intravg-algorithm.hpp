#pragma once
#include <vector>

struct WorkerProfile {
    int id;
    double compute_flops;   // 算力
    double bandwidth_bps;   // 到 Root 的带宽
    double max_alpha_mem;   // 显存允许的最大分配比例 (0.0 ~ 1.0)
};

struct LayerTask {
    double total_flops;     // 本层总计算量
    double input_bytes;     // 输入广播数据量
    double output_bytes;    // 输出汇聚数据量
};

struct AllocationResult {
    std::vector<double> alphas; // 分配比例
    double estimated_latency;   // 该 VG 处理本层的预估耗时(s)
};

AllocationResult SolveCCWF(
    const std::vector<WorkerProfile>& workers,
    const LayerTask& task
);

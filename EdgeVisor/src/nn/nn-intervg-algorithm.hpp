#pragma once
#include <vector>

struct VGProfile {
    int id;
    double unit_time_ms;      // [输入] 处理单层 Layer 的耗时 ms
    int max_layers_capacity;  // [约束] 显存允许的最大 Layer 数量 (整数)
    double next_link_bw_gbps; // [输入] 到下一个 VG 的带宽 Gbps (若为最后一个 VG，则为到 Host 的带宽或设为极的大值/0)
};

struct ModelConfig {
    int total_layers;          // 模型总层数 (例如 32)
    double activation_size_gb; // 层间传输的数据量 GB (用于计算固定通信开销)
};

/**
 * 求解层划分问题 (Optimal Layer Partitioning)
 * 使用动态规划最小化流水线瓶颈时间。
 * 
 * @param vgs VG 列表，按流水线顺序排列
 * @param model 模型配置
 * @return std::vector<int> 包含每个 VG 分配的 Layer 数量，顺序与输入 vgs 一致。失败时返回空向量。
 */
std::vector<int> SolveLayerPartition(
    const std::vector<VGProfile>& vgs,
    const ModelConfig& model
);

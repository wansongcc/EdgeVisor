#include "nn-intervg-algorithm.hpp"
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

std::vector<int> SolveLayerPartition(
    const std::vector<VGProfile>& vgs,
    const ModelConfig& model
) {
    if (vgs.empty() || model.total_layers <= 0) {
        return {};
    }

    int num_vgs = (int)vgs.size();
    int total_layers = model.total_layers;
    double infinity = std::numeric_limits<double>::max();

    // dp[k][l]: 使用前 k 个 VG 处理前 l 个 Layer 时的最小瓶颈时间
    // 维度 n_vgs+1 x total_layers+1
    std::vector<std::vector<double>> dp(num_vgs + 1, std::vector<double>(total_layers + 1, infinity));
    
    // split_point[k][l]: 最优决策下，前 k-1 个 VG 处理的层数 j。
    // 即当前第 k 个 VG 处理了 l - j 层。
    std::vector<std::vector<int>> split_point(num_vgs + 1, std::vector<int>(total_layers + 1, -1));

    // 初始化：0个VG处理0层，耗时为0。
    dp[0][0] = 0.0;

    for (int k = 1; k <= num_vgs; ++k) {
        const auto& vg = vgs[k - 1]; // 当前 VG (注意索引偏移)
        
        // 我们需要计算使用 k 个 VG 覆盖 l 层的情况
        for (int l = 0; l <= total_layers; ++l) {
            
            // 尝试之前的切分点 j (前 k-1 个 VG 处理了 j 层)
            // 当前 VG 分配 count = l - j 层
            // j 的范围从 0 到 l
            for (int j = 0; j <= l; ++j) {
                // 如果前序状态不可达，不仅无法计算，也没有意义，跳过
                if (dp[k - 1][j] == infinity) continue;

                int count = l - j;

                // 显存约束检查
                if (count > vg.max_layers_capacity) continue;

                // 计算 Stage 耗时
                // T_stage = (n * T_unit) + T_comm
                
                double t_calc = count * vg.unit_time_ms;
                
                double t_comm = 0.0;
                if (count > 0) {
                    // 若带宽有效且 > 0
                    if (vg.next_link_bw_gbps > 1e-9) {
                        // Time (sec) = Size (Gb) / BW (Gbps)
                        // Size in GB, so * 8 to get Gb
                        // BW in Gbps
                        double transfer_seconds = (model.activation_size_gb * 8.0) / vg.next_link_bw_gbps;
                        t_comm = transfer_seconds * 1000.0; // 转换为 ms
                    } else {
                        // 带宽为 0，视为无法传输（除非无需传输）
                        // 根据题意，如果 count > 0 且 bw ~ 0，这可能是个无限大的代价
                        // 这里我们就假设如果是最后一个节点或者bw特别大/0的处理方式由输入保证合理性
                        // 一般来说如果是最后一个节点，next_link_bw_gbps 可能不需要或很大。
                        // 如果在中间且带宽为0，这路径不通。
                        // 为了鲁棒性，如果不是最后一个节点且带宽极小，设为极大代价
                        // 但题目说"若为最后一个 VG，则为到 Host 的带宽或设为极的大值/0"
                        // 我们只在 bw > 0 时加 t_comm, 若 bw=0 且 count > 0 我们暂且假设无开销(如最后节点)或不可达?
                        // 根据题目描述 "T_comm...是一项固定开销，只要 n > 0 就存在"，并没有说带宽为0怎么处理。
                        // 通常最后一个节点不需要传给谁，所以 T_comm = 0 比较合理。
                        t_comm = 0.0; 
                    }
                }

                double stage_cost = t_calc + t_comm;

                // 瓶颈是当前 stage 耗时 与 之前所有 stage 的最大耗时 中的 较大者
                double current_bottleneck = std::max(dp[k - 1][j], stage_cost);

                // Min-Max 决策
                if (current_bottleneck < dp[k][l]) {
                    dp[k][l] = current_bottleneck;
                    split_point[k][l] = j;
                }
            }
        }
    }

    // 检查是否有解
    if (dp[num_vgs][total_layers] == infinity) {
        return {}; // 无法找到满足约束的分配方案
    }

    // 回溯路径
    std::vector<int> allocation(num_vgs);
    int curr_l = total_layers;
    for (int k = num_vgs; k >= 1; --k) {
        int prev_l = split_point[k][curr_l];
        // 应该不会出现 prev_l = -1，如果有解的话
        int layers_for_k = curr_l - prev_l;
        allocation[k - 1] = layers_for_k;
        curr_l = prev_l;
    }

    return allocation;
}

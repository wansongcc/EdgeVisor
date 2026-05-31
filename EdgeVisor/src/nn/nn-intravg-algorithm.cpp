#include "nn-intravg-algorithm.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

AllocationResult SolveCCWF(const std::vector<WorkerProfile>& workers, const LayerTask& task) {
    if (workers.empty()) {
        return {{}, 0.0};
    }

    size_t n = workers.size();
    
    // 1. 预计算参数 H_i (起跑线) 和 K_i (单位代价)
    // T_i(alpha_i) = H_i + alpha_i * K_i
    std::vector<double> H(n);
    std::vector<double> K(n);

    double min_H = std::numeric_limits<double>::max();
    double max_H_plus_K = 0.0; // 假设 worst case 为某个节点承担 100% 任务 (如果显存允许)

    for (size_t i = 0; i < n; ++i) {
        // H_i = T_recv = input_bytes / bandwidth
        // 注意：如果 bandwidth 为 0，需要处理，这里假设 bandwidth > 0
        H[i] = task.input_bytes / workers[i].bandwidth_bps;

        // K_i = T_comp_unit + T_send_unit
        // T_comp_unit = total_flops / compute_flops
        // T_send_unit = output_bytes / bandwidth
        double t_comp_full = task.total_flops / workers[i].compute_flops;
        double t_send_full = task.output_bytes / workers[i].bandwidth_bps;
        K[i] = t_comp_full + t_send_full;

        if (H[i] < min_H) min_H = H[i];
    
        // 粗略估算上界：所有任务给最慢的人（如果不考虑显存限制）
        // 实际上只需要一个足够大的上界即可
        double t_full = H[i] + 1.0 * K[i];
        if (t_full > max_H_plus_K) {
            max_H_plus_K = t_full;
        }
    }

    // 2. 二分查找最优时间 T*
    // 搜索范围：
    // Low: 至少需要最小的接收时间 (不可能比任何人的起跑线还快，如果不分配任务给他的话其实是 0，
    //      但只要分配任务，时间就是 H_i + ...)
    double low = min_H; 
    double high = max_H_plus_K * 2.0; // 安全上界
    
    // 二分迭代次数，或者用精度控制
    const int max_iter = 100;
    
    double best_T = high;

    for (int iter = 0; iter < max_iter; ++iter) {
        double mid = low + (high - low) / 2.0;
        
        double sum_alpha = 0.0;
        for (size_t i = 0; i < n; ++i) {
            // T = H_i + alpha * K_i  => alpha = (T - H_i) / K_i
            double alpha = 0.0;
            if (mid > H[i]) {
                alpha = (mid - H[i]) / K[i];
            }
            
            // 应用显存上限
            if (alpha > workers[i].max_alpha_mem) {
                alpha = workers[i].max_alpha_mem;
            }
            // 比例不能为负 (上面 check mid > H[i] 保证了)
            
            sum_alpha += alpha;
        }

        if (sum_alpha >= 1.0) {
            // 时间充裕，尝试减小时间
            best_T = mid;
            high = mid;
        } else {
            // 时间不够，无法分配完 100% 任务
            low = mid;
        }
    }

    // 3. 计算最终 Alpha
    std::vector<double> alphas(n);
    double final_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double alpha = 0.0;
        if (best_T > H[i]) {
            alpha = (best_T - H[i]) / K[i];
        }
        if (alpha > workers[i].max_alpha_mem) {
            alpha = workers[i].max_alpha_mem;
        }
        // max(0.0)
        if (alpha < 0.0) alpha = 0.0;
        
        alphas[i] = alpha;
        final_sum += alpha;
    }

    // 4. 归一化 (消除浮点误差，或者如果 sum_alpha > 1.0 的微小修正)
    if (final_sum > 0.0) {
        for (size_t i = 0; i < n; ++i) {
            alphas[i] /= final_sum;
        }
    }

    return {alphas, best_T};
}

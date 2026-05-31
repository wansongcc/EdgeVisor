#include "nn-kv_reused-algorithm.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>

static int findBottleneckDeviceIndex(const std::vector<DeviceStatus>& devices) {
    int src_idx = -1;
    double max_time = -1.0;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].execution_time_ms > max_time) {
            max_time = devices[i].execution_time_ms;
            src_idx = (int)i;
        }
    }
    return src_idx;
}

std::vector<RebalanceMove> RebalanceHeadMoves(const std::vector<DeviceStatus>& devices) {
    std::vector<RebalanceMove> moves;
    if (devices.empty()) return moves;

    const int src_idx = findBottleneckDeviceIndex(devices);
    if (src_idx < 0) return moves;

    const auto& src_dev = devices[(size_t)src_idx];
    const int current_heads = src_dev.current_compute.count();
    if (current_heads <= 1) return moves; // Must keep at least 1 head.

    const double t_src_unit = src_dev.execution_time_ms / (double)current_heads;
    const int max_movable = current_heads - 1;

    int move_left = 0;
    int move_right = 0;

    // --- Calculate Move to Left ---
    if (src_idx > 0) {
        const auto& left_dev = devices[(size_t)(src_idx - 1)];
        const int l_heads = left_dev.current_compute.count();
        const double t_left_unit = (l_heads > 0) ? (left_dev.execution_time_ms / (double)l_heads) : t_src_unit;
        if (src_dev.execution_time_ms > left_dev.execution_time_ms) {
            const double delta_raw = (src_dev.execution_time_ms - left_dev.execution_time_ms) / (t_src_unit + t_left_unit);
            move_left = (int)std::floor(delta_raw);
        }
    }

    // --- Calculate Move to Right ---
    if (src_idx < (int)devices.size() - 1) {
        const auto& right_dev = devices[(size_t)(src_idx + 1)];
        const int r_heads = right_dev.current_compute.count();
        const double t_right_unit = (r_heads > 0) ? (right_dev.execution_time_ms / (double)r_heads) : t_src_unit;
        if (src_dev.execution_time_ms > right_dev.execution_time_ms) {
            const double delta_raw = (src_dev.execution_time_ms - right_dev.execution_time_ms) / (t_src_unit + t_right_unit);
            move_right = (int)std::floor(delta_raw);
        }
    }

    if (move_left < 0) move_left = 0;
    if (move_right < 0) move_right = 0;

    // Apply limit based on movable heads
    if (move_left + move_right > max_movable) {
        const float total_req = (float)(move_left + move_right);
        const float ratio = (total_req > 0.0f) ? ((float)max_movable / total_req) : 0.0f;
        int new_move_left = (int)(move_left * ratio);
        int new_move_right = (int)(move_right * ratio);
        if (new_move_left + new_move_right > max_movable) {
            if (new_move_left > 0) new_move_left--;
            else if (new_move_right > 0) new_move_right--;
        }
        move_left = new_move_left;
        move_right = new_move_right;
    }

    // 3. Apply KV-Cache constraints (do not mutate ranges here; just clamp moves)
    if (move_left > 0 && src_idx > 0) {
        const auto& left_kv = devices[(size_t)(src_idx - 1)].kv_cache_holding;
        const int current_L_end = devices[(size_t)(src_idx - 1)].current_compute.end;
        int max_possible = left_kv.end - current_L_end;
        if (max_possible < 0) max_possible = 0;
        move_left = std::min(move_left, max_possible);
    }

    if (move_right > 0 && src_idx < (int)devices.size() - 1) {
        const auto& right_kv = devices[(size_t)(src_idx + 1)].kv_cache_holding;
        const int current_R_start = devices[(size_t)(src_idx + 1)].current_compute.start;
        int max_possible = current_R_start - right_kv.start;
        if (max_possible < 0) max_possible = 0;
        move_right = std::min(move_right, max_possible);
    }

    // Emit moves (head-only for now)
    if (move_left > 0 && src_idx > 0) {
        moves.push_back(RebalanceMove{
            /*from_device_id=*/src_dev.device_id,
            /*to_device_id=*/devices[(size_t)(src_idx - 1)].device_id,
            /*cmdKind=*/1,
            /*headMove=*/move_left,
            /*ffnMove=*/0,
        });
    }

    if (move_right > 0 && src_idx < (int)devices.size() - 1) {
        moves.push_back(RebalanceMove{
            /*from_device_id=*/src_dev.device_id,
            /*to_device_id=*/devices[(size_t)(src_idx + 1)].device_id,
            /*cmdKind=*/1,
            /*headMove=*/move_right,
            /*ffnMove=*/0,
        });
    }

    return moves;
}

std::vector<HeadRange> RebalanceHeads(const std::vector<DeviceStatus>& devices) {
    std::vector<HeadRange> new_ranges;
    for (const auto& dev : devices) {
        new_ranges.push_back(dev.current_compute);
    }

    if (devices.empty()) return new_ranges;

    // 1. 寻找瓶颈设备 (T_max)
    int src_idx = findBottleneckDeviceIndex(devices);

    if (src_idx == -1) return new_ranges; // Should not happen in normal cases

    const auto& src_dev = devices[src_idx];
    int current_heads = src_dev.current_compute.count();
    
    // Prevent divide by zero if heads is 0, though unlikely in practice for active device
    if (current_heads == 0) return new_ranges;

    double t_src_unit = src_dev.execution_time_ms / current_heads;

    // 2. 计算理论迁移量 (Load Balancing)
    // Heads available to move. Must keep at least 1.
    int max_movable = current_heads - 1;
    if (max_movable <= 0) return new_ranges;

    int move_left = 0;
    int move_right = 0;

    // --- Calculate Move to Left ---
    if (src_idx > 0) {
        const auto& left_dev = devices[src_idx - 1];
        int l_heads = left_dev.current_compute.count();
        double t_left_unit = (l_heads > 0) ? (left_dev.execution_time_ms / l_heads) : t_src_unit; // Fallback estimate
        
        // Only move if src is slower
        if (src_dev.execution_time_ms > left_dev.execution_time_ms) {
             // Formula: delta = (T_src - T_L) / (t_src + t_L)
             double delta_raw = (src_dev.execution_time_ms - left_dev.execution_time_ms) / 
                                (t_src_unit + t_left_unit);
             move_left = static_cast<int>(std::floor(delta_raw));
        }
    }

    // --- Calculate Move to Right ---
    if (src_idx < (int)devices.size() - 1) {
        const auto& right_dev = devices[src_idx + 1];
        int r_heads = right_dev.current_compute.count();
        double t_right_unit = (r_heads > 0) ? (right_dev.execution_time_ms / r_heads) : t_src_unit;
        
        if (src_dev.execution_time_ms > right_dev.execution_time_ms) {
            double delta_raw = (src_dev.execution_time_ms - right_dev.execution_time_ms) / 
                               (t_src_unit + t_right_unit);
            move_right = static_cast<int>(std::floor(delta_raw));
        }
    }
    
    // Apply limit based on movable heads
    if (move_left + move_right > max_movable) {
         // Scale down proportionally
         float total_req = (float)(move_left + move_right);
         float ratio = (float)max_movable / total_req;
         
         // Prioritize integer arithmetic carefully or just use float scaling
         int new_move_left = static_cast<int>(move_left * ratio);
         int new_move_right = static_cast<int>(move_right * ratio);
         
         // Ensure we don't exceed max_movable due to rounding, distribute remainder if any?
         // Simplest is to just clamp. 
         if (new_move_left + new_move_right > max_movable) {
             if (new_move_left > 0) new_move_left--;
         }
         
         move_left = new_move_left;
         move_right = new_move_right;
    }

    // 3. 应用 KV-Cache 约束并执行移动

    // --- Attempt Move Left ---
    // Src gives [Src.start, Src.start + k - 1] to Left.
    // Neighbors are contiguous: Left.end = Src.start - 1.
    // Left new range becomes [Left.start, Left.end + k].
    // Constraint: New Left range must be within Left.kv_cache_holding.
    // Specifically, [Left.end + 1, Left.end + k] must be in Left.kv.
    if (move_left > 0 && src_idx > 0) {
        const auto& left_kv = devices[src_idx - 1].kv_cache_holding;
        int current_L_end = new_ranges[src_idx - 1].end;
        
        // Calculate max allowed overlap into Src's territory based on Left's KV
        int max_possible = left_kv.end - current_L_end;
        if (max_possible < 0) max_possible = 0;
        
        move_left = std::min(move_left, max_possible);
        
        // Apply Move Left
        if (move_left > 0) {
            new_ranges[src_idx - 1].end += move_left;
            new_ranges[src_idx].start += move_left;
        }
    }

    // --- Attempt Move Right ---
    // Src gives [Src.end - k + 1, Src.end] to Right.
    // Neighbors are contiguous: Right.start = Src.end + 1.
    // Right new range becomes [Right.start - k, Right.end].
    // Constraint: New Right range must be within Right.kv_cache_holding.
    // Specifically, [Right.start - k, Right.start - 1] must be in Right.kv.
    if (move_right > 0 && src_idx < (int)devices.size() - 1) {
        const auto& right_kv = devices[src_idx + 1].kv_cache_holding;
        int current_R_start = new_ranges[src_idx + 1].start;
        
        // Calculate max allowed overlap into Src's territory based on Right's KV
        int max_possible = current_R_start - right_kv.start;
        if (max_possible < 0) max_possible = 0;
        
        move_right = std::min(move_right, max_possible);

        // Apply Move Right
        if (move_right > 0) {
            new_ranges[src_idx].end -= move_right;
            new_ranges[src_idx + 1].start -= move_right;
        }
    }

    return new_ranges;
}

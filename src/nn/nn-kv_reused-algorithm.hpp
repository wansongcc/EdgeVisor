#pragma once
#include <vector>
#include <cmath>
#include <iostream>

struct HeadRange {
    int start; // 起始 Head ID (包含)，例如 2
    int end;   // 结束 Head ID (包含)，例如 5。表示 [2, 3, 4, 5] 共 4 个 Heads
    
    int count() const { return end - start + 1; }
    
    // 判断 range 是否完全包含 target_range
    bool contains(const HeadRange& target) const {
        return start <= target.start && end >= target.end;
    }
};

struct DeviceStatus {
    int device_id;
    double execution_time_ms;      // 本层实际执行时间
    HeadRange current_compute;     // 本层实际计算的 Heads 范围
    HeadRange kv_cache_holding;    // 本设备持有的 KV-Cache 范围 (这是硬约束)
};

// A single migration action: move slices from one device to another.
// - from_device_id/to_device_id: should correspond to nodeIndex in the runtime plan.
// - cmdKind: 1=headSplit 2=ffnSplit 3=both (compatible with PlanCommandKind)
// - headMove: number of heads to move
// - ffnMove: number of FFN hidden units to move (0 if not used)
struct RebalanceMove {
    int from_device_id;
    int to_device_id;
    int cmdKind;
    int headMove;
    int ffnMove;
};

/**
 * @param current_layer_status: 当前层所有设备的状态列表
 * @return std::vector<HeadRange>: 下一层每个设备应该计算的 Heads 范围
 */
std::vector<HeadRange> RebalanceHeads(const std::vector<DeviceStatus>& devices);

// Compute rebalancing actions in the form of migration moves.
// Current implementation generates head moves only (ffnMove=0, cmdKind=1).
std::vector<RebalanceMove> RebalanceHeadMoves(const std::vector<DeviceStatus>& devices);

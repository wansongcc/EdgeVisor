#pragma once
#include <vector>
#include <map>
#include "nn-task-algorithm.hpp"
#include "nn-intravg-algorithm.hpp"
#include "nn-intervg-algorithm.hpp"

// 聚合最终的初始化结果
struct InitResult {
    // 1. 拓扑与分组 (Result from RRAGC)
    Result rragc_result; 
    
    // 2. 组内划分详情 (Result from CCWF for each VG)
    // Map: VG_ID -> AllocationResult (alphas, latency)
    std::map<int, AllocationResult> intravg_allocations;

    // 3. 组间层数划分 (Result from OLP)
    // Vector index corresponds to pipeline order, value is num layers
    std::vector<int> intervg_layer_allocation;
};

// 全局初始化算法接口
// 串联 RRAGC -> CCWF -> OLP
InitResult RunInitialization(
    const std::vector<Device>& devices,
    const std::vector<Link>& links,
    const Config& rragc_config,
    const LayerTask& layer_task_template, // 用于组内划分的单层任务模板
    const ModelConfig& model_config       // 用于组间划分的模型配置
);

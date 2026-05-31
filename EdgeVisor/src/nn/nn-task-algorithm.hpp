#ifndef NN_TASK_ALGORITHM_HPP
#define NN_TASK_ALGORITHM_HPP

#include <vector>
#include <map>
#include <iostream>

struct Device {
    int id;
    double compute; // FLOPS
    double memory;  // Bytes
};

struct Link {
    int src_id;
    int dst_id;
    double bandwidth; // Gbps
};

struct Config {
    int K;                  // 目标分组数
    double P_min;           // 最小算力阈值
    double M_min;           // 最小显存阈值
    double alpha;           // 带宽权重 (0.6~0.7)
    double beta;            // 算力权重 (0.3~0.4)
};

struct Result {
    std::map<int, int> device_to_vg_map; // Key: DeviceID, Value: VG_ID (0 to K-1)
    std::vector<int> vg_roots;           // 存储每个 VG 的 Root Device ID，索引对应 VG_ID
    std::vector<int> pipeline_order;     // VG_ID 的排列顺序，例如 {0, 2, 1} 表示 VG0->VG2->VG1
};

class RRAGC {
public:
    /**
     * @brief Executes the Root-Anchored Reliability-Aware Graph Contraction (R-RAGC) algorithm.
     * 
     * @param devices List of physical devices.
     * @param links List of physical links between devices.
     * @param config Algorithm configuration parameters.
     * @return Result structure containing the grouping and routing information.
     */
    static Result solve(const std::vector<Device>& devices, const std::vector<Link>& links, const Config& config);

    // Simple example function to demonstrate usage
    static void run_example();
};

#endif // NN_TASK_ALGORITHM_HPP

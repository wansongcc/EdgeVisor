#include <iostream>
#include <vector>
#include <iomanip>
#include "../nn/nn-init-algorithm.hpp"

// 使用命名空间简化代码
using std::cout;
using std::endl;
using std::vector;

int main() {
    cout << "=== Distributed Llama Initialization Test ===" << endl;

    // 1. 模拟物理设备 (Devices)
    // 假设有 4 台设备：
    // Dev 0: High Compute, High Mem (Root 1 Candidate)
    // Dev 1: Low Compute (Worker for 0)
    // Dev 2: High Compute (Root 2 Candidate)
    // Dev 3: Low Compute (Worker for 2)
    vector<Device> devices = {
        {0, 100e12, 16.0}, // 100 TFLOPS, 16GB
        {1, 20e12,  8.0},  // 20 TFLOPS, 8GB
        {2, 80e12,  16.0}, // 80 TFLOPS, 16GB
        {3, 20e12,  8.0}   // 20 TFLOPS, 8GB
    };

    // 2. 模拟物理链路 (Links)
    // 拓扑：
    // [1] <-> [0] <=====> [2] <-> [3]
    // 0-1: 局域网 10Gbps
    // 2-3: 局域网 10Gbps
    // 0-2: 广域网/跨机柜 5Gbps
    vector<Link> links = {
        {0, 1, 10.0}, {1, 0, 10.0}, // Intra-VG 1
        {2, 3, 10.0}, {3, 2, 10.0}, // Intra-VG 2
        {0, 2, 5.0},  {2, 0, 5.0}   // Inter-VG
    };

    // 3. 配置参数
    Config rragc_config;
    rragc_config.K = 2; // 期望切分成 2 个 VG
    rragc_config.P_min = 50.0; // 这个可以求出来
    rragc_config.M_min = 10.0; // 这个也可以求出来
    rragc_config.alpha = 0.7; // 这两个直接给就可以
    rragc_config.beta = 0.3;

    LayerTask layer_task;
    layer_task.total_flops = 10e12;     // 这个可以根据模型给出来
    layer_task.input_bytes = 100e6;     // 这个也可以根据模型给出来
    layer_task.output_bytes = 100e6;    // 这个也可以根据模型给出来

    ModelConfig model_config;
    model_config.total_layers = 16;     // 这个也可以根据模型给出来
    model_config.activation_size_gb = 0.1; // 这个也可以根据模型给出来

    // 4. 执行初始化全流程
    cout << "\n>>> Starting Initialization Pipeline..." << endl;
    InitResult result = RunInitialization(
        devices, 
        links, 
        rragc_config, 
        layer_task, 
        model_config
    );

    // 5. 结果验证与打印
    cout << "\n>>> Initialization Complete. Report:" << endl;

    // A. RRAGC 结果
    cout << "\n[Topology Partitioning (RRAGC)]" << endl;
    const auto& topo = result.rragc_result;
    cout << "  Pipeline Order: ";
    for (int vg_id : topo.pipeline_order) {
        cout << "VG" << vg_id << " ";
        if (vg_id != topo.pipeline_order.back()) cout << "-> ";
    }
    cout << endl;

    for (int vg_id : topo.pipeline_order) {
        cout << "  VG " << vg_id << " Root: Device " << topo.vg_roots[vg_id] << endl;
    }

    // B. CCWF 结果 (组内划分)
    cout << "\n[Intra-VG Allocation (CCWF)]" << endl;
    for (int vg_id : topo.pipeline_order) {
        if (result.intravg_allocations.count(vg_id)) {
            const auto& alloc = result.intravg_allocations[vg_id];
            cout << "  VG " << vg_id << " Estimated Latency per Layer: " << alloc.estimated_latency * 1000.0 << " ms" << endl;
            cout << "    Split Ratios (Alphas):" << endl;
            
            // 找到 VG 成员打印 (需要一点反向查找的逻辑，这里简单打印 raw vector)
            for (size_t i = 0; i < alloc.alphas.size(); ++i) {
                cout << "      Worker " << i << ": " << std::fixed << std::setprecision(2) << alloc.alphas[i] * 100.0 << "%";
                if (i == 0) cout << " (Root)";
                cout << endl;
            }
        }
    }

    // C. OLP 结果 (组间划分)
    cout << "\n[Inter-VG Partitioning (OLP)]" << endl;
    const auto& layer_alloc = result.intervg_layer_allocation;
    if (layer_alloc.empty()) {
        cout << "  FAILED to partition layers!" << endl;
    } else {
        int layer_start = 0;
        for (size_t i = 0; i < layer_alloc.size(); ++i) {
            int vg_id = topo.pipeline_order[i];
            int count = layer_alloc[i];
            cout << "  VG " << vg_id << ": " << count << " Layers (Layers " 
                 << layer_start << " - " << (layer_start + count - 1) << ")" << endl;
            layer_start += count;
        }
    }

    return 0;
}

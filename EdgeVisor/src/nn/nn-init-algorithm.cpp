#include "nn-init-algorithm.hpp"
#include <iostream>
#include <map>
#include <algorithm>

// Helper to look up bandwidth
static double get_bandwidth(const std::vector<Link>& links, int src, int dst) {
    for (const auto& l : links) {
        if (l.src_id == src && l.dst_id == dst) return l.bandwidth;
    }
    return 0.0;
}

InitResult RunInitialization(
    const std::vector<Device>& devices,
    const std::vector<Link>& links,
    const Config& rragc_config,
    const LayerTask& layer_task_template,
    const ModelConfig& model_config
) {
    InitResult final_result;

    // Step 1: RRAGC - Network Topology Partitioning
    // Output: VG groups, Roots, Pipeline Order
    std::cout << "[Init] Step 1: Running RRAGC..." << std::endl;
    final_result.rragc_result = RRAGC::solve(devices, links, rragc_config);

    // Map Device ID to Device object for quick lookup
    std::map<int, Device> device_map;
    for (const auto& d : devices) device_map[d.id] = d;

    // Organize devices by VG
    std::map<int, std::vector<int>> vg_members;
    for (const auto& [dev_id, vg_id] : final_result.rragc_result.device_to_vg_map) {
        vg_members[vg_id].push_back(dev_id);
    }

    // Step 2: CCWF - Intra-VG Application (Profiling & Parallelism)
    // Run CCWF for each VG to get unit_time and alpha allocations
    std::cout << "[Init] Step 2: Running CCWF for each VG..." << std::endl;
    
    // We need to prepare VGProfiles for Step 3
    std::vector<VGProfile> vg_profiles_for_olp;

    // Iterate through VGs in Pipeline Order (important for Step 3 readiness)
    for (int vg_id : final_result.rragc_result.pipeline_order) {
        int root_id = final_result.rragc_result.vg_roots[vg_id];
        const auto& member_ids = vg_members[vg_id];

        // Construct WorkerProfile list for this VG
        std::vector<WorkerProfile> workers;
        int total_mem_capacity_layers = 0; // Simple estimation logic

        for (int dev_id : member_ids) {
            const auto& dev = device_map[dev_id];
            
            // Assume Max Memory Ratio calculation:
            // This is simplified. In reality, you'd check free VRAM vs Layer Size.
            // Let's assume each device can hold up to 100% of a layer for calculation if memory allows.
            // For OLP constraint purposes (Total Layers per VG), we sum up or take min?
            // Usually, pipeline parallel limitation is total VRAM of the VG.
            // Let's assume a dummy logic: 1GB memory = 1 layer capacity (just for demo)
            total_mem_capacity_layers += (int)(dev.memory / 0.5); // Assume 0.5GB per layer

            // Bandwidth to Root
            // If dev is Root, BW is infinite (or very high internal memory BW)
            // If dev is Worker, BW is Link(Root -> Worker)
            double bw_to_root = 1000.0; // Default high for itself
            if (dev_id != root_id) {
                bw_to_root = get_bandwidth(links, root_id, dev_id);
                // If link missing, assume 0 (will cause H_i to be inf)
                if (bw_to_root <= 1e-9) bw_to_root = 1e-9; 
            }

            workers.push_back({
                dev_id,
                dev.compute,
                bw_to_root * 1e9, // Gbps to bps
                1.0 // max_alpha_mem (simplified to 1.0 for now)
            });
        }

        // Run CCWF
        AllocationResult ccwf_res = SolveCCWF(workers, layer_task_template);
        final_result.intravg_allocations[vg_id] = ccwf_res;

        // Prepare info for OLP
        
        // Find next VG in pipeline to determine bandwidth
        double next_link_bw = 0.0;
        
        // Find index of current vg_id in pipeline_order
        auto it = std::find(final_result.rragc_result.pipeline_order.begin(), 
                            final_result.rragc_result.pipeline_order.end(), 
                            vg_id);
        
        if (it != final_result.rragc_result.pipeline_order.end() && 
            std::next(it) != final_result.rragc_result.pipeline_order.end()) {
            
            int next_vg_id = *std::next(it);
            int next_root_id = final_result.rragc_result.vg_roots[next_vg_id];
            
            // Bandwidth from Current Root to Next Root
            next_link_bw = get_bandwidth(links, root_id, next_root_id);
        }

        vg_profiles_for_olp.push_back({
            vg_id,
            ccwf_res.estimated_latency * 1000.0, // seconds -> ms
            total_mem_capacity_layers,
            next_link_bw // Gbps
        });
    }

    // Step 3: OLP - Inter-VG Layer Partitioning
    std::cout << "[Init] Step 3: Running OLP (DP Algorithm)..." << std::endl;
    final_result.intervg_layer_allocation = SolveLayerPartition(vg_profiles_for_olp, model_config);

    if (final_result.intervg_layer_allocation.empty()) {
        std::cerr << "[Init] OLP Failed to find a valid partition!" << std::endl;
    }

    return final_result;
}

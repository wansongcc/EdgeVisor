#include "../nn/nn-kv_reused-algorithm.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    // Construct Test Case
    // Dev 0: Time 100ms, Compute [0, 9], KV [0, 11] (向右 overlap 2个)
    DeviceStatus d0;
    d0.device_id = 0;
    d0.execution_time_ms = 100.0;
    d0.current_compute = {0, 9};
    d0.kv_cache_holding = {0, 11};

    // Dev 1: Time 200ms (瓶颈), Compute [10, 19], KV [8, 21] (向左 overlap 2个, 向右 overlap 2个)
    DeviceStatus d1;
    d1.device_id = 1;
    d1.execution_time_ms = 200.0;
    d1.current_compute = {10, 19};
    d1.kv_cache_holding = {8, 21};

    // Dev 2: Time 100ms, Compute [20, 29], KV [18, 29] (向左 overlap 2个)
    DeviceStatus d2;
    d2.device_id = 2;
    d2.execution_time_ms = 100.0;
    d2.current_compute = {20, 29};
    d2.kv_cache_holding = {18, 29};

    std::vector<DeviceStatus> devices = {d0, d1, d2};

    std::cout << "Initial State:" << std::endl;
    for (const auto& d : devices) {
        std::cout << "Dev " << d.device_id 
                  << ": Time=" << d.execution_time_ms << "ms"
                  << ", Compute=[" << d.current_compute.start << ", " << d.current_compute.end << "]"
                  << " (Count: " << d.current_compute.count() << ")"
                  << ", KV=[" << d.kv_cache_holding.start << ", " << d.kv_cache_holding.end << "]"
                  << std::endl;
    }

    std::cout << "\nRunning RebalanceHeads (IBSA)..." << std::endl;
    std::vector<HeadRange> new_ranges = RebalanceHeads(devices);

    std::cout << "\nRebalanced Ranges (Next Layer):" << std::endl;
    for (size_t i = 0; i < new_ranges.size(); ++i) {
        std::cout << "Dev " << devices[i].device_id 
                  << ": [" << new_ranges[i].start << ", " << new_ranges[i].end << "]"
                  << " Count=" << new_ranges[i].count()
                  << std::endl;
    }
    
    return 0;
}

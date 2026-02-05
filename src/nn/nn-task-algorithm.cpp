#include "nn-task-algorithm.hpp"
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <set>
#include <iostream>
#include <iomanip>

// Helper to access bandwidth in adjacency map
static double get_link_bw(const std::map<int, std::map<int, double>>& adj, int u, int v) {
    auto it = adj.find(u);
    if (it != adj.end()) {
        auto it2 = it->second.find(v);
        if (it2 != it->second.end()) {
            return it2->second;
        }
    }
    return 0.0;
}

// Helper to get bi-directional connection strength (max of u->v and v->u)
static double get_connection_strength(const std::map<int, std::map<int, double>>& adj, int u, int v) {
    return std::max(get_link_bw(adj, u, v), get_link_bw(adj, v, u));
}

Result RRAGC::solve(const std::vector<Device>& devices, const std::vector<Link>& links, const Config& config) {
    Result result;
    if (devices.empty()) return result;

    int N = devices.size();
    int K = std::min(config.K, N); // Ensure K <= N

    // Build Adjacency Matrix
    std::map<int, std::map<int, double>> adj;
    double global_max_link_bw = 0.0;
    for (const auto& l : links) {
        adj[l.src_id][l.dst_id] = l.bandwidth; // Assume directed or user provides both
        global_max_link_bw = std::max(global_max_link_bw, l.bandwidth);
    }
    
    // Map device ID to Device object for quick lookup
    std::map<int, Device> device_map;
    for (const auto& d : devices) {
        device_map[d.id] = d;
    }

    // =========================================================
    // Phase 1: Connectivity-Penalized Anchor Identification
    // =========================================================

    // 1. Data Aggregation: Calculate total_bandwidth
    std::map<int, double> total_bw;
    double max_bw = -1.0, min_bw = std::numeric_limits<double>::max();
    double max_c = -1.0, min_c = std::numeric_limits<double>::max();

    for (const auto& d : devices) {
        double bw_sum = 0.0;
        // Check outgoing
        if (adj.count(d.id)) {
            for (auto const& [neighbor, bw] : adj.at(d.id)) {
                bw_sum += bw;
            }
        }
        // Check incoming (to capture full connectivity degree)
        for (const auto& l : links) {
            if (l.dst_id == d.id) {
                bw_sum += l.bandwidth;
            }
        }
        
        total_bw[d.id] = bw_sum;

        if (bw_sum > max_bw) max_bw = bw_sum;
        if (bw_sum < min_bw) min_bw = bw_sum;

        if (d.compute > max_c) max_c = d.compute;
        if (d.compute < min_c) min_c = d.compute;
    }

    // Handle single device or uniform values case
    if (min_bw > max_bw) { min_bw = max_bw = 0; } // Should not happen if N > 0
    if (min_c > max_c) { min_c = max_c = 0; }

    // 2 & 3. Normalize and Calculate Raw Score
    std::map<int, double> score_raw;
    for (const auto& d : devices) {
        double b_hat = (std::abs(max_bw - min_bw) < 1e-9) ? 1.0 : (total_bw[d.id] - min_bw) / (max_bw - min_bw);
        double c_hat = (std::abs(max_c - min_c) < 1e-9) ? 1.0 : (d.compute - min_c) / (max_c - min_c);
        score_raw[d.id] = config.alpha * b_hat + config.beta * c_hat;
    }

    // 4. Iteratively Select Roots
    std::vector<int> selected_roots;
    std::set<int> selected_roots_set;

    for (int k = 0; k < K; ++k) {
        int best_candidate = -1;
        double best_adj_score = -1.0;

        for (const auto& d : devices) {
            if (selected_roots_set.count(d.id)) continue;

            double bw_link_u = 0.0;
            if (!selected_roots.empty()) {
                for (int root_id : selected_roots) {
                    double bw = get_connection_strength(adj, d.id, root_id);
                    if (bw > bw_link_u) bw_link_u = bw;
                }
            }

            double rho_u = (global_max_link_bw > 1e-9) ? (bw_link_u / global_max_link_bw) : 0.0;
            double score_adj = score_raw[d.id] * (1.0 - rho_u);

            if (score_adj > best_adj_score) {
                best_adj_score = score_adj;
                best_candidate = d.id;
            }
        }

        if (best_candidate != -1) {
            selected_roots.push_back(best_candidate);
            selected_roots_set.insert(best_candidate);
        } else {
            // Fallback if we run out of candidates (unexpected)
            break; 
        }
    }
    result.vg_roots = selected_roots;
    
    // Safety check just in case K was not reached
    K = selected_roots.size(); 

    // =========================================================
    // Phase 2: Star Topology Clustering
    // =========================================================
    
    // Temporary structure to hold VG members
    std::map<int, std::vector<int>> vg_members; // VG Index -> List of Device IDs
    std::map<int, int> dev_to_vg;

    // Initialize VGs with roots
    for (int i = 0; i < K; ++i) {
        int root_id = selected_roots[i];
        vg_members[i].push_back(root_id);
        dev_to_vg[root_id] = i;
    }

    // Assign workers
    std::vector<int> isolated_nodes;
    for (const auto& d : devices) {
        if (selected_roots_set.count(d.id)) continue;

        int best_vg = -1;
        double max_bw_to_root = -1.0;

        for (int i = 0; i < K; ++i) {
            int root_id = selected_roots[i];
            double bw = get_connection_strength(adj, d.id, root_id);
            if (bw > max_bw_to_root) {
                max_bw_to_root = bw;
                best_vg = i;
            }
        }

        if (max_bw_to_root > 0.0) {
            vg_members[best_vg].push_back(d.id);
            dev_to_vg[d.id] = best_vg;
        } else {
            isolated_nodes.push_back(d.id);
        }
    }

    // Handle isolated nodes - assign to VG with MINIMUM total compute
    for (int iso_id : isolated_nodes) {
        int target_vg = -1;
        double min_vg_compute = std::numeric_limits<double>::max();

        for (int i = 0; i < K; ++i) {
            double current_vg_compute = 0;
            for (int member_id : vg_members[i]) {
                current_vg_compute += device_map[member_id].compute;
            }
            if (current_vg_compute < min_vg_compute) {
                min_vg_compute = current_vg_compute;
                target_vg = i;
            }
        }
        
        // If K=0 (no roots), impossible, but safe check
        if (target_vg != -1) {
            vg_members[target_vg].push_back(iso_id);
            dev_to_vg[iso_id] = target_vg;
        }
    }

    // =========================================================
    // Phase 3: Minimum-Sacrifice Viability Enforcement
    // =========================================================

    bool optimization_active = true;
    int max_iterations = N * 2; // Prevent infinite loops
    int iter_count = 0;

    while (optimization_active && iter_count < max_iterations) {
        optimization_active = false;
        iter_count++;

        // Calculate stats for all VGs
        struct VGStats {
            double total_compute;
            double total_memory;
            bool is_critical;
        };
        std::vector<VGStats> vg_stats(K);
        std::vector<int> critical_vgs;

        for (int i = 0; i < K; ++i) {
            double c_sum = 0, m_sum = 0;
            for (int member_id : vg_members[i]) {
                c_sum += device_map[member_id].compute;
                m_sum += device_map[member_id].memory;
            }
            vg_stats[i] = {c_sum, m_sum, (c_sum < config.P_min || m_sum < config.M_min)};
            if (vg_stats[i].is_critical) {
                critical_vgs.push_back(i);
            }
        }

        if (critical_vgs.empty()) break; // All good

        for (int weak_vg_idx : critical_vgs) {
            // Check if still critical (might have been fixed by previous moves in this loop? 
            // No, we re-eval stats at start of loop, but let's re-eval to be precise or just proceed)
            // Simpler to stick to loop snapshot.

            int best_donor_node = -1;
            int best_donor_vg = -1;
            double min_delta_loss = std::numeric_limits<double>::max();
            bool found_candidate = false;

            int weak_root = selected_roots[weak_vg_idx];

            for (int neighbor_vg_idx = 0; neighbor_vg_idx < K; ++neighbor_vg_idx) {
                if (neighbor_vg_idx == weak_vg_idx) continue;
                // Only take from non-critical? 
                // "Candidate node w is not neighbor VG's Root"
                // "Removing w does not make neighbor VG critical"
                
                // If neighbor is already critical, we probably shouldn't steal from it unless it has excess resources 
                // but the instruction says: "Moving w... neighbor VG still satisfies survival threshold"
                // This implies neighbor MUST be satisfying threshold BEFORE removal.
                if (vg_stats[neighbor_vg_idx].is_critical) continue;

                for (int w_id : vg_members[neighbor_vg_idx]) {
                    if (w_id == selected_roots[neighbor_vg_idx]) continue; // Cannot move root

                    // Check connection to weak root
                    double bw_to_weak = get_connection_strength(adj, w_id, weak_root);
                    if (bw_to_weak <= 1e-9) continue; // Must have physical connection

                    // Check viability after removal
                    double w_compute = device_map[w_id].compute;
                    double w_memory = device_map[w_id].memory;
                    
                    double new_neighbor_compute = vg_stats[neighbor_vg_idx].total_compute - w_compute;
                    double new_neighbor_memory = vg_stats[neighbor_vg_idx].total_memory - w_memory;

                    if (new_neighbor_compute < config.P_min || new_neighbor_memory < config.M_min) continue;

                    // Calculate Delta Loss
                    double bw_to_current = get_connection_strength(adj, w_id, selected_roots[neighbor_vg_idx]);
                    double delta_loss = bw_to_current - bw_to_weak;

                    if (delta_loss < min_delta_loss) {
                        min_delta_loss = delta_loss;
                        best_donor_node = w_id;
                        best_donor_vg = neighbor_vg_idx;
                        found_candidate = true;
                    }
                }
            }

            if (found_candidate) {
                // Perform Move
                // Remove from neighbor
                auto& donors = vg_members[best_donor_vg];
                donors.erase(std::remove(donors.begin(), donors.end(), best_donor_node), donors.end());
                
                // Add to weak
                vg_members[weak_vg_idx].push_back(best_donor_node);
                dev_to_vg[best_donor_node] = weak_vg_idx;

                // Mark that we made a change, so we should re-evaluate
                optimization_active = true;
                
                // Update local stats view to reflect change immediately for next checking within this loop?
                // The algorithm usually implies iterative refinement. 
                // I will update stats locally to prevent a donor VG from being drained to death in one pass.
                vg_stats[best_donor_vg].total_compute -= device_map[best_donor_node].compute;
                vg_stats[best_donor_vg].total_memory -= device_map[best_donor_node].memory;
                // If it became critical (shouldn't by logic above), mark it? Logic ensured it didn't.
                
                vg_stats[weak_vg_idx].total_compute += device_map[best_donor_node].compute;
                vg_stats[weak_vg_idx].total_memory += device_map[best_donor_node].memory;
                
                // Check if weak_vg is still critical
                if (vg_stats[weak_vg_idx].total_compute >= config.P_min && 
                    vg_stats[weak_vg_idx].total_memory >= config.M_min) {
                    vg_stats[weak_vg_idx].is_critical = false;
                }
            }
        }
    }

    result.device_to_vg_map = dev_to_vg;

    // =========================================================
    // Phase 4: Max-Min Bottleneck Ordering
    // =========================================================

    if (K > 0) {
        std::vector<int> p(K);
        std::iota(p.begin(), p.end(), 0);

        // Precompute Root-to-Root bandwidths
        std::vector<std::vector<double>> root_adj(K, std::vector<double>(K, 0.0));
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                if (i == j) continue;
                root_adj[i][j] = get_connection_strength(adj, selected_roots[i], selected_roots[j]);
            }
        }

        std::vector<int> best_p = p;
        double max_min_bw = -1.0;

        // If K is very small (1), loop doesn't matter
        if (K == 1) {
            result.pipeline_order = p;
        } else {
            do {
                double current_min = std::numeric_limits<double>::max();
                for (size_t k = 0; k < K - 1; ++k) {
                    double bw = root_adj[p[k]][p[k+1]];
                    if (bw < current_min) current_min = bw;
                }

                if (current_min > max_min_bw) {
                    max_min_bw = current_min;
                    best_p = p;
                }
            } while (std::next_permutation(p.begin(), p.end()));
            
            result.pipeline_order = best_p;
        }
    }

    return result;
}

void RRAGC::run_example() {
    std::cout << "=== RRAGC Algorithm Example ===" << std::endl;

    // 1. Setup Dummy Devices
    std::vector<Device> devices = {
        {0, 100.0, 16.0}, // High compute, Root candidate
        {1, 50.0,  8.0},
        {2, 90.0,  16.0}, // High compute, Root candidate
        {3, 40.0,  4.0},
        {4, 30.0,  4.0},
        {5, 20.0,  2.0} // Weak node
    };

    // 2. Setup Dummy Links
    // 0 connected to 1, 2
    // 2 connected to 3
    // 4 connected to 1
    // 5 connected to 0
    std::vector<Link> links = {
        {0, 1, 10.0},
        {1, 0, 10.0},
        {0, 2, 5.0},
        {2, 0, 5.0},
        {2, 3, 8.0},
        {3, 2, 8.0},
        {1, 4, 6.0},
        {4, 1, 6.0},
        {0, 5, 2.0},
        {5, 0, 2.0},
        {2, 4, 1.0}, // weak link
        {4, 2, 1.0}
    };

    // 3. Config
    Config config;
    config.K = 2;
    // TBD: we must give different P_min and M_min for different models
    config.P_min = 60.0;
    config.M_min = 10.0;
    // config.alpha + config.beta = 1.0
    // config.alpha is linked to bandwidth importance
    // config.beta is linked to compute importance
    config.alpha = 0.7;
    config.beta = 0.3;

    // 4. Run Solve
    Result res = RRAGC::solve(devices, links, config);

    // 5. Output
    std::cout << "Selected Roots: ";
    for (int r : res.vg_roots) std::cout << r << " ";
    std::cout << std::endl;

    std::cout << "Pipeline Order (VG Indices): ";
    for (int idx : res.pipeline_order) std::cout << idx << " ";
    std::cout << std::endl;

    std::cout << "Device Assignments:" << std::endl;
    std::map<int, std::vector<int>> vg_reverse_map;
    for (auto const& [dev, vg] : res.device_to_vg_map) {
        vg_reverse_map[vg].push_back(dev);
    }
    
    for (int i = 0; i < config.K; ++i) {
        std::cout << "VG " << i << " (Root " << (i < res.vg_roots.size() ? std::to_string(res.vg_roots[i]) : "N/A") << "): ";
        for (int d : vg_reverse_map[i]) {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "===============================" << std::endl;
}

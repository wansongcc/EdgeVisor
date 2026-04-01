import itertools
import math


def _get_link_bw(adj, u, v):
    return float(adj.get(u, {}).get(v, 0.0))


def _get_connection_strength(adj, u, v):
    return max(_get_link_bw(adj, u, v), _get_link_bw(adj, v, u))


def _get_bandwidth(links, src, dst):
    for link in links:
        if int(link["src_id"]) == int(src) and int(link["dst_id"]) == int(dst):
            return float(link["bandwidth"])
    return 0.0


def _to_memory_gb(mem_value):
    mem = float(mem_value)
    # Compatible with both "GB-like numbers" and bytes.
    return mem / 1e9 if mem > 1e6 else mem


def rragc_solve(devices, links, cfg):
    if not devices:
        return {"device_to_vg_map": {}, "vg_roots": [], "pipeline_order": []}

    n = len(devices)
    k = min(int(cfg["K"]), n)

    adj = {}
    global_max_link_bw = 0.0
    for link in links:
        src = int(link["src_id"])
        dst = int(link["dst_id"])
        bw = float(link["bandwidth"])
        adj.setdefault(src, {})[dst] = bw
        global_max_link_bw = max(global_max_link_bw, bw)

    device_map = {int(d["id"]): d for d in devices}

    total_bw = {}
    max_bw = -1.0
    min_bw = float("inf")
    max_c = -1.0
    min_c = float("inf")
    for d in devices:
        dev_id = int(d["id"])
        bw_sum = 0.0
        for _, bw in adj.get(dev_id, {}).items():
            bw_sum += bw
        for link in links:
            if int(link["dst_id"]) == dev_id:
                bw_sum += float(link["bandwidth"])
        total_bw[dev_id] = bw_sum
        max_bw = max(max_bw, bw_sum)
        min_bw = min(min_bw, bw_sum)
        c = float(d["compute"])
        max_c = max(max_c, c)
        min_c = min(min_c, c)

    if min_bw > max_bw:
        min_bw = max_bw = 0.0
    if min_c > max_c:
        min_c = max_c = 0.0

    alpha = float(cfg["alpha"])
    beta = float(cfg["beta"])
    score_raw = {}
    for d in devices:
        dev_id = int(d["id"])
        b_hat = 1.0 if abs(max_bw - min_bw) < 1e-9 else (total_bw[dev_id] - min_bw) / (max_bw - min_bw)
        c_val = float(d["compute"])
        c_hat = 1.0 if abs(max_c - min_c) < 1e-9 else (c_val - min_c) / (max_c - min_c)
        score_raw[dev_id] = alpha * b_hat + beta * c_hat

    selected_roots = []
    selected_roots_set = set()
    for _ in range(k):
        best_candidate = None
        best_score = -1.0
        for d in devices:
            dev_id = int(d["id"])
            if dev_id in selected_roots_set:
                continue
            bw_link_u = 0.0
            if selected_roots:
                bw_link_u = max(_get_connection_strength(adj, dev_id, r) for r in selected_roots)
            rho = (bw_link_u / global_max_link_bw) if global_max_link_bw > 1e-9 else 0.0
            score_adj = score_raw[dev_id] * (1.0 - rho)
            if score_adj > best_score:
                best_score = score_adj
                best_candidate = dev_id
        if best_candidate is None:
            break
        selected_roots.append(best_candidate)
        selected_roots_set.add(best_candidate)

    k = len(selected_roots)

    vg_members = {i: [selected_roots[i]] for i in range(k)}
    dev_to_vg = {selected_roots[i]: i for i in range(k)}

    isolated = []
    for d in devices:
        dev_id = int(d["id"])
        if dev_id in selected_roots_set:
            continue
        best_vg = -1
        best_bw = -1.0
        for i in range(k):
            root_id = selected_roots[i]
            bw = _get_connection_strength(adj, dev_id, root_id)
            if bw > best_bw:
                best_bw = bw
                best_vg = i
        if best_bw > 0.0:
            vg_members[best_vg].append(dev_id)
            dev_to_vg[dev_id] = best_vg
        else:
            isolated.append(dev_id)

    for iso_id in isolated:
        target_vg = -1
        min_vg_compute = float("inf")
        for i in range(k):
            c_sum = sum(float(device_map[mid]["compute"]) for mid in vg_members[i])
            if c_sum < min_vg_compute:
                min_vg_compute = c_sum
                target_vg = i
        if target_vg >= 0:
            vg_members[target_vg].append(iso_id)
            dev_to_vg[iso_id] = target_vg

    max_iterations = n * 2
    iters = 0
    while iters < max_iterations:
        iters += 1
        optimization_active = False

        vg_stats = []
        critical = []
        for i in range(k):
            c_sum = sum(float(device_map[mid]["compute"]) for mid in vg_members[i])
            m_sum = sum(float(device_map[mid]["memory"]) for mid in vg_members[i])
            is_critical = c_sum < float(cfg["P_min"]) or m_sum < float(cfg["M_min"])
            vg_stats.append({"compute": c_sum, "memory": m_sum, "critical": is_critical})
            if is_critical:
                critical.append(i)

        if not critical:
            break

        for weak_vg in critical:
            best_node = None
            best_donor_vg = None
            min_delta_loss = float("inf")
            weak_root = selected_roots[weak_vg]

            for donor_vg in range(k):
                if donor_vg == weak_vg:
                    continue
                if vg_stats[donor_vg]["critical"]:
                    continue
                for w_id in list(vg_members[donor_vg]):
                    if w_id == selected_roots[donor_vg]:
                        continue
                    bw_to_weak = _get_connection_strength(adj, w_id, weak_root)
                    if bw_to_weak <= 1e-9:
                        continue
                    w_compute = float(device_map[w_id]["compute"])
                    w_memory = float(device_map[w_id]["memory"])
                    new_c = vg_stats[donor_vg]["compute"] - w_compute
                    new_m = vg_stats[donor_vg]["memory"] - w_memory
                    if new_c < float(cfg["P_min"]) or new_m < float(cfg["M_min"]):
                        continue
                    bw_to_current = _get_connection_strength(adj, w_id, selected_roots[donor_vg])
                    delta_loss = bw_to_current - bw_to_weak
                    if delta_loss < min_delta_loss:
                        min_delta_loss = delta_loss
                        best_node = w_id
                        best_donor_vg = donor_vg

            if best_node is not None:
                vg_members[best_donor_vg].remove(best_node)
                vg_members[weak_vg].append(best_node)
                dev_to_vg[best_node] = weak_vg
                optimization_active = True

                vg_stats[best_donor_vg]["compute"] -= float(device_map[best_node]["compute"])
                vg_stats[best_donor_vg]["memory"] -= float(device_map[best_node]["memory"])
                vg_stats[weak_vg]["compute"] += float(device_map[best_node]["compute"])
                vg_stats[weak_vg]["memory"] += float(device_map[best_node]["memory"])

        if not optimization_active:
            break

    if k == 0:
        pipeline_order = []
    elif k == 1:
        pipeline_order = [0]
    else:
        root_adj = [[0.0 for _ in range(k)] for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                root_adj[i][j] = _get_connection_strength(adj, selected_roots[i], selected_roots[j])
        best_perm = list(range(k))
        max_min_bw = -1.0
        for perm in itertools.permutations(range(k)):
            current_min = float("inf")
            for idx in range(k - 1):
                current_min = min(current_min, root_adj[perm[idx]][perm[idx + 1]])
            if current_min > max_min_bw:
                max_min_bw = current_min
                best_perm = list(perm)
        pipeline_order = best_perm

    for vg_id in vg_members:
        vg_members[vg_id] = sorted(vg_members[vg_id])

    return {
        "device_to_vg_map": dev_to_vg,
        "vg_roots": selected_roots,
        "pipeline_order": pipeline_order,
        "vg_members": vg_members,
    }


def ccwf_solve(workers, task):
    if not workers:
        return {"alphas": [], "estimated_latency_s": 0.0}

    h_vals = []
    k_vals = []
    min_h = float("inf")
    max_h_plus_k = 0.0

    for w in workers:
        bw = float(w["bandwidth_bps"])
        comp = float(w["compute_flops"])
        t_recv = float(task["input_bytes"]) / bw
        t_comp_full = float(task["total_flops"]) / comp
        t_send_full = float(task["output_bytes"]) / bw
        k = t_comp_full + t_send_full
        h_vals.append(t_recv)
        k_vals.append(k)
        min_h = min(min_h, t_recv)
        max_h_plus_k = max(max_h_plus_k, t_recv + k)

    low = min_h
    high = max_h_plus_k * 2.0
    best_t = high

    for _ in range(100):
        mid = low + (high - low) / 2.0
        sum_alpha = 0.0
        for i, w in enumerate(workers):
            alpha = 0.0
            if mid > h_vals[i]:
                alpha = (mid - h_vals[i]) / k_vals[i]
            alpha = min(alpha, float(w["max_alpha_mem"]))
            sum_alpha += alpha
        if sum_alpha >= 1.0:
            best_t = mid
            high = mid
        else:
            low = mid

    alphas = []
    alpha_sum = 0.0
    for i, w in enumerate(workers):
        alpha = 0.0
        if best_t > h_vals[i]:
            alpha = (best_t - h_vals[i]) / k_vals[i]
        alpha = min(alpha, float(w["max_alpha_mem"]))
        alpha = max(alpha, 0.0)
        alphas.append(alpha)
        alpha_sum += alpha

    if alpha_sum > 0.0:
        alphas = [a / alpha_sum for a in alphas]

    return {"alphas": alphas, "estimated_latency_s": best_t}


def olp_solve(vgs, model):
    if not vgs or int(model["total_layers"]) <= 0:
        return []

    num_vgs = len(vgs)
    total_layers = int(model["total_layers"])
    inf = float("inf")
    dp = [[inf] * (total_layers + 1) for _ in range(num_vgs + 1)]
    split = [[-1] * (total_layers + 1) for _ in range(num_vgs + 1)]
    dp[0][0] = 0.0

    for k in range(1, num_vgs + 1):
        vg = vgs[k - 1]
        for l in range(total_layers + 1):
            for j in range(l + 1):
                if dp[k - 1][j] == inf:
                    continue
                count = l - j
                if count > int(vg["max_layers_capacity"]):
                    continue
                t_calc = count * float(vg["unit_time_ms"])
                t_comm = 0.0
                if count > 0 and float(vg["next_link_bw_gbps"]) > 1e-9:
                    transfer_seconds = (float(model["activation_size_gb"]) * 8.0) / float(vg["next_link_bw_gbps"])
                    t_comm = transfer_seconds * 1000.0
                stage_cost = t_calc + t_comm
                bottleneck = max(dp[k - 1][j], stage_cost)
                if bottleneck < dp[k][l]:
                    dp[k][l] = bottleneck
                    split[k][l] = j

    if dp[num_vgs][total_layers] == inf:
        return []

    allocation = [0] * num_vgs
    curr_l = total_layers
    for k in range(num_vgs, 0, -1):
        prev_l = split[k][curr_l]
        allocation[k - 1] = curr_l - prev_l
        curr_l = prev_l
    return allocation


def run_initialization(devices, links, rragc_config, layer_task, model_config):
    rr = rragc_solve(devices, links, rragc_config)
    device_map = {int(d["id"]): d for d in devices}

    vg_profiles = []
    intravg = {}
    pipeline_order = rr["pipeline_order"]
    vg_members = rr["vg_members"]

    for idx, vg_id in enumerate(pipeline_order):
        root_id = rr["vg_roots"][vg_id]
        members = list(vg_members[vg_id])
        if root_id in members:
            members.remove(root_id)
        members = [root_id] + members

        workers = []
        total_mem_capacity_layers = 0
        for dev_id in members:
            dev = device_map[dev_id]
            total_mem_capacity_layers += int(_to_memory_gb(dev["memory"]) / 0.5)
            if dev_id == root_id:
                bw_to_root = 1000.0
            else:
                bw_to_root = _get_bandwidth(links, root_id, dev_id)
                if bw_to_root <= 1e-9:
                    bw_to_root = 1e-9
            workers.append(
                {
                    "id": dev_id,
                    "compute_flops": float(dev["compute"]),
                    "bandwidth_bps": bw_to_root * 1e9,
                    "max_alpha_mem": 1.0,
                }
            )

        ccwf = ccwf_solve(workers, layer_task)
        intravg[str(vg_id)] = {
            "worker_ids": members,
            "alphas": ccwf["alphas"],
            "estimated_latency_s": ccwf["estimated_latency_s"],
        }

        next_link_bw = 0.0
        if idx + 1 < len(pipeline_order):
            next_vg_id = pipeline_order[idx + 1]
            next_root = rr["vg_roots"][next_vg_id]
            next_link_bw = _get_bandwidth(links, root_id, next_root)

        vg_profiles.append(
            {
                "id": vg_id,
                "unit_time_ms": ccwf["estimated_latency_s"] * 1000.0,
                "max_layers_capacity": total_mem_capacity_layers,
                "next_link_bw_gbps": next_link_bw,
            }
        )

    inter_layers = olp_solve(vg_profiles, model_config)

    return {
        "rragc": {
            "device_to_vg_map": rr["device_to_vg_map"],
            "vg_roots": rr["vg_roots"],
            "pipeline_order": pipeline_order,
            "vg_members": vg_members,
        },
        "intravg": intravg,
        "vg_profiles": vg_profiles,
        "intervg_layers": inter_layers,
        "model_config": model_config,
    }


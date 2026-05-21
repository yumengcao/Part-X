import numpy as np

def _flatten_theta_regions(theta_dict):

    regions = []
    if not theta_dict:
        return regions
    for it_key, subdict in theta_dict.items():
        if not subdict:
            continue
        for subk, bounds in subdict.items():
            # 这里假定 bounds 已经是 [(lo1,hi1),(lo2,hi2),...] 这种格式
            # 若你的实际结构是一个 subregion object，请在此处把它转换为上面格式
            if bounds is None:
                continue
            regions.append(bounds)
    return regions

def compute_misclassification_mc_iter(theta_minus, theta_plus, region_bounds, func, n_samples=50000, seed=123):
    
    rng = np.random.RandomState(seed)
    lo = np.array([b[0] for b in region_bounds], dtype=float)
    hi = np.array([b[1] for b in region_bounds], dtype=float)
    d = len(lo)
    pts = rng.uniform(low=lo, high=hi, size=(int(n_samples), d))

    # 评估 f(x)（逐点 eval，保持通用和与现有代码一致的语义）
    try:
        vals = np.asarray(func(pts)).ravel()
        if vals.size != n_samples:
            # if returned different size, force fallback to loop
            raise ValueError("vectorized func returned unexpected length")
    except Exception:
        # 逐点调用的安全回退（支持 func(x) 或 func(x, r)）
        vals = np.empty(n_samples, dtype=float)
        for i in range(n_samples):
            x = pts[i]
            try:
                out = func(x, 1)   # 先尝试带 replication
            except TypeError:
                out = func(x)     # 再尝试不带 replication
            # 如果返回数组/列表，取其均值；否则直接取标量
            if hasattr(out, "__iter__"):
                out = float(np.mean(out))
            else:
                out = float(out)
            vals[i] = out
    
    # vals = np.empty(n_samples, dtype=float)
    # for i in range(n_samples):
    #     X = pts[i]  # eval 中使用的名字是 X
    #     vals[i] = float(eval(func_str))

    true_in = vals <= 0.0  # level set

    # 展平 theta_plus / theta_minus 得到 region 列表
    plus_regions = _flatten_theta_regions(theta_plus)
    minus_regions = _flatten_theta_regions(theta_minus)

    # masks
    mask_pruned = np.zeros(n_samples, dtype=bool)
    mask_maintained = np.zeros(n_samples, dtype=bool)

    
    for b_list in plus_regions:
        if not b_list:
            continue
        lo_arr = np.array([bb[0] for bb in b_list], dtype=float)
        hi_arr = np.array([bb[1] for bb in b_list], dtype=float)
        inside = np.all((pts >= lo_arr) & (pts <= hi_arr), axis=1)
        mask_pruned |= inside

    # 对每个 minus region 标记 pts
    for b_list in minus_regions:
        if not b_list:
            continue
        lo_arr = np.array([bb[0] for bb in b_list], dtype=float)
        hi_arr = np.array([bb[1] for bb in b_list], dtype=float)
        inside = np.all((pts >= lo_arr) & (pts <= hi_arr), axis=1)
        mask_maintained |= inside

    # 计算错误
    incorrect_prune_count = np.count_nonzero(mask_pruned & true_in)        # pruned but actually in L
    incorrect_maintain_count = np.count_nonzero(mask_maintained & (~true_in))  # maintained but actually out of L

    V_total = float(np.prod(hi - lo))
    cell_vol_est = V_total / float(n_samples)
    V_incorrect_prune = incorrect_prune_count * cell_vol_est
    V_incorrect_maintain = incorrect_maintain_count * cell_vol_est
    V_mis = V_incorrect_prune + V_incorrect_maintain

    return {
        'V_total': V_total,
        'V_incorrect_prune': float(V_incorrect_prune),
        'V_incorrect_maintain': float(V_incorrect_maintain),
        'V_misclassified': float(V_mis),
        'ratio_misclassified': float(V_mis / V_total),
        'ratio_incorrect_prune': float(V_incorrect_prune / V_total),
        'ratio_incorrect_maintain': float(V_incorrect_maintain / V_total),
        'n_samples': int(n_samples)
    }
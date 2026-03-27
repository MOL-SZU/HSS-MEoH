import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    HSS (Hypervolume-based Subset Selection): 贪心选择超体积贡献最大的 k 个点

    参数:
        points: np.ndarray
            所有候选点，形状为 (N, D)，其中 N 是点数，D 是目标维度
        k: int
            需要选择的点数
        reference_point: np.ndarray, optional
            参考点，形状为 (D,)。如果为 None，则使用 points 的最大值 * 1.1

    返回:
        subset: np.ndarray
            选出的子集，形状为 (k, D)
    """
    import numpy as np
    import heapq

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k <= 0:
        return np.zeros((0, D), dtype=pts.dtype)
    if k >= N:
        return pts.copy()

    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Clip points to the reference (points worse than ref don't help beyond ref)
    clipped = np.minimum(pts, ref)
    # positive deltas: how much each point can extend towards the reference (minimization HV convention)
    delta_pos = np.clip(ref - clipped, a_min=0.0, a_max=None)  # shape (N, D)

    # If no positive contribution anywhere, fallback to farthest by L2
    if not np.any(delta_pos > 0):
        dists = np.linalg.norm(ref - pts, axis=1)
        idx = np.argsort(-dists)[:k]
        return pts[idx].copy()

    # Stage 1: cheap candidate reduction using L1 proxy (sum of positive deltas)
    proxy = np.sum(delta_pos, axis=1)
    min_candidates = max(50, 6 * k)
    top_m = int(min(N, min_candidates))
    # ensure at least k candidates
    top_m = max(top_m, k)
    cand_idx = np.argsort(-proxy)[:top_m]
    cand_idx = np.asarray(cand_idx, dtype=int)
    delta_cand = delta_pos[cand_idx]  # (Nc, D)
    Nc = delta_cand.shape[0]

    # Stage 2: directional scalarizations (compact, efficient)
    # number of directions: scale with D and k but keep small to be fast
    num_dir = int(min(120, max(24, 6 * min(D, max(1, k // 2)))))
    eps = 1e-12
    # sample positive directions via abs(normal) then normalize
    R = np.random.normal(size=(num_dir, D))
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    W = np.abs(R / norms) + eps  # (num_dir, D)

    # hv directional values: for each candidate and direction j, hv_val = min_m ( delta_cand[i,m] / W[j,m] )
    with np.errstate(divide='ignore', invalid='ignore'):
        vals = delta_cand[:, None, :] / W[None, :, :]  # (Nc, num_dir, D)
        hv_vals = np.min(vals, axis=2)  # (Nc, num_dir)
    hv_vals = np.clip(hv_vals, a_min=0.0, a_max=None)

    # normalize per-direction by median to robustly balance directions
    dir_med = np.median(hv_vals, axis=0)
    dir_scale = np.where(dir_med <= 0.0, 1.0, dir_med)
    hv_norm = hv_vals / (dir_scale[None, :] + eps)

    # direction weights via softmax of log-median (favor informative directions)
    temp = 0.7
    log_med = np.log(dir_scale + eps)
    shifted = log_med - np.max(log_med)
    exps = np.exp(shifted / max(temp, eps))
    dir_weights = exps / (np.sum(exps) + eps)
    if not np.isfinite(np.sum(dir_weights)) or np.sum(dir_weights) <= 0.0:
        dir_weights = np.ones_like(dir_weights) / float(len(dir_weights))

    # concave transform to soften large outliers (power < 1)
    power = 0.6
    hv_pow = np.power(hv_norm, power)  # (Nc, num_dir)

    # Precompute an upper bound score per candidate for quick fallback checks
    base_scores = hv_pow.dot(dir_weights)  # (Nc,)

    # Lazy-greedy on reduced pool using directional approximations
    cur_best = np.zeros(num_dir, dtype=float)
    selected_local = []
    available = np.ones(Nc, dtype=bool)
    # Max-heap: store (-gain, local_idx, version)
    heap = [(-float(base_scores[i]), int(i), 0) for i in range(Nc)]
    heapq.heapify(heap)
    selected_size = 0
    max_iters = min(k * 50 + Nc, Nc * 10 + k * 10)

    iters = 0
    while selected_size < k and heap and iters < max_iters:
        iters += 1
        neg_gain, local_idx, version = heapq.heappop(heap)
        stored_gain = -neg_gain
        if not available[local_idx]:
            continue
        # if version matches current selected_size, accept as up-to-date
        if version == selected_size:
            # accept
            selected_local.append(int(local_idx))
            available[local_idx] = False
            # update cur_best (directional transformed best)
            # elementwise max
            cur_best = np.maximum(cur_best, hv_pow[local_idx])
            selected_size += 1
            continue
        # otherwise recompute marginal gain against current cur_best
        diff = hv_pow[local_idx] - cur_best
        # only positive contributions matter
        np.maximum(diff, 0.0, out=diff)
        marg = float(diff.dot(dir_weights))
        # push updated with current version
        heapq.heappush(heap, (-marg, int(local_idx), selected_size))

    # If not enough selected (rare), fill by highest base_scores among remaining
    if selected_size < k:
        rem = np.where(available)[0]
        if rem.size > 0:
            order = np.argsort(-base_scores[rem])
            need = k - selected_size
            pick_local = rem[order[:need]]
            for li in pick_local:
                if available[li]:
                    selected_local.append(int(li))
                    available[li] = False
                    selected_size += 1
                    if selected_size == k:
                        break

    # Final safety: if still short (shouldn't), fill by global proxy excluding already selected
    if len(selected_local) < k:
        chosen_global = set(cand_idx[selected_local].tolist())
        remain_global = [i for i in np.argsort(-proxy) if i not in chosen_global]
        need = k - len(selected_local)
        for idx in remain_global[:need]:
            selected_local.append(int(np.where(cand_idx == idx)[0][0]) if idx in cand_idx else -1)

        # if some entries were -1 (outside candidate pool), replace with direct global picks
        final_globals = []
        for li in selected_local:
            if li == -1:
                # pick next remaining global index not already chosen
                for idx in remain_global:
                    if idx not in chosen_global:
                        chosen_global.add(idx)
                        final_globals.append(idx)
                        break
            else:
                final_globals.append(int(cand_idx[li]))
        # ensure unique and correct length
        selected_globals = []
        seen = set()
        for g in final_globals:
            if g not in seen:
                selected_globals.append(g)
                seen.add(g)
            if len(selected_globals) == k:
                break
        # pad if necessary
        if len(selected_globals) < k:
            for i in range(N):
                if i not in seen:
                    selected_globals.append(i)
                    seen.add(i)
                if len(selected_globals) == k:
                    break
        subset = pts[np.array(selected_globals, dtype=int), :].copy()
        return subset

    # Map local candidate indices to global indices
    selected_globals = cand_idx[np.array(selected_local, dtype=int)]
    # Ensure exactly k and unique (if duplicates for any reason, pad by proxy)
    unique_globals = []
    seen = set()
    for g in selected_globals:
        if g not in seen:
            unique_globals.append(int(g))
            seen.add(int(g))
        if len(unique_globals) == k:
            break
    if len(unique_globals) < k:
        for i in np.argsort(-proxy):
            if i not in seen:
                unique_globals.append(int(i))
                seen.add(int(i))
            if len(unique_globals) == k:
                break

    subset = pts[np.array(unique_globals[:k], dtype=int), :].copy()
    return subset


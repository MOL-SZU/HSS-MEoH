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

    # lazy import for pygmo hypervolume
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo (pg) is required for hypervolume computations but is not available.") from e

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N, D = points.shape

    # trivial cases
    if k <= 0 or N == 0:
        return np.empty((0, D), dtype=float)
    if k >= N:
        return points.copy()

    # reference point
    if reference_point is None:
        ref = np.max(points, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).ravel()
        if ref.size != D:
            raise ValueError("reference_point must have the same dimensionality as points")

    # compute singleton hypervolumes (closed form for minimization to reference)
    diffs = ref.reshape(1, -1) - points  # (N, D)
    nonneg = np.maximum(diffs, 0.0)
    indiv_hv = np.prod(nonneg, axis=1)

    # ------- New scoring: blend of singleton HV and a normalized "isolation" (distance) score -------
    # Compute isolation score as mean distance to a small set of top-anchor points (cheap proxy for isolation)
    # This is lighter than full pairwise NN for large N.
    top_anchor_count = min(10, N)
    anchors_idx = np.argsort(-indiv_hv)[:top_anchor_count]
    anchors = points[anchors_idx]
    # distances shape (N, top_anchor_count)
    diff = points[:, None, :] - anchors[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    d = np.sqrt(d2)
    iso = np.mean(d, axis=1)  # larger means more isolated / diverse
    # normalize isolation to [0,1]
    eps = 1e-12
    iso_min = iso.min()
    iso_max = iso.max()
    iso_norm = (iso - iso_min) / (iso_max - iso_min + eps)

    # combine scores: hv * (1 + gamma * iso_norm)
    gamma = 0.7  # weight of distance-based diversity in score
    combined_score = indiv_hv * (1.0 + gamma * iso_norm)

    # Build a compact candidate pool to limit expensive HV evaluations:
    POOL_MIN = 30
    POOL_PER_K = 8
    POOL_MAX = 600
    pool_size = int(min(N, max(POOL_MIN, min(POOL_MAX, POOL_PER_K * max(1, k)))))
    pool_size = max(pool_size, min(N, k))

    # Preselect a larger shortlist by combined score then sample a diverse subset with farthest-first.
    shortlist_multiplier = 4
    shortlist_size = min(N, max(pool_size * shortlist_multiplier, 200))
    sorted_by_combined = np.argsort(-combined_score)
    shortlist = list(sorted_by_combined[:shortlist_size])

    if len(shortlist) <= pool_size:
        pool_indices = shortlist.copy()
    else:
        # Farthest-first selection within shortlist to ensure diversity in the pool
        pool_indices = []
        first = shortlist[0]
        pool_indices.append(int(first))
        remaining = shortlist[1:]
        rem_pts = points[np.array(remaining, dtype=int)]
        sel_pt = points[int(first)].reshape(1, -1)
        min_d2 = np.sum((rem_pts - sel_pt) ** 2, axis=1)
        while len(pool_indices) < pool_size and len(remaining) > 0:
            idx_arg = int(np.argmax(min_d2))
            chosen = remaining[idx_arg]
            pool_indices.append(int(chosen))
            # remove chosen
            remaining.pop(idx_arg)
            min_d2 = np.delete(min_d2, idx_arg)
            if len(remaining) == 0 or len(pool_indices) >= pool_size:
                break
            new_pt = points[int(chosen)].reshape(1, -1)
            rem_array = points[np.array(remaining, dtype=int)]
            d2_new = np.sum((rem_array - new_pt) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2_new)

    # Prepare lazy heap for pool entries: (-estimate, idx, version)
    # Use combined_score as initial (optimistic) estimate because it blends hv and diversity
    heap = []
    for idx in pool_indices:
        # small numerical guard
        est = float(combined_score[int(idx)])
        heap.append((-est, int(idx), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_points = np.empty((0, D), dtype=float)
    current_hv = 0.0
    version = 0
    tol = 1e-12

    def hv_with_candidate(curr_pts: np.ndarray, cand_idx: int) -> float:
        if curr_pts.shape[0] == 0:
            arr = points[cand_idx].reshape(1, -1)
            return float(pg.hypervolume(arr).compute(ref))
        else:
            arr = np.vstack([curr_pts, points[cand_idx].reshape(1, -1)])
            return float(pg.hypervolume(arr).compute(ref))

    # batch-verified lazy greedy inside pool
    BATCH_SIZE = min(10, max(3, pool_size // 8))
    MAX_ITERS = max(2000, k * 300)
    iters = 0

    while len(selected) < k and heap and iters < MAX_ITERS:
        iters += 1
        neg_est, idx, ver = heapq.heappop(heap)
        if idx in selected_set:
            continue
        est = -neg_est

        # if estimate is up-to-date accept and compute hv precisely for update
        if ver == version:
            if est < 0 and est > -tol:
                est = 0.0
            # accept
            selected.append(int(idx))
            selected_set.add(int(idx))
            current_points = np.vstack([current_points, points[idx].reshape(1, -1)])
            current_hv = float(pg.hypervolume(current_points).compute(ref))
            version += 1
            continue

        # stale entry: gather a small batch of top candidates for verification
        batch = [(idx, est)]
        while len(batch) < BATCH_SIZE and heap:
            neg_e, j, vj = heapq.heappop(heap)
            if j in selected_set:
                continue
            batch.append((j, -neg_e))

        # compute true gains for the batch
        true_info = []
        for (j, _) in batch:
            hvw = hv_with_candidate(current_points, j)
            tg = hvw - current_hv
            if tg < 0 and tg > -tol:
                tg = 0.0
            true_info.append((int(j), float(tg), float(hvw)))

        # choose best candidate by true marginal gain, tie-break by indiv_hv
        best_j, best_gain, best_hv_with = max(true_info, key=lambda t: (t[1], indiv_hv[t[0]]))
        if best_gain <= tol:
            break

        # accept best
        selected.append(int(best_j))
        selected_set.add(int(best_j))
        current_points = np.vstack([current_points, points[best_j].reshape(1, -1)])
        current_hv = float(best_hv_with)
        prev_version = version
        version += 1

        # push back the rest of batch with their computed gains tied to prev_version
        for (j, tg, hvw) in true_info:
            if j == best_j:
                continue
            heapq.heappush(heap, (-float(tg), int(j), prev_version))

    # If we didn't reach k, fill from remaining pool (by combined score then indiv_hv) then global indiv_hv
    if len(selected) < k:
        pool_remaining = [i for i in pool_indices if i not in selected_set]
        pool_remaining_sorted = sorted(pool_remaining, key=lambda i: combined_score[i], reverse=True)
        for idx in pool_remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))

    if len(selected) < k:
        for idx in np.argsort(-indiv_hv):
            if idx in selected_set:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            if len(selected) >= k:
                break

    # Final safety pad (unlikely)
    if len(selected) < k:
        rng = np.random.default_rng(2026)
        unselected = [i for i in range(N) if i not in selected_set]
        need = k - len(selected)
        if len(unselected) == 0:
            picks = rng.choice(N, size=need, replace=True)
            selected.extend([int(x) for x in picks])
        else:
            picks = rng.choice(unselected, size=min(need, len(unselected)), replace=False)
            selected.extend([int(x) for x in picks])
            if len(selected) < k:
                extra = rng.choice(N, size=k - len(selected), replace=True)
                selected.extend([int(x) for x in extra])

    selected = selected[:k]
    selected = sorted(int(i) for i in selected)
    return points[np.array(selected, dtype=int), :].copy()


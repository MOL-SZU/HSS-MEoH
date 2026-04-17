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

    # hypervolume library (pygmo) is required
    try:
        import pygmo as pg
    except Exception as _pg_err:
        raise ImportError("pygmo (pg) is required for hypervolume computations but is not available.") from _pg_err

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

    # singleton hypervolume (closed form for minimization to reference)
    diffs = ref.reshape(1, -1) - points  # (N, D)
    nonneg = np.maximum(diffs, 0.0)
    indiv_hv = np.prod(nonneg, axis=1)

    # Build a compact candidate pool: balance top indiv hv and diversity (farthest-first on shortlist)
    POOL_MIN = 40
    POOL_PER_K = 6
    POOL_MAX = 600
    pool_size = int(min(N, max(POOL_MIN, min(POOL_MAX, POOL_PER_K * max(1, k)))))
    pool_size = max(pool_size, min(N, k))

    # shortlist size (a bit larger than final pool to allow diversity sampling)
    shortlist_size = min(N, max(200, 8 * k))
    sorted_idx_desc = np.argsort(-indiv_hv)
    shortlist = list(sorted_idx_desc[:shortlist_size])

    if len(shortlist) <= pool_size:
        pool_indices = shortlist.copy()
    else:
        # farthest-first sampling within shortlist for diversity
        pool_indices = []
        first = shortlist[0]
        pool_indices.append(int(first))
        remaining = shortlist[1:].copy()
        # compute squared distances from remaining to selected set incrementally
        rem_array = points[np.array(remaining, dtype=int)]
        sel_pt = points[int(first)].reshape(1, -1)
        min_d2 = np.sum((rem_array - sel_pt) ** 2, axis=1)
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

    # Prepare selection
    selected = []
    selected_set = set()
    current_points = np.empty((0, D), dtype=float)
    current_hv = 0.0
    tol = 1e-12

    # helper to compute hv(curr U {cand})
    def hv_with_candidate(curr_pts: np.ndarray, cand_idx: int) -> float:
        if curr_pts.shape[0] == 0:
            arr = points[cand_idx].reshape(1, -1)
            return float(pg.hypervolume(arr).compute(ref))
        else:
            arr = np.vstack([curr_pts, points[cand_idx].reshape(1, -1)])
            return float(pg.hypervolume(arr).compute(ref))

    # Main greedy loop but evaluate only a small top subset per iteration to reduce HV calls
    # top_evals controls how many candidates (by indiv_hv) we evaluate exactly each round
    MAX_ITERS = max(1000, k * 100)
    iters = 0
    while len(selected) < k and iters < MAX_ITERS:
        iters += 1
        remaining_pool = [i for i in pool_indices if i not in selected_set]
        if not remaining_pool:
            break

        # number of exact HV evaluations this round (scale with k but capped)
        top_evals = max(10, min(len(remaining_pool), int(4 * k)))
        # choose top candidates by indiv_hv as affordable shortlist this round
        cand_sorted = sorted(remaining_pool, key=lambda i: indiv_hv[i], reverse=True)[:top_evals]

        best_idx = None
        best_gain = -np.inf
        best_hv_with = None

        # evaluate true marginal gains for chosen candidates
        for j in cand_sorted:
            hvw = hv_with_candidate(current_points, j)
            gain = hvw - current_hv
            # numerical guard
            if gain < 0 and gain > -tol:
                gain = 0.0
            if best_idx is None:
                best_idx = int(j)
                best_gain = float(gain)
                best_hv_with = float(hvw)
            else:
                # choose by larger gain, tie-break by indiv_hv then by smaller index for determinism
                if (gain > best_gain + tol) or (abs(gain - best_gain) <= tol and (indiv_hv[j] > indiv_hv[best_idx] or (indiv_hv[j] == indiv_hv[best_idx] and j < best_idx))):
                    best_idx = int(j)
                    best_gain = float(gain)
                    best_hv_with = float(hvw)

        # If the best evaluated candidate gives negligible gain, stop early
        if best_idx is None or best_gain <= tol:
            break

        # Accept best
        selected.append(int(best_idx))
        selected_set.add(int(best_idx))
        current_points = np.vstack([current_points, points[best_idx].reshape(1, -1)])
        current_hv = float(best_hv_with)

    # Fill remaining slots from the remaining pool by indiv_hv, then globally by indiv_hv
    if len(selected) < k:
        pool_remaining = [i for i in pool_indices if i not in selected_set]
        pool_remaining_sorted = sorted(pool_remaining, key=lambda i: indiv_hv[i], reverse=True)
        for idx in pool_remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))

    if len(selected) < k:
        for idx in sorted_idx_desc:
            if idx in selected_set:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            if len(selected) >= k:
                break

    # Final safety pad if still short (very unlikely)
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
    # preserve original ordering by index to be deterministic
    selected = sorted(int(i) for i in selected)
    return points[np.array(selected, dtype=int), :].copy()


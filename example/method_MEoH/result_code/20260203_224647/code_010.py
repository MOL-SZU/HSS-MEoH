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

    EPS = 1e-12

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape

    if k <= 0 or N == 0:
        return np.zeros((0, D), dtype=float)

    if reference_point is None:
        reference_point = points.max(axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(D,)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Build axis-aligned boxes between each point and the reference point
    lows = np.minimum(points, reference_point)
    highs = np.maximum(points, reference_point)
    extents = np.maximum(highs - lows, 0.0)
    indiv_vols = np.prod(extents, axis=1)

    # Degenerate case: all zero volumes -> fallback deterministic selection by closeness
    if np.all(indiv_vols <= EPS):
        idxs = np.argsort(np.sum((reference_point - points) ** 2, axis=1))[:min(k, N)]
        return points[idxs].copy()

    # Global bounding box for uniform sampling
    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    global_span = np.maximum(global_high - global_low, 0.0)
    if not np.all(global_span > 0):
        # If degenerate span on any axis, fallback to top volumes
        idxs = np.argsort(-indiv_vols)[:min(k, N)]
        return points[idxs].copy()
    global_vol = float(np.prod(global_span))

    if global_vol <= EPS:
        idxs = np.argsort(-indiv_vols)[:min(k, N)]
        return points[idxs].copy()

    # Monte-Carlo sample budget: reduced for speed but scales with k and D
    M = int(np.clip(800 + 200 * min(k, 40) + 80 * D, 600, 12000))

    rng = np.random.default_rng()
    samples = global_low + rng.random((M, D)) * global_span  # (M, D)

    # Compute coverage matrix: covers[i, m] = True if box i covers sample m
    ge_low = samples[:, None, :] >= (lows[None, :, :] - EPS)
    le_high = samples[:, None, :] <= (highs[None, :, :] + EPS)
    within = np.logical_and(ge_low, le_high).all(axis=2)  # shape (M, N)
    covers = within.T  # shape (N, M), boolean

    # Use optional sample weighting (uniform here)
    sample_weights = np.ones(M, dtype=float)

    # Initial uncovered samples
    covered = np.zeros(M, dtype=bool)
    remaining = set(range(N))
    selected = []

    # Initial marginal weighted counts (float)
    marg_counts = covers.astype(float).dot(sample_weights)

    # Normalize individual volumes
    max_vol = float(indiv_vols.max()) if indiv_vols.max() > 0 else 1.0
    indiv_vol_norm = indiv_vols / max_vol

    # New score function parameters (different from original)
    # Reduce volumetric bias (smaller beta) to emphasize sampled marginal gain,
    # but increase exponent to sharpen contrast among large volumes.
    beta = 0.35   # lower weight of volumetric bias
    alpha = 1.25  # sharper exponent to accentuate very large volumes
    # Scale volumetric bias to sample-count units, but use sqrt(M) to moderate influence
    vol_bias = beta * (indiv_vol_norm ** alpha) * float(np.sqrt(max(1, M)))

    # Initial scores: marginal (approx) + volumetric bias
    scores = marg_counts + vol_bias

    # Build max-heap (as min-heap of negative values). Tie-breaker: larger indiv_vol preferred
    heap = []
    for i in range(N):
        if indiv_vols[i] <= EPS:
            continue
        heapq.heappush(heap, (-float(scores[i]), -float(indiv_vols[i]), int(i)))

    # Lazy-greedy selection based on score (marginal_count + vol_bias)
    while len(selected) < min(k, N) and heap:
        neg_score, neg_vol, idx = heapq.heappop(heap)
        idx = int(idx)
        if idx not in remaining:
            continue
        # Recompute actual marginal weighted count with current 'covered' mask
        if covers[idx].any():
            actual_marg = float(np.dot(covers[idx].astype(float), (~covered).astype(float) * sample_weights))
        else:
            actual_marg = 0.0
        if actual_marg <= EPS:
            remaining.discard(idx)
            continue
        # recompute score using actual marginal
        actual_score = actual_marg + beta * (indiv_vol_norm[idx] ** alpha) * float(np.sqrt(max(1, M)))
        # If score changed significantly, push updated
        if abs(actual_score + neg_score) > 1e-7:
            heapq.heappush(heap, (-float(actual_score), neg_vol, idx))
            continue
        # accept idx
        selected.append(idx)
        remaining.discard(idx)
        # update covered samples
        covered |= covers[idx]

    # If still fewer than k selected, fill deterministically by individual volumes among remaining
    if len(selected) < k:
        rem = np.array([i for i in range(N) if i in remaining], dtype=int)
        if rem.size > 0:
            order = rem[np.argsort(-indiv_vols[rem])]
            for i in order:
                if len(selected) >= k:
                    break
                selected.append(int(i))
                remaining.discard(int(i))

    selected = selected[:k]

    # Focused local improvement via weighted sampled-coverage based swaps (smaller, targeted)
    if selected:
        sel_set = set(selected)
        sel_arr = np.array(selected, dtype=int)
        cover_counts = covers[sel_arr, :].astype(float).sum(axis=0)  # (M,)
        curr_covered_weight = float(np.dot((cover_counts > 0).astype(float), sample_weights))
        unselected = np.array([i for i in range(N) if i not in sel_set], dtype=int)
        if unselected.size > 0:
            u_order = unselected[np.argsort(-indiv_vols[unselected])]
            top_unselected = u_order[:min(150, u_order.size)]
        else:
            top_unselected = np.array([], dtype=int)

        max_swaps = min(3 * k + 30, 300)
        swaps = 0
        improved = True
        while improved and swaps < max_swaps:
            improved = False
            swaps += 1
            # try for each selected element to find an improving swap
            for si_idx, s in enumerate(list(selected)):
                cover_counts_excl = cover_counts - covers[s].astype(float)
                for cand in top_unselected:
                    if cand in selected:
                        continue
                    new_counts = cover_counts_excl + covers[cand].astype(float)
                    new_covered_weight = float(np.dot((new_counts > 0).astype(float), sample_weights))
                    # accept swap if weighted coverage increases by a small margin
                    if new_covered_weight > curr_covered_weight + 1e-9:
                        # accept swap
                        selected[si_idx] = int(cand)
                        # update structures
                        cover_counts = new_counts
                        curr_covered_weight = new_covered_weight
                        sel_set = set(selected)
                        unselected = np.array([i for i in range(N) if i not in sel_set], dtype=int)
                        if unselected.size > 0:
                            u_order = unselected[np.argsort(-indiv_vols[unselected])]
                            top_unselected = u_order[:min(150, u_order.size)]
                        else:
                            top_unselected = np.array([], dtype=int)
                        improved = True
                        break
                if improved:
                    break

    # Final deterministic padding if necessary
    if len(selected) < k:
        rem = [i for i in range(N) if i not in selected]
        if rem:
            order = np.array(rem)[np.argsort(-indiv_vols[rem])]
            for i in order:
                if len(selected) >= k:
                    break
                selected.append(int(i))

    final_idx = np.array(selected[:k], dtype=int)
    subset = points[final_idx].copy()
    return subset


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
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

    # Monte-Carlo sample budget: tuned differently from original
    M = int(np.clip(1500 + 300 * min(k, 40) + 120 * D, 1000, 16000))

    rng = np.random.default_rng()
    samples = global_low + rng.random((M, D)) * global_span  # (M, D)

    # Compute coverage matrix: covers[i, m] = True if box i covers sample m
    ge_low = samples[:, None, :] >= (lows[None, :, :] - EPS)
    le_high = samples[:, None, :] <= (highs[None, :, :] + EPS)
    within = np.logical_and(ge_low, le_high).all(axis=2)  # shape (M, N)
    covers = within.T  # shape (N, M), boolean

    # Use optional sample weighting (uniform here, but kept as float for easy extension)
    sample_weights = np.ones(M, dtype=float)

    # Initial uncovered samples
    covered = np.zeros(M, dtype=bool)
    remaining = set(range(N))
    selected = []

    # Initial marginal weighted counts (float)
    # marg_counts[i] = sum_{m in samples covered by i} sample_weights[m]
    # use dot: covers (N,M) * sample_weights (M,) -> (N,)
    marg_counts = covers.astype(float).dot(sample_weights)

    # Normalize individual volumes
    max_vol = float(indiv_vols.max()) if indiv_vols.max() > 0 else 1.0
    indiv_vol_norm = indiv_vols / max_vol

    # Score function parameters (different/tuned from original)
    beta = 0.7  # weight of volumetric bias relative to sampled marginal gain
    alpha = 0.6  # exponent on normalized volume to moderate influence
    vol_bias = beta * (indiv_vol_norm ** alpha) * float(M)  # scale to sample-count units

    # Initial scores: marginal (approx) + volumetric bias
    scores = marg_counts + vol_bias

    # Build max-heap (as min-heap of negative values). Tie-breaker: larger indiv_vol preferred
    heap = []
    for i in range(N):
        if indiv_vols[i] <= EPS:
            continue
        # Push (neg_score, -indiv_vol, idx)
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
        actual_score = actual_marg + beta * (indiv_vol_norm[idx] ** alpha) * float(M)
        # If score changed, push updated
        if abs(actual_score + neg_score) > 1e-8:
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

    # Limited local improvement via weighted sampled-coverage based swaps
    if selected:
        sel_arr = np.array(selected, dtype=int)
        # cover_counts weighted per-sample (float): sum of covers by selected per sample * weight
        cover_counts = covers[sel_arr, :].astype(float).sum(axis=0)  # (M,)
        # count samples that are covered (weighted presence > 0)
        curr_covered_weight = float(np.dot((cover_counts > 0).astype(float), sample_weights))
        # Prepare candidate pool for swaps: top unselected by indiv_vol
        unselected = np.array([i for i in range(N) if i not in set(selected)], dtype=int)
        if unselected.size > 0:
            u_order = unselected[np.argsort(-indiv_vols[unselected])]
            top_unselected = u_order[:min(250, u_order.size)]
        else:
            top_unselected = np.array([], dtype=int)

        max_swaps = min(5 * k + 50, 500)
        swaps = 0
        improved = True
        while improved and swaps < max_swaps:
            improved = False
            swaps += 1
            # try for each selected element to find an improving swap
            for si_idx, s in enumerate(list(selected)):
                # compute cover_counts excluding s
                cover_counts_excl = cover_counts - covers[s].astype(float)
                # For each promising unselected candidate
                for cand in top_unselected:
                    if cand in selected:
                        continue
                    # new cover counts after swap
                    new_counts = cover_counts_excl + covers[cand].astype(float)
                    # weighted covered presence
                    new_covered_weight = float(np.dot((new_counts > 0).astype(float), sample_weights))
                    # accept swap if weighted coverage increases (strict)
                    if new_covered_weight > curr_covered_weight + 1e-9:
                        # accept swap
                        selected[si_idx] = int(cand)
                        # update structures
                        cover_counts = new_counts
                        curr_covered_weight = new_covered_weight
                        # refresh unselected/top_unselected sets
                        unselected = np.array([i for i in range(N) if i not in set(selected)], dtype=int)
                        if unselected.size > 0:
                            u_order = unselected[np.argsort(-indiv_vols[unselected])]
                            top_unselected = u_order[:min(250, u_order.size)]
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


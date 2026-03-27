import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np

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

    # Degenerate fallback
    if np.all(indiv_vols <= EPS):
        idxs = np.argsort(np.sum((reference_point - points) ** 2, axis=1))[:min(k, N)]
        return points[idxs].copy()

    # Global bounding box for possible Monte-Carlo ref evaluation
    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    global_span = np.maximum(global_high - global_low, 0.0)
    global_vol = float(np.prod(global_span)) if np.all(global_span > 0) else 0.0

    if global_vol <= EPS:
        idxs = np.argsort(-indiv_vols)[:min(k, N)]
        return points[idxs].copy()

    # Greedy selection driven by analytic pairwise overlaps (approximation)
    remaining = np.ones(N, dtype=bool)
    selected = []
    # sum of overlaps of each candidate with the currently selected set (approximate correction)
    sum_overlaps_with_selected = np.zeros(N, dtype=float)

    def pair_overlap_with_index(s_idx):
        # compute overlap volumes between box s_idx and all boxes (vectorized)
        # intersection extents = max(0, min(highs, highs[s]) - max(lows, lows[s]))
        hi = np.minimum(highs, highs[s_idx])
        lo = np.maximum(lows, lows[s_idx])
        inter = np.maximum(hi - lo, 0.0)
        # product across dims
        vols = np.prod(inter, axis=1)
        vols[s_idx] = 0.0
        return vols

    # Pre-prune candidates with zero individual volume
    remaining &= indiv_vols > EPS

    max_select = min(k, int(remaining.sum()))
    for _ in range(max_select):
        # approximate marginal = indiv_vols - sum_overlaps_with_selected
        approx_marginal = indiv_vols - sum_overlaps_with_selected
        approx_marginal[~remaining] = -np.inf  # exclude already selected or invalid
        # clamp small negatives to zero for stability
        approx_marginal = np.maximum(approx_marginal, 0.0)
        # pick best candidate
        best = int(np.argmax(approx_marginal))
        if approx_marginal[best] <= EPS:
            # nothing adds appreciable approximate marginal -> stop
            break
        # select it
        selected.append(best)
        remaining[best] = False
        # update overlap sums
        overlaps = pair_overlap_with_index(best)
        # For numerical stability, ensure no negative overlaps
        overlaps = np.maximum(overlaps, 0.0)
        sum_overlaps_with_selected += overlaps

    # If not enough selected, pad by largest individual volumes among remaining
    if len(selected) < k:
        rem_idx = np.where(remaining)[0]
        if rem_idx.size > 0:
            order = rem_idx[np.argsort(-indiv_vols[rem_idx])]
            for idx in order:
                if len(selected) >= k:
                    break
                selected.append(int(idx))
                remaining[int(idx)] = False

    selected = selected[:k]
    if len(selected) == 0:
        return np.zeros((0, D), dtype=float)

    # Monte-Carlo validation and limited local swap refinement using sampled coverage
    rng = np.random.default_rng()
    M = int(np.clip(2000 + 600 * min(k, 80) + 200 * D, 2000, 10000))
    samples = global_low + rng.random((M, D)) * global_span  # (M, D)

    # Precompute coverage matrix for all boxes on these samples (N, M) boolean
    ge_low = samples[:, None, :] >= (lows[None, :, :] - EPS)
    le_high = samples[:, None, :] <= (highs[None, :, :] + EPS)
    within = np.logical_and(ge_low, le_high).all(axis=2)  # shape (M, N)
    covers = within.T  # shape (N, M), boolean

    sel_arr = np.array(selected, dtype=int)
    cover_counts = covers[sel_arr, :].sum(axis=0).astype(int)  # per-sample counts
    curr_covered = int(np.count_nonzero(cover_counts > 0))
    # Candidate pool for swaps: top unselected by individual volume (fast heuristic)
    unselected_idx = np.array([i for i in range(N) if i not in set(selected)], dtype=int)
    if unselected_idx.size > 0:
        u_order = unselected_idx[np.argsort(-indiv_vols[unselected_idx])]
        top_unselected = u_order[:min(300, u_order.size)]
    else:
        top_unselected = np.array([], dtype=int)

    max_swaps = min(5 * k + 100, 600)
    swaps = 0
    improved = True
    sel_set = set(selected)
    while improved and swaps < max_swaps:
        improved = False
        swaps += 1
        # try each selected element for an improving swap
        for si_idx, s in enumerate(list(sel_arr)):
            # counts excluding s
            cover_excl = cover_counts - covers[s].astype(int)
            for cand in top_unselected:
                if cand in sel_set:
                    continue
                new_counts = cover_excl + covers[cand].astype(int)
                new_covered = int(np.count_nonzero(new_counts > 0))
                if new_covered > curr_covered:
                    # accept swap
                    sel_set.discard(int(s))
                    sel_set.add(int(cand))
                    sel_arr[si_idx] = int(cand)
                    cover_counts = new_counts
                    curr_covered = new_covered
                    # refresh top_unselected
                    unselected_idx = np.array([i for i in range(N) if i not in sel_set], dtype=int)
                    if unselected_idx.size > 0:
                        u_order = unselected_idx[np.argsort(-indiv_vols[unselected_idx])]
                        top_unselected = u_order[:min(300, u_order.size)]
                    else:
                        top_unselected = np.array([], dtype=int)
                    improved = True
                    break
            if improved:
                break

    # Final selected indices
    final_idx = np.array(list(sel_set), dtype=int)
    # If we might have fewer/more than k due to set operations, normalize to exactly k
    if final_idx.size > k:
        # pick k with best empirical contribution on samples
        # evaluate per-item marginal (samples)
        cov_mask = (covers[final_idx, :].sum(axis=0) > 0)
        # compute individual covered counts
        indiv_counts = covers[final_idx, :].sum(axis=1)
        order = np.argsort(-indiv_counts)
        final_idx = final_idx[order[:k]]
    elif final_idx.size < k:
        # pad by remaining best indiv_vol
        rem = np.array([i for i in range(N) if i not in set(final_idx)], dtype=int)
        if rem.size > 0:
            pad_order = rem[np.argsort(-indiv_vols[rem])]
            need = k - final_idx.size
            to_add = pad_order[:need]
            final_idx = np.concatenate([final_idx, to_add])

    final_idx = final_idx[:k]
    subset = points[final_idx].copy()
    return subset


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

    # Trivial degeneracy: all zero volumes -> fallback deterministic selection by closeness
    if np.all(indiv_vols <= EPS):
        idxs = np.argsort(np.sum((reference_point - points) ** 2, axis=1))[:min(k, N)]
        return points[idxs].copy()

    # Global bounding box for uniform sampling
    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    global_span = np.maximum(global_high - global_low, 0.0)
    global_vol = float(np.prod(global_span)) if np.all(global_span > 0) else 0.0

    if global_vol <= EPS:
        idxs = np.argsort(-indiv_vols)[:min(k, N)]
        return points[idxs].copy()

    # Number of Monte-Carlo samples: scale with k and D but bounded
    M = int(np.clip(2000 + 400 * min(k, 50) + 150 * D, 2000, 20000))

    rng = np.random.default_rng()
    samples = global_low + rng.random((M, D)) * global_span  # (M, D)

    # Compute coverage matrix: covers[i, m] = True if box i covers sample m
    # To save memory orientation: compute (M, N) then transpose to (N, M)
    # Condition: sample >= low and sample <= high (inclusive within EPS)
    ge_low = samples[:, None, :] >= (lows[None, :, :] - EPS)
    le_high = samples[:, None, :] <= (highs[None, :, :] + EPS)
    within = np.logical_and(ge_low, le_high).all(axis=2)  # shape (M, N)
    covers = within.T  # shape (N, M), boolean

    # Initial uncovered samples
    covered = np.zeros(M, dtype=bool)
    remaining = set(range(N))
    selected = []

    # Initial marginal counts (number of sample points a candidate covers)
    marg_counts = covers.sum(axis=1).astype(int)  # shape (N,)

    # Build max-heap (as min-heap of negative values). Tie-breaker: larger individual volume preferred.
    heap = []
    for i in range(N):
        if indiv_vols[i] <= EPS:
            continue
        heapq.heappush(heap, (-int(marg_counts[i]), -float(indiv_vols[i]), int(i)))

    # Lazy-greedy selection based on sampled marginal coverage counts
    while len(selected) < min(k, N) and heap:
        neg_count, neg_vol, idx = heapq.heappop(heap)
        idx = int(idx)
        if idx not in remaining:
            continue
        # Recompute actual marginal with current 'covered' mask
        # marginal = number of samples that candidate covers and are not yet covered
        if covers[idx].any():
            actual = int(np.count_nonzero(covers[idx] & (~covered)))
        else:
            actual = 0
        if actual == 0:
            remaining.discard(idx)
            continue
        if actual != -neg_count:
            # push updated count back and continue lazy behavior
            heapq.heappush(heap, (-actual, neg_vol, idx))
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

    # Limited local improvement via sampled-coverage based swaps
    # Precompute cover_counts for currently selected set (per-sample counts)
    if selected:
        sel_arr = np.array(selected, dtype=int)
        cover_counts = covers[sel_arr, :].sum(axis=0).astype(int)  # (M,)
        curr_covered_count = int(np.count_nonzero(cover_counts > 0))
        # Prepare candidate pool for swaps: top unselected by indiv_vol
        unselected = np.array([i for i in range(N) if i not in set(selected)], dtype=int)
        if unselected.size > 0:
            u_order = unselected[np.argsort(-indiv_vols[unselected])]
            top_unselected = u_order[:min(200, u_order.size)]
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
                cover_counts_excl = cover_counts - covers[s].astype(int)
                # For each promising unselected candidate
                for cand in top_unselected:
                    if cand in selected:
                        continue
                    # new cover counts after swap
                    new_counts = cover_counts_excl + covers[cand].astype(int)
                    new_covered_count = int(np.count_nonzero(new_counts > 0))
                    if new_covered_count > curr_covered_count:
                        # accept swap
                        selected[si_idx] = int(cand)
                        # update structures
                        cover_counts = new_counts
                        curr_covered_count = new_covered_count
                        # refresh unselected/top_unselected sets
                        unselected = np.array([i for i in range(N) if i not in set(selected)], dtype=int)
                        if unselected.size > 0:
                            u_order = unselected[np.argsort(-indiv_vols[unselected])]
                            top_unselected = u_order[:min(200, u_order.size)]
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


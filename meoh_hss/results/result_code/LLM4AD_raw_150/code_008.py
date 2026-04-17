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

    # Adaptive Monte-Carlo sample size: scale with k and D but bounded
    M = int(np.clip(1000 + 600 * min(k, 80) + 200 * D, 1000, 30000))

    rng = np.random.default_rng()
    samples = global_low + rng.random((M, D)) * global_span  # (M, D)

    # Compute coverage matrix: covers[i, m] = True if box i covers sample m
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

    # Estimate expected sample-count per candidate from volumes (principled prior)
    expected_counts = (indiv_vols / global_vol) * M  # float array, may be small
    # Combine sampled marginal with expected counts as composite score
    # lambda_tradeoff controls how much we trust volume-derived expectation vs sampled gain
    lambda_tradeoff = 0.6 if M < 5000 else 0.4
    composite_init = marg_counts + lambda_tradeoff * expected_counts

    # Build max-heap (as min-heap of negative values). Tie-breaker: larger expected_counts preferred.
    heap = []
    for i in range(N):
        if indiv_vols[i] <= EPS:
            continue
        # store (negative composite score as int/float, negative expected_count, index)
        heapq.heappush(heap, (-float(composite_init[i]), -float(expected_counts[i]), int(i)))

    # Lazy-greedy selection based on composite score where the sampled marginal part is updated lazily
    while len(selected) < min(k, N) and heap:
        neg_score, neg_exp, idx = heapq.heappop(heap)
        idx = int(idx)
        if idx not in remaining:
            continue
        # Recompute actual marginal with current 'covered' mask (number of newly covered samples)
        if covers[idx].any():
            actual_new = int(np.count_nonzero(covers[idx] & (~covered)))
        else:
            actual_new = 0
        actual_composite = actual_new + lambda_tradeoff * expected_counts[idx]
        # If candidate does not contribute new coverage, discard
        if actual_new == 0:
            remaining.discard(idx)
            continue
        # If composite changed (due to actual_new differing from earlier marg_counts), push back updated
        if not np.isclose(-neg_score, actual_composite):
            heapq.heappush(heap, (-float(actual_composite), neg_exp, idx))
            continue
        # accept idx
        selected.append(idx)
        remaining.discard(idx)
        # update covered samples
        covered |= covers[idx]

    # If still fewer than k selected, fill deterministically by expected_counts among remaining
    if len(selected) < k:
        rem = np.array([i for i in range(N) if i in remaining], dtype=int)
        if rem.size > 0:
            order = rem[np.argsort(-(expected_counts[rem] + 1e-12))]
            for i in order:
                if len(selected) >= k:
                    break
                selected.append(int(i))
                remaining.discard(int(i))

    selected = selected[:k]

    # Limited local improvement via sampled-coverage based swaps (attempt to increase sampled covered count)
    if selected:
        sel_set = set(selected)
        sel_arr = np.array(selected, dtype=int)
        cover_counts = covers[sel_arr, :].sum(axis=0).astype(int)  # (M,)
        curr_covered_count = int(np.count_nonzero(cover_counts > 0))
        # Candidate pool for swaps: top unselected by expected_counts
        unselected = np.array([i for i in range(N) if i not in sel_set], dtype=int)
        if unselected.size > 0:
            u_order = unselected[np.argsort(-(expected_counts[unselected] + 1e-12))]
            top_unselected = u_order[:min(300, u_order.size)]
        else:
            top_unselected = np.array([], dtype=int)

        max_swaps = min(5 * k + 50, 600)
        swaps = 0
        improved = True
        while improved and swaps < max_swaps:
            improved = False
            swaps += 1
            # try each selected element to find an improving swap
            for si_idx, s in enumerate(list(selected)):
                # compute cover_counts excluding s
                cover_counts_excl = cover_counts - covers[s].astype(int)
                for cand in top_unselected:
                    if cand in sel_set:
                        continue
                    new_counts = cover_counts_excl + covers[cand].astype(int)
                    new_covered_count = int(np.count_nonzero(new_counts > 0))
                    # Accept swap if strictly improves sampled coverage, or equal but better expected_counts
                    if new_covered_count > curr_covered_count or (
                        new_covered_count == curr_covered_count and expected_counts[cand] > expected_counts[s] * 1.001
                    ):
                        # accept swap
                        sel_set.discard(s)
                        sel_set.add(int(cand))
                        selected[si_idx] = int(cand)
                        # update structures
                        cover_counts = new_counts
                        curr_covered_count = new_covered_count
                        unselected = np.array([i for i in range(N) if i not in sel_set], dtype=int)
                        if unselected.size > 0:
                            u_order = unselected[np.argsort(-(expected_counts[unselected] + 1e-12))]
                            top_unselected = u_order[:min(300, u_order.size)]
                        else:
                            top_unselected = np.array([], dtype=int)
                        improved = True
                        break
                if improved:
                    break

    # Final deterministic padding if necessary (by expected_counts)
    if len(selected) < k:
        rem = [i for i in range(N) if i not in selected]
        if rem:
            order = np.array(rem)[np.argsort(-(expected_counts[rem] + 1e-12))]
            for i in order:
                if len(selected) >= k:
                    break
                selected.append(int(i))

    final_idx = np.array(selected[:k], dtype=int)
    subset = points[final_idx].copy()
    return subset


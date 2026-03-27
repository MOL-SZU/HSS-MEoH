import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape

    if k <= 0:
        return np.empty((0, d))
    k = min(k, n)

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def hv_of(arr_like):
        if arr_like is None:
            return 0.0
        arr = np.asarray(arr_like, dtype=float)
        if arr.size == 0:
            return 0.0
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        hv = pg.hypervolume(arr)
        return float(hv.compute(reference_point))

    # quick single-point hypervolumes (box sizes) as upper bounds and for prefiltering
    hv_single = np.prod(np.maximum(0.0, reference_point - points), axis=1)

    # If all zero volumes, return first k deterministically
    if np.all(hv_single <= 0):
        return points[:k].copy()

    # Prefilter top candidates by hv_single to limit pool size
    pool_size = min(n, max(4 * k, 100))  # balance runtime and solution quality
    # ensure pool_size at least k
    pool_size = max(pool_size, k)
    top_idxs = np.argsort(-hv_single)[:pool_size].tolist()

    # Lazy greedy selection using hv marginals to reduce recomputations
    selected_indices = []
    selected_mask = np.zeros(n, dtype=bool)
    selected_points_list = []
    curr_hv = 0.0

    # Initialize max-heap with negative estimates (use hv_single as optimistic estimate)
    heap = [(-hv_single[i], i) for i in top_idxs]
    heapq.heapify(heap)

    tol = 1e-12
    # Use lazy greedy: pop top, recompute true marginal; if still best, select; else push updated and continue
    while len(selected_indices) < k and heap:
        est_neg, idx = heapq.heappop(heap)
        est = -est_neg
        if selected_points_list:
            # compute true marginal gain
            candidate_set = np.vstack([selected_points_list, points[idx]])
            hv_with = hv_of(candidate_set)
        else:
            hv_with = hv_single[idx]
        true_gain = hv_with - curr_hv
        # If heap empty or true_gain is no less than next estimate, accept
        if not heap or true_gain >= -heap[0][0] - tol:
            # accept this idx
            if true_gain < -1e-15:
                # even if negative (rare due to numerical), we still accept to fill k
                pass
            selected_indices.append(idx)
            selected_mask[idx] = True
            selected_points_list.append(points[idx].copy())
            curr_hv = hv_with
        else:
            # push back with updated true gain (as estimate)
            heapq.heappush(heap, (-max(true_gain, 0.0), idx))

    # If we didn't fill k due to empty heap (shouldn't), fill from remaining best singles
    if len(selected_indices) < k:
        remaining = [i for i in np.argsort(-hv_single) if not selected_mask[i]]
        for i in remaining:
            selected_indices.append(i)
            selected_mask[i] = True
            selected_points_list.append(points[i].copy())
            if len(selected_indices) == k:
                break
        curr_hv = hv_of(np.vstack(selected_points_list)) if selected_points_list else 0.0

    selected_points = np.array(selected_points_list, dtype=float)

    # Constrained best-improvement pairwise-swap local search:
    # Only consider swaps with the top 'swap_pool' unselected candidates by hv_single to limit runtime
    max_swap_iters = 100
    swap_pool = min(n, max(3 * k, 200))
    swap_iter = 0
    improved = True
    while improved and swap_iter < max_swap_iters:
        swap_iter += 1
        improved = False
        best_impr = 0.0
        best_swap = None  # (pos_in_selected, unselected_idx, hv_after)
        # gather most promising unselected candidates
        unselected_candidates = [i for i in np.argsort(-hv_single) if not selected_mask[i]]
        if not unselected_candidates:
            break
        unselected_candidates = unselected_candidates[:swap_pool]
        # try all swaps between selected positions and these candidates
        for pos in range(len(selected_indices)):
            # build base copy once per pos
            base_set = selected_points.copy()
            for cand in unselected_candidates:
                base_set[pos] = points[cand]
                hv_after = hv_of(base_set)
                improvement = hv_after - curr_hv
                if improvement > best_impr + tol:
                    best_impr = improvement
                    best_swap = (pos, cand, hv_after)
        if best_swap is not None and best_impr > tol:
            pos, cand_idx, hv_after = best_swap
            old_idx = selected_indices[pos]
            selected_mask[old_idx] = False
            selected_indices[pos] = cand_idx
            selected_mask[cand_idx] = True
            selected_points[pos] = points[cand_idx].copy()
            curr_hv = hv_after
            improved = True

    # Ensure exactly k points in returned array (order not important)
    final_selected = np.array(selected_points, dtype=float)
    if final_selected.shape[0] > k:
        final_selected = final_selected[:k]
    elif final_selected.shape[0] < k:
        # pad with best remaining by hv_single
        remaining_all = [i for i in np.argsort(-hv_single) if not selected_mask[i]]
        need = k - final_selected.shape[0]
        if remaining_all and need > 0:
            extras = np.array([points[i] for i in remaining_all[:need]], dtype=float)
            if final_selected.size == 0:
                final_selected = extras
            else:
                final_selected = np.vstack([final_selected, extras])

    return final_selected.copy()


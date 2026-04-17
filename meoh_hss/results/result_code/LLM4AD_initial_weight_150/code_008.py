import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    import random
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = pts.shape

    if k <= 0:
        return np.empty((0, d), dtype=float)
    if k >= n:
        return pts.copy()

    # prepare reference point
    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    # ensure reference is slightly larger than minima to avoid negative volumes
    eps = 1e-12
    min_coords = np.min(pts, axis=0)
    reference_point = np.maximum(reference_point, min_coords + eps)

    # fast box-volume upper bound for single-point hypervolume (very cheap)
    diffs = reference_point[None, :] - pts
    diffs_clipped = np.maximum(diffs, 0.0)
    single_box_hv = np.prod(diffs_clipped + eps, axis=1)  # small eps to avoid exact zeros

    # choose compact pool: keep the top candidates by single_box_hv
    # pool size scales with k but capped for speed
    pool_size = int(min(n, max(5 * k, 200)))
    # but don't exceed n
    pool_size = max(pool_size, k)
    order = np.argsort(-single_box_hv)
    pool_indices = list(order[:pool_size])

    # hypervolume wrapper
    def hv_of(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return 0.0
        arr2 = np.atleast_2d(arr)
        hv = pg.hypervolume(arr2)
        return float(hv.compute(reference_point))

    # Lazy greedy (CELF) on reduced pool
    # initialize heap with single-box hv as optimistic gains (fast)
    heap = [(-single_box_hv[idx], idx, 0) for idx in pool_indices]
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_selected_size = 0
    curr_hv = 0.0
    iter_id = 0
    tol = 1e-12

    while len(selected) < k and heap:
        neg_gain, idx, stamp = heapq.heappop(heap)
        cached_gain = -neg_gain
        # skip if already selected
        if idx in selected_set:
            continue
        if stamp == current_selected_size:
            # accept this candidate
            selected.append(idx)
            selected_set.add(idx)
            # update current hv exactly
            sel_pts = pts[selected]
            curr_hv = hv_of(sel_pts)
            current_selected_size += 1
            iter_id += 1
            continue
        # else recompute true marginal gain
        if current_selected_size == 0:
            true_gain = single_box_hv[idx]
        else:
            # compute hv of selected + candidate (stacking)
            cand_stack = np.vstack([pts[selected], pts[idx]])
            after = hv_of(cand_stack)
            true_gain = after - curr_hv
            # numerical floor
            if true_gain < 0 and true_gain > -1e-14:
                true_gain = 0.0
        # push back with updated stamp
        heapq.heappush(heap, (-true_gain, idx, current_selected_size))

    # if not enough (shouldn't happen) fill with best remaining global candidates
    if len(selected) < k:
        remaining = [i for i in order if i not in selected_set]
        for i in remaining:
            selected.append(int(i))
            selected_set.add(int(i))
            if len(selected) == k:
                break
    selected = selected[:k]
    selected_points = pts[np.array(selected, dtype=int)].copy()
    curr_hv = hv_of(selected_points) if selected_points.size else 0.0

    # lightweight randomized swap refinement inside pool to improve hv with limited budget
    max_swaps = 60
    swaps = 0
    unselected_in_pool = [i for i in pool_indices if i not in selected_set]
    if unselected_in_pool:
        while swaps < max_swaps and len(unselected_in_pool) > 0:
            swaps += 1
            improved = False
            # sample small sets to keep runtime low
            sample_un = random.sample(unselected_in_pool, min(30, len(unselected_in_pool)))
            sample_pos = random.sample(range(len(selected_points)), min(6, len(selected_points)))
            best_impr = 0.0
            best_swap = None
            for pos in sample_pos:
                for cand in sample_un:
                    candidate = selected_points.copy()
                    candidate[pos] = pts[cand]
                    hv_cand = hv_of(candidate)
                    gain = hv_cand - curr_hv
                    if gain > best_impr + 1e-12:
                        best_impr = gain
                        best_swap = (pos, cand, hv_cand)
            if best_swap is not None:
                pos, cand_idx, hv_after = best_swap
                old_idx = selected[pos]
                selected_set.remove(old_idx)
                selected[pos] = cand_idx
                selected_set.add(cand_idx)
                selected_points[pos] = pts[cand_idx].copy()
                curr_hv = hv_after
                unselected_in_pool = [i for i in pool_indices if i not in selected_set]
                improved = True
            if not improved:
                break

    # final safety: ensure exactly k returned
    if len(selected) < k:
        # fill from global best not already selected
        for i in order:
            if i not in selected_set:
                selected.append(int(i))
                selected_set.add(int(i))
                selected_points = np.vstack([selected_points, pts[i].copy()]) if selected_points.size else pts[i:i+1].copy()
                if len(selected) == k:
                    break

    selected_points = np.asarray(selected_points, dtype=float)
    if selected_points.shape[0] > k:
        selected_points = selected_points[:k]
    return selected_points.copy()


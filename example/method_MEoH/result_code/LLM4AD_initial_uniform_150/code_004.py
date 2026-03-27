import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for exact hypervolume computation") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k <= 0:
        return np.empty((0, D), dtype=pts.dtype)
    if N == 0:
        return np.empty((0, D), dtype=pts.dtype)
    if k >= N:
        return pts.copy()

    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Helper: compute hypervolume of given indices (list or array)
    def hv_of_indices(idxs):
        if len(idxs) == 0:
            return 0.0
        data = pts[np.array(idxs, dtype=int), :]
        hv = pg.hypervolume(data)
        return float(hv.compute(reference_point))

    # Precompute individual hypervolumes
    individual_hv = np.empty(N, dtype=float)
    for i in range(N):
        individual_hv[i] = hv_of_indices([i])
    max_indiv = float(np.max(individual_hv)) if N > 0 else 0.0
    if max_indiv <= 0:
        norm_indiv = np.zeros_like(individual_hv)
    else:
        norm_indiv = individual_hv / max_indiv

    # Precompute a normalization for diversity (use diagonal of bounding box)
    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    max_dist = np.linalg.norm(bbox_max - bbox_min)
    if max_dist <= 0:
        max_dist = 1.0

    # Parameters for the modified score function (different settings)
    alpha = 0.6   # weight of normalized individual hv boost
    beta = 0.9    # weight of diversity boost (min distance to selected set)
    tol = 1e-12

    # Lazy greedy with modified score:
    # Each heap entry: (-score, idx, version)
    current_selected = []
    selected_set = set()
    hv_current = 0.0

    heap = []
    version = 0

    # Initial scores: marginal_gain = individual_hv (since hv_current = 0)
    for i in range(N):
        marginal = individual_hv[i]
        # Modified score multiplies marginal by (1 + alpha * norm_indiv)
        score = marginal * (1.0 + alpha * norm_indiv[i])
        # No diversity at start (no selected points), push to heap
        heap.append((-score, int(i), version))
    heapq.heapify(heap)

    def diversity_boost_for(i):
        # If no selected points, no diversity boost
        if not current_selected:
            return 1.0
        # compute min Euclidean distance from point i to any selected point
        sel_pts = pts[np.array(current_selected, dtype=int), :]
        dists = np.linalg.norm(sel_pts - pts[int(i)], axis=1)
        min_dist = float(np.min(dists))
        return 1.0 + beta * (min_dist / max_dist)

    # Greedily select k points using modified score, with lazy updates
    while len(current_selected) < k and heap:
        neg_score, idx, entry_version = heapq.heappop(heap)
        if idx in selected_set:
            continue
        if entry_version != version:
            # recompute true marginal_gain and modified score with diversity
            with_idx = hv_of_indices(current_selected + [int(idx)])
            marginal = with_idx - hv_current
            if marginal < 0:
                marginal = 0.0
            score = marginal * (1.0 + alpha * norm_indiv[int(idx)]) * diversity_boost_for(int(idx))
            heapq.heappush(heap, (-score, int(idx), version))
            continue
        # up-to-date entry: accept it
        # Accept even if score is zero to fill k
        current_selected.append(int(idx))
        selected_set.add(int(idx))
        hv_current = hv_of_indices(current_selected)
        version += 1

    # If not enough selected (heap exhausted), fill arbitrarily
    if len(current_selected) < k:
        for i in range(N):
            if i not in selected_set:
                current_selected.append(int(i))
                selected_set.add(int(i))
                if len(current_selected) >= k:
                    break
        hv_current = hv_of_indices(current_selected)

    # Local 1-swap improvement: try replacing one selected point with an unselected one if it improves hv
    max_swap_iters = 60
    swap_iter = 0
    improved = True
    candidate_limit = min(300, N)

    # Prepare unselected candidates ordered by normalized individual hv (descending)
    unselected_all_sorted = sorted([i for i in range(N) if i not in selected_set],
                                   key=lambda x: -norm_indiv[x])[:candidate_limit]

    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        outer_break = False
        # Re-evaluate candidate list each iter to allow newly unselected items to be considered
        unselected = [i for i in range(N) if i not in selected_set]
        if not unselected:
            break
        unselected_sorted = sorted(unselected, key=lambda x: -norm_indiv[x])[:candidate_limit]
        for s in list(current_selected):
            if outer_break:
                break
            base_set = [idx for idx in current_selected if idx != s]
            # Quick screening: compute a cheap upper bound for promising u's by individual hv
            for u in unselected_sorted:
                if u == s:
                    continue
                swapped_hv = hv_of_indices(base_set + [u])
                if swapped_hv > hv_current + tol:
                    # perform swap
                    for pos, val in enumerate(current_selected):
                        if val == s:
                            current_selected[pos] = int(u)
                            break
                    selected_set.remove(s)
                    selected_set.add(int(u))
                    hv_current = swapped_hv
                    improved = True
                    outer_break = True
                    break
    # Finalize: return selected points (in selection order)
    selected_indices = current_selected[:k]
    subset = pts[np.array(selected_indices, dtype=int), :].copy()
    return subset


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
    eps = 1e-12
    if max_indiv <= 0:
        norm_indiv = np.zeros_like(individual_hv)
    else:
        # soften the influence via a power transform to reduce domination of very large indiv hv
        norm_indiv = (individual_hv / (max_indiv + eps)) ** 0.8

    # Diversity normalization: use median distance to centroid for robust scaling
    centroid = np.mean(pts, axis=0)
    d_to_centroid = np.linalg.norm(pts - centroid, axis=1)
    median_dist = float(np.median(d_to_centroid))
    if median_dist <= 0:
        median_dist = 1.0
    # fallback bounding diagonal (for extreme degeneracy)
    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    if diag <= 0:
        diag = 1.0

    # New score parameters (different from original)
    alpha = 0.35   # softer individual hv boost
    beta = 1.5     # stronger diversity emphasis
    div_exp = 1.2  # exponent for diversity scaling
    tol = 1e-12

    # Lazy greedy with modified score:
    current_selected = []
    selected_set = set()
    hv_current = 0.0

    heap = []
    version = 0

    # Initial marginal gains are individual_hv
    for i in range(N):
        marginal = individual_hv[i]
        indiv_boost = 1.0 + alpha * norm_indiv[i]
        # no diversity boost initially
        score = marginal * indiv_boost
        heap.append((-score, int(i), version))
    heapq.heapify(heap)

    def diversity_boost_for(i):
        if not current_selected:
            return 1.0
        sel_pts = pts[np.array(current_selected, dtype=int), :]
        dists = np.linalg.norm(sel_pts - pts[int(i)], axis=1)
        min_dist = float(np.min(dists))
        # scale by median_dist (robust) but cap by diag to avoid extreme values
        scaled = (min_dist / (median_dist + eps)) ** div_exp
        # make sure scaling not exploding
        scaled = float(np.minimum(scaled, (min_dist / (diag + eps)) ** (div_exp / 2) + scaled))
        return 1.0 + beta * scaled

    # Greedy selection loop with lazy updates
    while len(current_selected) < k and heap:
        neg_score, idx, entry_version = heapq.heappop(heap)
        if idx in selected_set:
            continue
        if entry_version != version:
            # recompute true marginal gain and push updated
            with_idx = hv_of_indices(current_selected + [int(idx)])
            marginal = with_idx - hv_current
            if marginal < 0:
                marginal = 0.0
            indiv_boost = 1.0 + alpha * norm_indiv[int(idx)]
            score = marginal * indiv_boost * diversity_boost_for(int(idx))
            heapq.heappush(heap, (-score, int(idx), version))
            continue
        # accept this item
        current_selected.append(int(idx))
        selected_set.add(int(idx))
        hv_current = hv_of_indices(current_selected)
        version += 1

    # Fill arbitrarily if not enough
    if len(current_selected) < k:
        for i in range(N):
            if i not in selected_set:
                current_selected.append(int(i))
                selected_set.add(int(i))
                if len(current_selected) >= k:
                    break
        hv_current = hv_of_indices(current_selected)

    # Local 1-swap improvement with a slightly larger budget
    max_swap_iters = 80
    swap_iter = 0
    improved = True
    candidate_limit = min(200, N)

    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        outer_break = False
        unselected = [i for i in range(N) if i not in selected_set]
        if not unselected:
            break
        # prioritize by softened norm_indiv
        unselected_sorted = sorted(unselected, key=lambda x: -norm_indiv[x])[:candidate_limit]
        for s in list(current_selected):
            if outer_break:
                break
            base_set = [idx for idx in current_selected if idx != s]
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

    selected_indices = current_selected[:k]
    subset = pts[np.array(selected_indices, dtype=int), :].copy()
    return subset


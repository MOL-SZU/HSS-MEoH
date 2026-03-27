import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    import random
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape
    if k <= 0:
        return np.empty((0, d))
    k = min(k, n)

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def hv_of(arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0.0
        arr2 = np.atleast_2d(arr)
        hv = pg.hypervolume(arr2)
        return float(hv.compute(reference_point))

    # Precompute normalization scale
    bbox_diag = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    if bbox_diag <= 0:
        bbox_diag = 1.0
    eps = 1e-12

    # Individual hypervolumes (one-point HV)
    indiv_hv = np.zeros(n, dtype=float)
    for i in range(n):
        indiv_hv[i] = hv_of(points[i])

    # CELF-like priority queue entries: (-score, idx, last_iter, cached_marginal_hv)
    heap = []
    # initial alpha baseline (lower to allow diversity early)
    alpha_init = 0.4
    for i in range(n):
        diversity_score = 0.0
        score = alpha_init * indiv_hv[i] + (1.0 - alpha_init) * diversity_score
        heap.append((-score, i, -1, float(indiv_hv[i])))
    heapq.heapify(heap)

    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []
    selected_points = []
    curr_hv = 0.0
    iter_id = 0
    tol = 1e-12

    while len(selected_indices) < k and heap:
        neg_score, idx, last_iter, cached_marginal = heapq.heappop(heap)
        cached_score = -neg_score
        # If cached score was computed this iter, accept
        if last_iter == iter_id:
            selected_indices.append(idx)
            selected_mask[idx] = True
            selected_points.append(points[idx].copy())
            # recompute exact current hv for numerical safety
            curr_hv = hv_of(np.vstack(selected_points))
            iter_id += 1
            continue

        # Recompute true marginal hv
        if len(selected_points) == 0:
            true_marginal = hv_of(points[idx:idx+1])
        else:
            cand_set = np.vstack([np.array(selected_points), points[idx]])
            true_marginal = hv_of(cand_set) - curr_hv
            if true_marginal < 0 and true_marginal > -1e-12:
                true_marginal = 0.0

        # Diversity: use minimum distance to selected set normalized by bbox_diag
        if len(selected_points) == 0:
            diversity_score = 1.0
        else:
            sel_arr = np.vstack(selected_points)
            dists = np.linalg.norm(sel_arr - points[idx], axis=1)
            min_dist = float(np.min(dists))
            diversity_score = float(min_dist / (bbox_diag + eps))

        # Adaptive alpha: start modest (favor diversity a bit), then progressively favor HV more
        # alpha goes from ~0.4 up to ~0.95 as more points are selected
        frac = len(selected_points) / max(1, k)
        alpha = float(np.clip(0.4 + 0.6 * frac, 0.2, 0.95))

        # small stochastic tie-breaker to avoid deterministic locks
        noise = 1e-9 * random.random()

        new_score = alpha * float(true_marginal) + (1.0 - alpha) * diversity_score + noise

        # push back with updated cached marginal computed in this iteration
        heapq.heappush(heap, (-new_score, idx, iter_id, float(true_marginal)))
        # continue loop to pop again

    # If not enough selected (heap exhausted), fill greedily by remaining indiv_hv
    if len(selected_indices) < k:
        for i in range(n):
            if not selected_mask[i]:
                selected_indices.append(i)
                selected_mask[i] = True
                selected_points.append(points[i].copy())
                if len(selected_indices) == k:
                    break
        curr_hv = hv_of(np.vstack(selected_points)) if len(selected_points) > 0 else 0.0

    selected_points = np.array(selected_points)

    # Bounded randomized swap refinement (greedy local search)
    max_swap_iters = 300
    if n - k > 0 and k > 0:
        swap_iter = 0
        improved = True
        while improved and swap_iter < max_swap_iters:
            swap_iter += 1
            improved = False
            unselected_pool = [i for i in range(n) if not selected_mask[i]]
            if len(unselected_pool) == 0:
                break
            # sample pools
            sample_unselected = random.sample(unselected_pool, min(40, len(unselected_pool)))
            sample_selected_pos = random.sample(range(k), min(8, k))
            best_impr = 0.0
            best_swap = None
            for pos in sample_selected_pos:
                for cand in sample_unselected:
                    candidate_set = selected_points.copy()
                    candidate_set[pos] = points[cand]
                    hv_cand = hv_of(candidate_set)
                    improvement = hv_cand - curr_hv
                    if improvement > best_impr + tol:
                        best_impr = improvement
                        best_swap = (pos, cand, hv_cand)
            if best_swap is not None:
                pos, cand_idx, hv_after = best_swap
                old_idx = selected_indices[pos]
                selected_mask[old_idx] = False
                selected_indices[pos] = cand_idx
                selected_mask[cand_idx] = True
                selected_points[pos] = points[cand_idx].copy()
                curr_hv = hv_after
                improved = True

    return np.array(selected_points)


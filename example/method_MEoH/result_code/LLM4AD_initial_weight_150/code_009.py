import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    import random
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    random.seed(42)

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape

    if k <= 0 or n == 0:
        return np.empty((0, d))
    k = min(k, n)

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def hv_of(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return 0.0
        arr2 = np.atleast_2d(arr)
        hv = pg.hypervolume(arr2)
        return float(hv.compute(reference_point))

    # compute individual one-point hypervolumes as rough quality measure
    indiv_hv = np.empty(n, dtype=float)
    for i in range(n):
        indiv_hv[i] = hv_of(points[i:i+1])

    # Pareto front prefilter (minimization orientation assumed, as in previous codes)
    # A dominates B if all coords <= and any <.
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        pi = points[i]
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            pj = points[j]
            # if j dominates i, mark i non-pareto
            if np.all(pj <= pi) and np.any(pj < pi):
                is_pareto[i] = False
                break
    pareto_idx = np.nonzero(is_pareto)[0].tolist()
    nonpareto_idx = [i for i in range(n) if not is_pareto[i]]

    # Candidate pool: prefer Pareto front sorted by indiv_hv desc, but ensure pool >= k
    max_pool = max(100, min(n, int(10 * k)))  # keep modest pool relative to k and n
    pareto_sorted = sorted(pareto_idx, key=lambda i: -indiv_hv[i])
    pool_idx = pareto_sorted[:max_pool]
    # if pool too small, add best non-pareto by indiv_hv
    if len(pool_idx) < max_pool:
        extra = sorted(nonpareto_idx, key=lambda i: -indiv_hv[i])
        needed = max_pool - len(pool_idx)
        pool_idx.extend(extra[:needed])
    pool_set = set(pool_idx)

    # normalization for diversity (bounding-box diagonal)
    max_pt = np.max(points, axis=0)
    min_pt = np.min(points, axis=0)
    diag = np.linalg.norm(max_pt - min_pt)
    if diag <= 0:
        diag = 1.0

    # CELF-style lazy greedy heap entries: (-score, idx, last_iter, cached_marginal)
    heap = []
    alpha_init = 0.35  # initial weight favoring marginal hv less to encourage diversity early
    for i in pool_idx:
        diversity = 1.0  # before any selection
        score = alpha_init * indiv_hv[i] + (1.0 - alpha_init) * diversity
        heap.append((-score, i, -1, float(indiv_hv[i])))
    heapq.heapify(heap)

    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []
    selected_points = []
    curr_hv = 0.0
    iter_id = 0
    tol = 1e-12

    # Lazy greedy selection using candidate pool
    while len(selected_indices) < k and heap:
        neg_score, idx, last_iter, cached_marginal = heapq.heappop(heap)
        score_cached = -neg_score
        # accept if cached score computed this iter
        if last_iter == iter_id:
            selected_indices.append(idx)
            selected_mask[idx] = True
            selected_points.append(points[idx].copy())
            curr_hv = hv_of(np.vstack(selected_points))
            iter_id += 1
            continue

        # recompute true marginal hv for this candidate
        if len(selected_points) == 0:
            true_marginal = hv_of(points[idx:idx+1])
        else:
            cand_set = np.vstack([np.array(selected_points), points[idx]])
            true_marginal = hv_of(cand_set) - curr_hv
            if true_marginal < 0 and true_marginal > -1e-12:
                true_marginal = 0.0

        # diversity: normalized min distance to selected set
        if len(selected_points) == 0:
            diversity = 1.0
        else:
            sel_arr = np.vstack(selected_points)
            dists = np.linalg.norm(sel_arr - points[idx], axis=1)
            min_dist = float(np.min(dists))
            diversity = float(min_dist / diag)

        # adaptive alpha increases as we select more points
        frac = len(selected_points) / max(1, k)
        alpha = float(np.clip(alpha_init + 0.6 * frac, 0.2, 0.95))

        # tiny noise tie-breaker
        noise = 1e-12 * (random.random() - 0.5)

        new_score = alpha * float(true_marginal) + (1.0 - alpha) * diversity + noise

        # push back with updated cached marginal and mark computed at this iter (last_iter=iter_id)
        heapq.heappush(heap, (-new_score, idx, iter_id, float(true_marginal)))
        # continue to pop next

    # if not enough selected (heap exhausted), fill by best remaining indiv_hv (from all points)
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if not selected_mask[i]]
        remaining_sorted = sorted(remaining, key=lambda i: -indiv_hv[i])
        for i in remaining_sorted:
            selected_indices.append(i)
            selected_mask[i] = True
            selected_points.append(points[i].copy())
            if len(selected_indices) == k:
                break
        if len(selected_points) > 0:
            curr_hv = hv_of(np.vstack(selected_points))

    selected_points = np.array(selected_points, dtype=float)

    # Bounded best-improvement swap refinement restricted to promising unselected candidates
    max_swap_iters = 200
    swap_iter = 0
    improved = True
    # build list of top unselected candidates by indiv_hv (to keep runtime limited)
    def top_unselected(limit=200):
        unselected = [i for i in range(n) if not selected_mask[i]]
        unselected_sorted = sorted(unselected, key=lambda i: -indiv_hv[i])
        return unselected_sorted[:min(limit, len(unselected_sorted))]

    while improved and swap_iter < max_swap_iters:
        swap_iter += 1
        improved = False
        best_impr = 0.0
        best_swap = None  # (pos, new_idx, hv_after)
        unselected_pool = top_unselected(limit=max(200, 10 * k))
        if len(unselected_pool) == 0:
            break
        # try swaps for every selected position against pool (deterministic best-improvement)
        for pos in range(len(selected_indices)):
            base_set = selected_points.copy()
            for cand in unselected_pool:
                base_set[pos] = points[cand]
                hv_after = hv_of(base_set)
                improvement = hv_after - curr_hv
                if improvement > best_impr + tol:
                    best_impr = improvement
                    best_swap = (pos, cand, hv_after)
            # continue scanning to find best overall
        if best_swap is not None:
            pos, cand_idx, hv_after = best_swap
            old_idx = selected_indices[pos]
            selected_mask[old_idx] = False
            selected_indices[pos] = cand_idx
            selected_mask[cand_idx] = True
            selected_points[pos] = points[cand_idx].copy()
            curr_hv = hv_after
            improved = True

    # if for some reason selected_points < k (edge), pad with best indiv_hv remaining
    if selected_points.shape[0] < k:
        remaining = [i for i in range(n) if not selected_mask[i]]
        remaining_sorted = sorted(remaining, key=lambda i: -indiv_hv[i])
        add_list = []
        for i in remaining_sorted:
            add_list.append(points[i].copy())
            if len(add_list) + selected_points.shape[0] >= k:
                break
        if add_list:
            if selected_points.shape[0] == 0:
                selected_points = np.vstack(add_list)
            else:
                selected_points = np.vstack([selected_points, np.vstack(add_list)])
        # truncate to k
        if selected_points.shape[0] > k:
            selected_points = selected_points[:k]

    return np.array(selected_points[:k], dtype=float)


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import pygmo as pg
    import time

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D).")
    n, d = points.shape
    if not (isinstance(k, int) and 0 < k <= n):
        raise ValueError("k must be an integer with 0 < k <= number of points")

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,) matching points' dimension")

    # Helper to compute hypervolume for a set of indices
    def hv_of_indices(indices):
        if len(indices) == 0:
            return 0.0
        try:
            arr = points[np.array(indices)]
            return float(pg.hypervolume(arr).compute(reference_point))
        except Exception:
            return 0.0

    # Precompute individual hypervolumes for all singletons
    individual_hv = np.zeros(n, dtype=float)
    for i in range(n):
        try:
            individual_hv[i] = float(pg.hypervolume(points[[i]]).compute(reference_point))
        except Exception:
            individual_hv[i] = 0.0

    # Trivial case
    if k == n:
        return points.copy()

    # Initialization: k-means++ style seeding that balances individual HV and spread
    selected_indices = []
    unselected = set(range(n))
    # start with the best individual hv
    first = int(np.argmax(individual_hv))
    selected_indices.append(first)
    unselected.remove(first)

    # incremental selection
    for _ in range(1, k):
        sel_pts = points[np.array(selected_indices)]
        # compute distance to nearest selected for every unselected point
        un_idx = np.array(list(unselected), dtype=int)
        if un_idx.size == 0:
            break
        un_pts = points[un_idx]
        # pairwise distances: for each unselected, min distance to any selected
        # norm along axis
        # To be efficient, compute squared distances
        # Broadcast
        dif = un_pts[:, None, :] - sel_pts[None, :, :]
        dist2 = np.sum(dif * dif, axis=2)
        min_dist = np.sqrt(np.min(dist2, axis=1))
        # score = individual_hv * (min_dist + eps)
        eps = 1e-12
        scores = individual_hv[un_idx] * (min_dist + eps)
        # pick the argmax score
        pick_pos = int(np.argmax(scores))
        pick = int(un_idx[pick_pos])
        selected_indices.append(pick)
        unselected.remove(pick)

    selected_set = set(selected_indices)
    current_hv = hv_of_indices(selected_indices)

    # Local improvement: pairwise-exchange (swap) heuristic
    # We will attempt swaps between selected and promising unselected candidates
    max_iters = max(10, 5 * k)  # cap iterations to limit runtime
    tol = 1e-12
    start_time = time.time()
    time_limit = 5.0  # seconds allowed for local search (modest)
    iter_count = 0
    # Prepare ordering of unselected by individual hv for candidate pool
    all_indices = np.arange(n, dtype=int)
    # Precompute ranking of unselected by individual hv descending
    order_by_ind = np.argsort(individual_hv)[::-1]

    while iter_count < max_iters and (time.time() - start_time) < time_limit:
        iter_count += 1
        best_improve = 0.0
        best_swap = None  # tuple (sel_idx_pos_in_selected_indices, unselected_index)
        # Choose a reasonable pool of unselected candidates to try (for speed)
        pool_size = min(max(10 * k, 200), n - k) if (n - k) > 0 else 0
        if pool_size <= 0:
            break
        # Build candidate pool: top by individual hv among unselected
        candidates = []
        for idx in order_by_ind:
            if idx in selected_set:
                continue
            candidates.append(idx)
            if len(candidates) >= pool_size:
                break
        # For each selected element, try replacing it with each candidate
        # We allow early exit if we find a good swap
        made_swap = False
        for sel_pos, sel_idx in enumerate(list(selected_indices)):
            # try each candidate
            for un_idx in candidates:
                # skip if same
                if un_idx == sel_idx:
                    continue
                # form new selection with swap
                new_sel = list(selected_indices)
                new_sel[sel_pos] = int(un_idx)
                hv_new = hv_of_indices(new_sel)
                marginal = hv_new - current_hv
                if marginal > best_improve + tol:
                    best_improve = marginal
                    best_swap = (sel_pos, int(un_idx))
            # small optimization: if we already found a large improvement, we can break early
            # but to keep deterministic quality we still check all selected unless a very large improvement
            if best_improve > 0 and best_improve >= 1e-6:
                # prefer quick acceptance if significantly improving
                break

        # If a beneficial swap found, apply it and continue
        if best_swap is not None and best_improve > tol:
            sel_pos, new_idx = best_swap
            old_idx = selected_indices[sel_pos]
            selected_indices[sel_pos] = new_idx
            selected_set.remove(old_idx)
            selected_set.add(new_idx)
            current_hv += best_improve
            made_swap = True
        if not made_swap:
            break  # no improving swap found -> local optimum

    # As a final safeguard, if we have fewer than k (shouldn't happen), fill with top individual hv
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: individual_hv[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected_indices) >= k:
                break
            selected_indices.append(int(idx))
            selected_set.add(int(idx))

    # Ensure exactly k results
    selected_indices = selected_indices[:k]
    subset = points[np.array(selected_indices, dtype=int)]
    return subset


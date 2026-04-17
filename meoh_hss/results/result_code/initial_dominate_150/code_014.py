import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations. "
                          "Please install pygmo (e.g. pip install pygmo) or provide it in the environment.") from e

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N, D = points.shape

    if k <= 0 or N == 0:
        return np.empty((0, D))

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError(f"reference_point must have shape ({D},), got {reference_point.shape}")

    # If k >= N, return all points (or pad with duplicates if k > N)
    if k >= N:
        if k == N:
            return points.copy()
        extra = k - N
        idx_extra = np.random.choice(N, size=extra, replace=True)
        return np.vstack([points.copy(), points[idx_extra]])

    rng = np.random.default_rng()

    # Helper: compute hypervolume of a set of points
    def hv_of_array(arr):
        if arr.size == 0:
            return 0.0
        return float(pg.hypervolume(arr).compute(reference_point))

    # Greedy initialization: iteratively pick point with largest marginal contribution
    selected = []
    remaining = set(range(N))
    current_points = np.empty((0, D))
    current_hv = 0.0

    for _ in range(k):
        best_idx = None
        best_contrib = -np.inf

        # Evaluate marginal contribution for each candidate
        for i in list(remaining):
            candidate = points[i].reshape(1, -1)
            if current_points.shape[0] == 0:
                hv_union = hv_of_array(candidate)
            else:
                hv_union = hv_of_array(np.vstack([current_points, candidate]))
            contrib = hv_union - current_hv
            # numerical tolerance
            if contrib < 0 and contrib > -1e-12:
                contrib = 0.0
            # tie-breaking by distance to current set (encourage diversity)
            if contrib > best_contrib:
                best_contrib = contrib
                best_idx = i
            elif np.isclose(contrib, best_contrib):
                # choose the one with larger minimum distance to current_points (if any)
                if current_points.shape[0] > 0:
                    cur_min_dist = np.min(np.linalg.norm(current_points - points[best_idx], axis=1))
                    cand_min_dist = np.min(np.linalg.norm(current_points - points[i], axis=1))
                    if cand_min_dist > cur_min_dist + 1e-12:
                        best_idx = i

        if best_idx is None:
            # fallback: pick random
            best_idx = remaining.pop()
        else:
            remaining.remove(best_idx)

        selected.append(best_idx)
        current_points = np.vstack([current_points, points[best_idx].reshape(1, -1)])
        current_hv = hv_of_array(current_points)

    # Local search: bounded best-improvement swaps with sampling to limit runtime
    # Budget of hypervolume evaluations (match typical SA budget)
    eval_budget = max(2000, 200 * k)
    eval_count = 0
    tol = 1e-12

    selected_set = set(selected)
    remaining_list = list(set(range(N)) - selected_set)

    # Precompute selected array for quick replacement
    selected_array = points[np.array(selected)]

    # We'll attempt iterative best-improvement swaps until no improving swap found or budget exhausted
    while eval_count < eval_budget:
        best_delta = 0.0
        best_swap = None  # tuple (idx_in_selected_list, candidate_unselected_index)
        # Determine sampling sizes per selected element to keep evaluations bounded
        # target per-iteration evaluations
        remaining_count = len(remaining_list)
        # If remaining small, we can evaluate all; otherwise sample some
        if remaining_count == 0:
            break
        # budget per iteration (to find the best swap in this pass)
        per_iter_budget = min(eval_budget - eval_count, max(50, eval_budget // max(1, k)))
        # per-selected sample size
        per_selected_sample = max(1, per_iter_budget // max(1, len(selected)))
        # For each selected point, sample up to per_selected_sample candidates from remaining_list
        for s_idx_in_list, s_global_idx in enumerate(list(selected)):
            # sample candidates (without replacement if possible)
            if remaining_count <= per_selected_sample:
                candidates = remaining_list
            else:
                # sample indices from remaining_list
                candidates = list(rng.choice(remaining_list, size=per_selected_sample, replace=False))
            for r_global_idx in candidates:
                # form swapped set: replace selected[s_idx_in_list] with r_global_idx
                swapped_indices = selected.copy()
                swapped_indices[s_idx_in_list] = r_global_idx
                swapped_array = points[np.array(swapped_indices)]
                new_hv = hv_of_array(swapped_array)
                eval_count += 1
                delta = new_hv - current_hv
                if delta > best_delta + tol:
                    best_delta = delta
                    best_swap = (s_idx_in_list, r_global_idx)
                if eval_count >= eval_budget:
                    break
            if eval_count >= eval_budget:
                break
        # If found an improving swap, apply it
        if best_swap is not None and best_delta > tol:
            s_pos, r_idx = best_swap
            old_global = selected[s_pos]
            # perform swap in selected list
            selected[s_pos] = r_idx
            # update sets and lists
            selected_set.remove(old_global)
            selected_set.add(r_idx)
            remaining_list = list(set(range(N)) - selected_set)
            # update current points and hv
            selected_array = points[np.array(selected)]
            current_hv = hv_of_array(selected_array)
            # continue search (with remaining budget)
            continue
        else:
            # no improving swap found within sampling or budget -> finish
            break

    # Ensure exactly k points and return them
    selected = list(selected)[:k]
    result = points[np.array(selected)]
    # If for some reason size mismatch (shouldn't), adjust
    if result.shape[0] < k:
        missing = k - result.shape[0]
        unselected = list(set(range(N)) - set(selected))
        if unselected:
            extra_idx = rng.choice(unselected, size=min(missing, len(unselected)), replace=False)
            result = np.vstack([result, points[extra_idx]])
        # if still short (shouldn't), pad with random duplicates
        if result.shape[0] < k:
            extra_idx = rng.choice(N, size=k - result.shape[0], replace=True)
            result = np.vstack([result, points[extra_idx]])
    elif result.shape[0] > k:
        result = result[:k]

    return result


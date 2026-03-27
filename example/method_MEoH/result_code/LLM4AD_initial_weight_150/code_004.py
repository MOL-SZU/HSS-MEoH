import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
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

    def hv_of(arr_like):
        # arr_like: array-like of shape (m, d) or empty -> returns scalar hv
        if arr_like is None:
            return 0.0
        arr = np.asarray(arr_like)
        if arr.size == 0:
            return 0.0
        # ensure 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        hv = pg.hypervolume(arr)
        return float(hv.compute(reference_point))

    # Precompute single-point hypervolumes as cheap upper bounds for marginal gains
    single_hv = np.zeros(n, dtype=float)
    for i in range(n):
        single_hv[i] = hv_of(points[i])

    # Parameters controlling batch sizes (tunable)
    batch_size = min(n, max(40, n // 10))      # number of top candidates evaluated exactly each greedy step
    swap_top_unselected = min(n, max(200, n // 5))  # number of top unselected candidates considered in swap phase
    max_swap_iters = 100
    tol = 1e-12

    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []
    selected_points = []
    curr_hv = 0.0

    remaining_indices = set(range(n))

    # Greedy selection with top-candidate screening
    while len(selected_indices) < k and remaining_indices:
        # select top candidates by single-point upper bound among remaining
        rem_list = np.array(list(remaining_indices), dtype=int)
        if rem_list.size <= batch_size:
            candidates = rem_list
        else:
            # argsort single_hv descending and pick top batch_size from remaining
            rem_hvs = single_hv[rem_list]
            top_idx = np.argpartition(-rem_hvs, batch_size - 1)[:batch_size]
            candidates = rem_list[top_idx]

        # compute true marginal gains for these candidates
        best_gain = -np.inf
        best_idx = None
        for idx in candidates:
            if len(selected_points) == 0:
                gain = single_hv[idx]
            else:
                cand_set = np.vstack([np.array(selected_points), points[idx]])
                gain = hv_of(cand_set) - curr_hv
            if gain > best_gain + tol:
                best_gain = gain
                best_idx = idx

        # If no positive gain (or all negative/zero), still add the best remaining to reach k
        if best_idx is None:
            # fallback: pick arbitrary remaining
            best_idx = next(iter(remaining_indices))
            best_gain = single_hv[best_idx] if len(selected_points) == 0 else hv_of(np.vstack([np.array(selected_points), points[best_idx]])) - curr_hv

        # Accept best_idx
        selected_indices.append(int(best_idx))
        selected_mask[best_idx] = True
        selected_points.append(points[best_idx].copy())
        # update current hv (recompute to avoid accumulation error)
        curr_hv = hv_of(np.vstack(selected_points)) if len(selected_points) > 0 else 0.0
        remaining_indices.remove(best_idx)

    # If not enough selected (shouldn't happen), fill arbitrarily
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

    # Deterministic pairwise-replacement hill-climb among top unselected candidates
    if n - k > 0:
        # prepare top unselected candidate list by single_hv
        unselected_idxs = np.where(~selected_mask)[0]
        if unselected_idxs.size > 0:
            # sort unselected by single_hv descending and keep top L
            L = min(swap_top_unselected, unselected_idxs.size)
            top_un_idx = unselected_idxs[np.argpartition(-single_hv[unselected_idxs], L - 1)[:L]]
        else:
            top_un_idx = np.array([], dtype=int)

        swap_iter = 0
        improved = True
        while improved and swap_iter < max_swap_iters:
            swap_iter += 1
            improved = False
            best_impr = 0.0
            best_swap = None  # (sel_pos, un_idx, hv_after)
            # evaluate all replacements of selected positions with top unselected candidates
            for pos in range(len(selected_points)):
                # create base set without the selected point at pos
                base_set = np.delete(selected_points, pos, axis=0)
                for cand in top_un_idx:
                    cand_pt = points[cand]
                    cand_set = np.vstack([base_set, cand_pt])
                    hv_cand = hv_of(cand_set)
                    improvement = hv_cand - curr_hv
                    if improvement > best_impr + tol:
                        best_impr = improvement
                        best_swap = (pos, int(cand), hv_cand)
            if best_swap is not None and best_impr > tol:
                pos, cand_idx, hv_after = best_swap
                # apply swap
                old_idx = selected_indices[pos]
                selected_mask[old_idx] = False
                selected_indices[pos] = cand_idx
                selected_mask[cand_idx] = True
                selected_points[pos] = points[cand_idx].copy()
                curr_hv = hv_after
                improved = True
                # update top_un_idx: remove new selected, add old if applicable
                # recompute unselected list top candidates
                unselected_idxs = np.where(~selected_mask)[0]
                if unselected_idxs.size > 0:
                    L = min(swap_top_unselected, unselected_idxs.size)
                    top_un_idx = unselected_idxs[np.argpartition(-single_hv[unselected_idxs], L - 1)[:L]]
                else:
                    top_un_idx = np.array([], dtype=int)

    return np.array(selected_points)


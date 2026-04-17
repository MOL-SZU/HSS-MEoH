import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

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
    k = int(min(k, n))

    if reference_point is None:
        # Slightly beyond the max to ensure reference dominates
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

    # Precompute single-point HV upper bounds (cheap proxies)
    single_hv = np.zeros(n, dtype=float)
    for i in range(n):
        single_hv[i] = hv_of(points[i])

    # Lazy greedy using a max-heap of (estimated_gain, idx).
    # We store negative values because heapq is a min-heap.
    heap = [(-single_hv[i], int(i)) for i in range(n)]
    heapq.heapify(heap)

    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []
    selected_points = []
    curr_hv = 0.0
    tol = 1e-12
    max_lazy_updates = max(10000, 20 * n)  # safety cap to avoid infinite loops
    lazy_updates = 0

    while len(selected_indices) < k and heap:
        lazy_updates += 1
        if lazy_updates > max_lazy_updates:
            # fallback: pick remaining highest single_hv not selected
            for idx in np.argsort(-single_hv):
                if not selected_mask[idx]:
                    selected_indices.append(int(idx))
                    selected_mask[idx] = True
                    selected_points.append(points[int(idx)].copy())
                    if len(selected_indices) == k:
                        break
            curr_hv = hv_of(np.vstack(selected_points)) if selected_points else 0.0
            break

        neg_est, idx = heapq.heappop(heap)
        est = -neg_est
        if selected_mask[idx]:
            continue  # already selected via some fallback swap
        # compute true marginal gain
        if len(selected_points) == 0:
            true_gain = single_hv[idx]
        else:
            cand_set = np.vstack([np.array(selected_points), points[idx]])
            true_gain = hv_of(cand_set) - curr_hv

        # If the estimate is close to true gain (within tol) or estimate <= true_gain (conservative),
        # accept it; otherwise reinsert with updated estimate (lazy update).
        if est <= true_gain + tol:
            # accept this point
            selected_indices.append(int(idx))
            selected_mask[idx] = True
            selected_points.append(points[idx].copy())
            # recompute curr_hv exactly to avoid accumulation error
            curr_hv = hv_of(np.vstack(selected_points)) if selected_points else 0.0
        else:
            # push updated estimate (true_gain) back into heap for reconsideration
            heapq.heappush(heap, (-true_gain, int(idx)))

    # If still not enough (rare), fill with highest single_hv remaining
    if len(selected_indices) < k:
        for idx in np.argsort(-single_hv):
            if not selected_mask[idx]:
                selected_indices.append(int(idx))
                selected_mask[idx] = True
                selected_points.append(points[int(idx)].copy())
                if len(selected_indices) == k:
                    break
        curr_hv = hv_of(np.vstack(selected_points)) if selected_points else 0.0

    selected_points = np.array(selected_points, dtype=float)

    # Focused deterministic swap-based local search:
    # Consider only top candidates by single_hv (both selected and unselected) to limit cost.
    if n - k > 0 and k > 0:
        top_un_limit = min(200, max(20, n // 10))
        top_sel_limit = min(k, max(20, k))  # consider at most k (or some cap)
        # top unselected by single_hv
        unselected_idxs = np.where(~selected_mask)[0]
        if unselected_idxs.size > 0:
            L = min(top_un_limit, unselected_idxs.size)
            top_un_idx = unselected_idxs[np.argpartition(-single_hv[unselected_idxs], L - 1)[:L]]
        else:
            top_un_idx = np.array([], dtype=int)
        # selected positions to consider (by their single_hv, lowest contribution first helps)
        sel_idxs_array = np.array(selected_indices, dtype=int)
        if sel_idxs_array.size > 0:
            S = min(top_sel_limit, sel_idxs_array.size)
            # choose S selected indices with smallest single_hv to potentially replace
            sel_by_sv = sel_idxs_array[np.argpartition(single_hv[sel_idxs_array], S - 1)[:S]]
        else:
            sel_by_sv = np.array([], dtype=int)

        improved = True
        swap_iters = 0
        max_swap_iters = 200
        while improved and swap_iters < max_swap_iters:
            swap_iters += 1
            improved = False
            best_impr = 0.0
            best_swap = None  # (selected_pos_idx_in_selected_points_list, un_idx, hv_after)
            # For each selected position in the considered subset, try replacing with top unselected candidates
            for sel_global_idx in sel_by_sv:
                # find local position in selected_indices list
                try:
                    pos = selected_indices.index(int(sel_global_idx))
                except ValueError:
                    continue  # might have been swapped away
                base_set = np.delete(selected_points, pos, axis=0)
                for cand in top_un_idx:
                    if cand == sel_global_idx:
                        continue
                    cand_pt = points[cand]
                    cand_set = np.vstack([base_set, cand_pt])
                    hv_cand = hv_of(cand_set)
                    improvement = hv_cand - curr_hv
                    if improvement > best_impr + tol:
                        best_impr = improvement
                        best_swap = (pos, int(cand), hv_cand)
            if best_swap is not None and best_impr > tol:
                pos, cand_idx, hv_after = best_swap
                old_global_idx = selected_indices[pos]
                # apply swap
                selected_mask[old_global_idx] = False
                selected_mask[cand_idx] = True
                selected_indices[pos] = cand_idx
                selected_points[pos] = points[cand_idx].copy()
                curr_hv = hv_after
                improved = True
                # update candidate pools
                unselected_idxs = np.where(~selected_mask)[0]
                if unselected_idxs.size > 0:
                    L = min(top_un_limit, unselected_idxs.size)
                    top_un_idx = unselected_idxs[np.argpartition(-single_hv[unselected_idxs], L - 1)[:L]]
                else:
                    top_un_idx = np.array([], dtype=int)
                sel_idxs_array = np.array(selected_indices, dtype=int)
                if sel_idxs_array.size > 0:
                    S = min(top_sel_limit, sel_idxs_array.size)
                    sel_by_sv = sel_idxs_array[np.argpartition(single_hv[sel_idxs_array], S - 1)[:S]]
                else:
                    sel_by_sv = np.array([], dtype=int)
            else:
                break  # no improving swap found

    # Final safety: ensure output shape (k, d)
    if selected_points.shape[0] > k:
        selected_points = selected_points[:k]
    elif selected_points.shape[0] < k:
        # pad with remaining points (shouldn't happen)
        for idx in range(n):
            if not selected_mask[idx]:
                selected_points = np.vstack([selected_points, points[idx].copy()])
                if selected_points.shape[0] == k:
                    break

    return np.array(selected_points, dtype=float)


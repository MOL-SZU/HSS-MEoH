import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape
    if not (isinstance(k, int) and k >= 0):
        raise ValueError("k must be a non-negative integer")
    if k == 0:
        return np.zeros((0, D), dtype=float)

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # If k >= N, return all points
    if k >= N:
        return points.copy()

    # Exact union-volume computation (recursive slicing) adapted to boxes anchored at ref
    def union_boxes_volume(boxes, ref):
        boxes = np.asarray(boxes, dtype=float)
        if boxes.size == 0:
            return 0.0
        # Only consider boxes with positive size (all coordinates > ref)
        mask_pos = np.all(boxes > ref, axis=1)
        boxes = boxes[mask_pos]
        if boxes.shape[0] == 0:
            return 0.0

        def _union_recursive(bxs, r):
            if bxs.shape[0] == 0:
                return 0.0
            d = r.size
            # 1D base case
            if d == 1:
                max_end = np.max(bxs[:, 0])
                return max(0.0, max_end - r[0])
            # split by distinct coordinates in current dimension including ref
            coords = np.unique(np.concatenate(([r[0]], bxs[:, 0])))
            coords.sort()
            total = 0.0
            for i in range(len(coords) - 1):
                x_left = coords[i]
                x_right = coords[i + 1]
                if x_right <= x_left:
                    continue
                # boxes that extend at least to x_right (their first coord >= x_right)
                mask = bxs[:, 0] >= x_right
                if not np.any(mask):
                    continue
                sub_boxes = bxs[mask][:, 1:]
                sub_ref = r[1:]
                sub_vol = _union_recursive(sub_boxes, sub_ref)
                if sub_vol > 0.0:
                    total += (x_right - x_left) * sub_vol
            return total

        return float(_union_recursive(boxes, ref))

    # Compute single-box exact volumes quickly (product of positive dims)
    single_volumes = np.prod(np.maximum(0.0, points - reference_point), axis=1)

    # If everything is zero volume in some dims -> pick top-k by single_volumes
    if np.all(single_volumes == 0.0):
        idxs = np.argsort(single_volumes)[::-1][:k]
        return points[idxs].copy()

    rng = np.random.default_rng()

    # INITIALIZATION: simple greedy base by single-box volumes (largest first)
    order_by_single = np.argsort(single_volumes)[::-1]
    selected_indices = []
    selected_mask = np.zeros(N, dtype=bool)
    for idx in order_by_single:
        if len(selected_indices) >= k:
            break
        if single_volumes[idx] <= 0.0:
            # if remaining single volumes are zero, still allow until k
            selected_indices.append(int(idx))
            selected_mask[idx] = True
        else:
            selected_indices.append(int(idx))
            selected_mask[idx] = True
    # Ensure we have exactly k (fallback)
    if len(selected_indices) < k:
        remaining = [i for i in range(N) if not selected_mask[i]]
        remaining_sorted = sorted(remaining, key=lambda x: single_volumes[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected_indices) >= k:
                break
            selected_indices.append(int(idx))
            selected_mask[idx] = True

    selected_indices = selected_indices[:k]
    selected_boxes = points[np.array(selected_indices, dtype=int), :].copy()
    current_volume = union_boxes_volume(selected_boxes, reference_point)

    # STOCHASTIC LOCAL SEARCH: try swaps to improve exact hypervolume
    max_iterations = 60
    candidate_pool_size = min(max(20, N // 10), N - k if N - k > 0 else 0)
    swap_trials = min(60, max(10, N // 5))
    # Precompute sampling weights for non-selected points (prefer larger single volumes)
    weights = single_volumes + 1e-12
    if np.sum(weights) == 0:
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)

    for it in range(max_iterations):
        improved = False
        best_swap = None  # (delta, remove_pos_in_selected, candidate_index, new_volume)
        # Build candidate pool once per iteration (non-selected)
        available = np.where(~selected_mask)[0]
        if available.size == 0:
            break
        # sample a set of candidate indices (without replacement) weighted by single volumes
        if candidate_pool_size > 0 and available.size > candidate_pool_size:
            p_avail = weights[available] / np.sum(weights[available])
            cand_pool = rng.choice(available, size=candidate_pool_size, replace=False, p=p_avail)
        else:
            cand_pool = available

        # For each selected element, try a limited number of candidate swaps
        for sel_pos, sel_idx in enumerate(list(selected_indices)):
            # random subset of candidates to try swapping in
            if cand_pool.size == 0:
                break
            n_trials = min(swap_trials, cand_pool.size)
            try_candidates = rng.choice(cand_pool, size=n_trials, replace=False)
            # Prepare boxes without current selected to speed some calls
            if k == 1:
                base_boxes = np.zeros((0, D), dtype=float)
            else:
                mask = np.ones(len(selected_indices), dtype=bool)
                mask[sel_pos] = False
                base_boxes = selected_boxes[mask]
            # For each candidate evaluate exact union volume when swapping
            for cand_idx in try_candidates:
                # form new set: base_boxes + candidate
                combined = cand_point = points[cand_idx:cand_idx+1, :]
                if base_boxes.shape[0] == 0:
                    new_vol = union_boxes_volume(combined, reference_point)
                else:
                    new_vol = union_boxes_volume(np.vstack((base_boxes, combined)), reference_point)
                delta = new_vol - current_volume
                if delta > 1e-12 and (best_swap is None or delta > best_swap[0]):
                    best_swap = (delta, sel_pos, int(cand_idx), new_vol)
        # If we found an improving swap, apply the best one
        if best_swap is not None:
            delta, sel_pos, cand_idx, new_vol = best_swap
            # perform swap
            removed_idx = selected_indices[sel_pos]
            selected_indices[sel_pos] = cand_idx
            selected_mask[removed_idx] = False
            selected_mask[cand_idx] = True
            # update selected_boxes and current_volume
            selected_boxes[sel_pos, :] = points[cand_idx, :]
            current_volume = float(new_vol)
            improved = True
        if not improved:
            # no improving swap found -> terminate early
            break

    # Final safety: if selection changed size (shouldn't), fix
    if len(selected_indices) < k:
        remaining = [i for i in range(N) if not selected_mask[i]]
        remaining_sorted = sorted(remaining, key=lambda x: single_volumes[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected_indices) >= k:
                break
            selected_indices.append(int(idx))
            selected_mask[idx] = True

    selected_indices = selected_indices[:k]
    subset = points[np.array(selected_indices, dtype=int), :].copy()
    return subset


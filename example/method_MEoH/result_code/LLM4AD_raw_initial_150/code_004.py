import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    try:
        import pygmo as pg
    except Exception:
        raise ImportError("pygmo is required for hypervolume computations (pip install pygmo).")

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, m = points.shape

    if k <= 0:
        return np.zeros((0, m))
    if k >= n:
        return points.copy()

    # Reference point handling
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (m,):
        raise ValueError("reference_point must have shape (D,) matching points' dimension")
    max_pts = np.max(points, axis=0)
    reference_point = np.maximum(reference_point, max_pts * 1.0 + 1e-8)

    # Filter out dominated points (Pareto pruning)
    pts = points.copy()
    n2 = pts.shape[0]
    dominated = np.zeros(n2, dtype=bool)
    for i in range(n2):
        if dominated[i]:
            continue
        ge = np.all(pts >= pts[i], axis=1)
        gt = np.any(pts > pts[i], axis=1)
        dom_by = np.where(ge & gt)[0]
        for j in dom_by:
            if j != i:
                dominated[i] = True
                break
    keep_idx = np.where(~dominated)[0]
    if keep_idx.size == 0:
        # fallback, return k arbitrary points
        return points[:min(k, n), :].copy()

    points_f = pts[keep_idx, :]
    orig_idx_map = keep_idx
    n_f = points_f.shape[0]

    # If we have fewer non-dominated points than k, include all and pad later
    if n_f == 0:
        return points[:min(k, n), :].copy()
    if k >= n_f:
        # will pad with dominated originals later
        selected_all = points_f.copy()
        if selected_all.shape[0] >= k:
            return selected_all[:k, :].copy()

    # Precompute single-point hypervolumes (upper bounds)
    single_hvs = np.empty(n_f, dtype=float)
    for i in range(n_f):
        try:
            single_hvs[i] = pg.hypervolume(points_f[i:i+1, :]).compute(reference_point)
        except Exception:
            single_hvs[i] = 0.0

    # Lazy greedy using a max-heap of estimated gains (initially single_hvs)
    selected_mask = np.zeros(n_f, dtype=bool)
    selected_idx = []
    selected_points = np.empty((0, m), dtype=float)
    current_hv = 0.0
    tol = 1e-12

    # Build heap: (-est_gain, idx, est_gain)
    heap = [(-single_hvs[i], i, single_hvs[i]) for i in range(n_f)]
    heapq.heapify(heap)

    selections = 0
    while selections < min(k, n_f) and heap:
        neg_est, idx, est = heapq.heappop(heap)
        if selected_mask[idx]:
            continue  # already selected
        # Recompute true marginal gain for idx
        cand = points_f[idx:idx+1, :]
        try:
            if selected_points.size == 0:
                hv_combined = pg.hypervolume(cand).compute(reference_point)
            else:
                hv_combined = pg.hypervolume(np.vstack([selected_points, cand])).compute(reference_point)
        except Exception:
            hv_combined = 0.0
        true_gain = hv_combined - current_hv
        # If the estimate was optimistic and true_gain is significantly smaller, push updated value
        if true_gain < est - tol:
            # push updated estimate back
            heapq.heappush(heap, (-true_gain, idx, true_gain))
            continue
        # Otherwise accept this candidate
        selected_mask[idx] = True
        selected_idx.append(idx)
        if selected_points.size == 0:
            selected_points = cand.copy()
        else:
            selected_points = np.vstack([selected_points, cand])
        try:
            current_hv = pg.hypervolume(selected_points).compute(reference_point)
        except Exception:
            current_hv = 0.0
        selections += 1

    # If we didn't select enough due to numerical issues, fill by highest single_hvs remaining
    if len(selected_idx) < min(k, n_f):
        remaining = np.where(~selected_mask)[0]
        order = remaining[np.argsort(-single_hvs[remaining])]
        for idx in order:
            if len(selected_idx) >= min(k, n_f):
                break
            selected_mask[idx] = True
            selected_idx.append(idx)
            cand = points_f[idx:idx+1, :]
            if selected_points.size == 0:
                selected_points = cand.copy()
            else:
                selected_points = np.vstack([selected_points, cand])
            try:
                current_hv = pg.hypervolume(selected_points).compute(reference_point)
            except Exception:
                current_hv = 0.0

    # Local 1-for-1 swap improvement (try to escape local opt)
    max_swap_iters = 5
    improved = True
    swap_iter = 0
    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        # For each selected position, try replacements
        for pos_in_sel, sel_f_idx in enumerate(list(selected_idx)):
            remaining = np.where(~selected_mask)[0]
            if remaining.size == 0:
                break
            # try candidates ordered by single_hvs descending
            cand_order = remaining[np.argsort(-single_hvs[remaining])]
            for cand_f_idx in cand_order:
                # propose swap
                trial = selected_points.copy()
                trial[pos_in_sel, :] = points_f[cand_f_idx, :]
                try:
                    hv_trial = pg.hypervolume(trial).compute(reference_point)
                except Exception:
                    hv_trial = 0.0
                if hv_trial > current_hv + tol:
                    # accept swap
                    selected_mask[sel_f_idx] = False
                    selected_mask[cand_f_idx] = True
                    selected_idx[pos_in_sel] = cand_f_idx
                    selected_points = trial
                    current_hv = hv_trial
                    improved = True
                    break
            if improved:
                break

    # Map selected indices back to original points
    final_filtered_indices = np.array(selected_idx, dtype=int)
    final_original_indices = orig_idx_map[final_filtered_indices]
    subset = points[final_original_indices, :].copy()

    # If fewer than k due to filtering, pad by taking from dominated originals first, ranked by single-point hv if possible
    if subset.shape[0] < k:
        needed = k - subset.shape[0]
        remaining_orig = np.setdiff1d(np.arange(n), final_original_indices, assume_unique=True)
        if remaining_orig.size > 0:
            # Try to prefer non-dominated remaining by using mapping if available
            # compute simple scores for remaining (single point hv if in filtered set, else approximate by point volume)
            scores = []
            for idx in remaining_orig:
                if idx in orig_idx_map:
                    # this shouldn't normally happen, but keep safe
                    fid = np.where(orig_idx_map == idx)[0]
                    sc = single_hvs[fid[0]] if fid.size else 0.0
                else:
                    # approximate by box volume (reference - point) product as cheap proxy
                    diff = reference_point - points[idx]
                    diff = np.maximum(diff, 0.0)
                    sc = np.prod(diff)
                scores.append(sc)
            scores = np.array(scores)
            order = np.argsort(-scores)
            take = remaining_orig[order][:needed]
            extra = points[take, :]
            subset = np.vstack([subset, extra])

    # Ensure shape (k, m)
    if subset.shape[0] > k:
        subset = subset[:k, :]
    return subset[:k, :].copy()


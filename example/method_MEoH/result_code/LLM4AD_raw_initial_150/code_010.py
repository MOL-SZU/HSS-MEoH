import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    HSS (Hypervolume-based Subset Selection): 贪心选择超体积贡献最大的 k 个点

    参数:
        points: np.ndarray
            所有候选点，形状为 (N, D)，其中 N 是点数，D 是目标维度
        k: int
            需要选择的点数
        reference_point: np.ndarray, optional
            参考点，形状为 (D,)。如果为 None，则使用 points 的最大值 * 1.1

    返回:
        subset: np.ndarray
            选出的子集，形状为 (k, D)
    """
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
    # ensure reference is strictly >= max point to get positive volumes
    reference_point = np.maximum(reference_point, max_pts + 1e-8)

    # Pareto (dominated) pruning - vectorized
    pts = points.copy()
    # For each i, check if exists j != i such that pts[j] >= pts[i] and any >
    ge = (pts[:, None, :] >= pts[None, :, :])  # shape (n, n, m)
    all_ge = np.all(ge, axis=2)                # (n, n)
    any_gt = np.any(pts[:, None, :] > pts[None, :, :], axis=2)  # (n, n)
    dominated_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        # if any j dominates i (j != i)
        dom_by = np.where((all_ge[:, i]) & (any_gt[:, i]))[0]
        # dom_by might include i itself in all_ge but any_gt excludes equality
        if dom_by.size > 0:
            dominated_mask[i] = True
    keep_idx = np.where(~dominated_mask)[0]
    if keep_idx.size == 0:
        # fallback: return first k points
        return points[:min(k, n), :].copy()

    points_f = pts[keep_idx, :]
    orig_idx_map = keep_idx.copy()
    n_f = points_f.shape[0]

    # If filtered points fewer than k, will later pad
    if n_f == 0:
        return points[:min(k, n), :].copy()
    if k >= n_f:
        # take all non-dominated, pad later
        selected_all = points_f.copy()
        if selected_all.shape[0] >= k:
            return selected_all[:k, :].copy()

    # Precompute single-point hypervolumes and box-volume proxies
    single_hvs = np.empty(n_f, dtype=float)
    box_vol = np.empty(n_f, dtype=float)
    eps = 1e-12
    diffs = reference_point[None, :] - points_f  # (n_f, m)
    diffs = np.maximum(diffs, 0.0)
    # compute box volume proxy
    box_vol = np.prod(diffs, axis=1)
    # compute single hypervolumes
    for i in range(n_f):
        try:
            single_hvs[i] = float(pg.hypervolume(points_f[i:i+1, :]).compute(reference_point))
        except Exception:
            single_hvs[i] = 0.0

    # Score function parameters (different from original): geometric mean with diversity boost
    alpha = 0.65  # weight on single_hvs in geometric mean
    diversity_lambda = 0.20  # strength of distance-based diversity boost
    # compute base_est scores
    # to avoid zeros collapse in geometric mean, add tiny eps
    base_est = (single_hvs + eps) ** alpha * (box_vol + eps) ** (1.0 - alpha)

    # Lazy greedy using max-heap of estimated gains (initially base_est)
    selected_mask = np.zeros(n_f, dtype=bool)
    selected_idx = []
    selected_points = np.empty((0, m), dtype=float)
    current_hv = 0.0
    tol = 1e-12

    # For normalization of distances (scale)
    scale = np.linalg.norm(reference_point - np.min(points, axis=0))
    if scale <= 0:
        scale = 1.0

    # Build heap entries: (-est_score, idx, est_score)
    heap = [(-float(base_est[i]), int(i), float(base_est[i])) for i in range(n_f)]
    heapq.heapify(heap)

    selections = 0
    while selections < min(k, n_f) and heap:
        neg_est, idx, est = heapq.heappop(heap)
        if selected_mask[idx]:
            continue
        # compute a lightweight diversity multiplier if we already have selections
        diversity_mult = 1.0
        if selected_points.size != 0:
            # min Euclidean distance to selected_points
            dists = np.linalg.norm(selected_points - points_f[idx:idx+1, :], axis=1)
            min_dist = np.min(dists)
            diversity_mult = 1.0 + diversity_lambda * (min_dist / scale)
        # optimistic estimated marginal gain (used previously)
        optimistic = est * diversity_mult

        # Recompute true marginal gain for idx (exact)
        cand = points_f[idx:idx+1, :]
        try:
            if selected_points.size == 0:
                hv_with = float(pg.hypervolume(cand).compute(reference_point))
            else:
                hv_with = float(pg.hypervolume(np.vstack([selected_points, cand])).compute(reference_point))
        except Exception:
            hv_with = 0.0
        true_gain = hv_with - current_hv

        # If optimistic estimate was too high, push updated estimate back (apply diversity factor in est stored)
        if true_gain < optimistic - tol:
            # update raw est as true_gain / diversity_mult for heap consistency (avoid storing inflated)
            updated_est = max(true_gain / (diversity_mult + 1e-18), 0.0)
            heapq.heappush(heap, (-updated_est, idx, updated_est))
            continue

        # Accept candidate
        selected_mask[idx] = True
        selected_idx.append(int(idx))
        if selected_points.size == 0:
            selected_points = cand.copy()
        else:
            selected_points = np.vstack([selected_points, cand])
        try:
            current_hv = float(pg.hypervolume(selected_points).compute(reference_point))
        except Exception:
            current_hv = 0.0
        selections += 1

    # If didn't select enough, fill by highest base_est among remaining
    if len(selected_idx) < min(k, n_f):
        remaining = np.where(~selected_mask)[0]
        order = remaining[np.argsort(-base_est[remaining])]
        for idx in order:
            if len(selected_idx) >= min(k, n_f):
                break
            selected_mask[int(idx)] = True
            selected_idx.append(int(idx))
            cand = points_f[int(idx):int(idx)+1, :]
            if selected_points.size == 0:
                selected_points = cand.copy()
            else:
                selected_points = np.vstack([selected_points, cand])
            try:
                current_hv = float(pg.hypervolume(selected_points).compute(reference_point))
            except Exception:
                current_hv = 0.0

    # Focused 1-for-1 swap improvement: try top candidates only for speed
    max_swap_iters = 6
    top_try = max(50, int(0.5 * n_f))  # only consider top fraction as replacements
    swap_iter = 0
    improved = True
    # precompute candidate ordering by base_est descending
    candidate_order_all = np.argsort(-base_est)
    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        for pos_in_sel, sel_f_idx in enumerate(list(selected_idx)):
            # remaining candidates indices
            remaining = np.where(~selected_mask)[0]
            if remaining.size == 0:
                break
            # try only top candidates among remaining according to base_est
            # intersect candidate_order_all with remaining
            candidates_to_try = [int(i) for i in candidate_order_all if (not selected_mask[int(i)])]
            if len(candidates_to_try) > top_try:
                candidates_to_try = candidates_to_try[:top_try]
            swapped = False
            for cand_f_idx in candidates_to_try:
                # propose swap
                trial = selected_points.copy()
                trial[pos_in_sel, :] = points_f[cand_f_idx, :]
                try:
                    hv_trial = float(pg.hypervolume(trial).compute(reference_point))
                except Exception:
                    hv_trial = 0.0
                if hv_trial > current_hv + tol:
                    # accept swap
                    selected_mask[int(sel_f_idx)] = False
                    selected_mask[int(cand_f_idx)] = True
                    selected_idx[pos_in_sel] = int(cand_f_idx)
                    selected_points = trial
                    current_hv = hv_trial
                    improved = True
                    swapped = True
                    break
            if swapped:
                break

    # Map selected indices back to original points
    final_filtered_indices = np.array(selected_idx, dtype=int)
    final_original_indices = orig_idx_map[final_filtered_indices]
    subset = points[final_original_indices, :].copy()

    # If fewer than k due to filtering, pad by taking from dominated originals ranked by box_vol proxy then single_hvs
    if subset.shape[0] < k:
        needed = k - subset.shape[0]
        remaining_orig = np.setdiff1d(np.arange(n), final_original_indices, assume_unique=True)
        if remaining_orig.size > 0:
            scores = np.empty(remaining_orig.size, dtype=float)
            for ii, idx in enumerate(remaining_orig):
                # if original index is in filtered mapping (rare), use precomputed values
                hit = np.where(orig_idx_map == idx)[0]
                if hit.size:
                    sc = base_est[hit[0]]
                else:
                    diff = reference_point - points[idx]
                    diff = np.maximum(diff, 0.0)
                    sc = np.prod(diff)
                scores[ii] = sc
            order = np.argsort(-scores)
            take = remaining_orig[order][:needed]
            extra = points[take, :]
            subset = np.vstack([subset, extra])

    # Ensure shape (k, m)
    if subset.shape[0] > k:
        subset = subset[:k, :]
    # If still less (shouldn't happen), pad with arbitrary points
    if subset.shape[0] < k:
        needed = k - subset.shape[0]
        extras = points[:needed, :]
        subset = np.vstack([subset, extras])

    return subset[:k, :].copy()


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
    reference_point = np.maximum(reference_point, max_pts * 1.0 + 1e-8)

    # Pareto pruning: keep non-dominated points
    pts = points.copy()
    n2 = pts.shape[0]
    dominated = np.zeros(n2, dtype=bool)
    for i in range(n2):
        if dominated[i]:
            continue
        # check if any other j dominates i
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
        selected_all = points_f.copy()
        if selected_all.shape[0] >= k:
            return selected_all[:k, :].copy()

    # Precompute single-point hypervolumes (exact) and box volumes (cheap)
    single_hvs = np.empty(n_f, dtype=float)
    box_vol = np.empty(n_f, dtype=float)
    for i in range(n_f):
        try:
            single_hvs[i] = pg.hypervolume(points_f[i:i+1, :]).compute(reference_point)
        except Exception:
            single_hvs[i] = 0.0
        diff = reference_point - points_f[i]
        diff = np.maximum(diff, 0.0)
        # if any diff is 0, volume may be zero; keep it
        box_vol[i] = np.prod(diff)

    # Diversity proxy: compute minimum Euclidean distance to nearest neighbor (in candidate set)
    # For speed, if n_f large, approximate by sampling neighbors for distance computation
    def compute_min_dists(X):
        nf = X.shape[0]
        # if small, compute full pairwise; else sample up to 500 others for each point
        if nf <= 1000:
            # pairwise distances
            dif = X[:, None, :] - X[None, :, :]
            d2 = np.sum(dif * dif, axis=2)
            # set diagonal to inf
            np.fill_diagonal(d2, np.inf)
            mind = np.sqrt(np.min(d2, axis=1))
            return mind
        else:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(nf, size=min(500, nf), replace=False)
            sample = X[sample_idx]
            # distance from each point to the sample (excluding self if sampled)
            dif = X[:, None, :] - sample[None, :, :]
            d2 = np.sum(dif * dif, axis=2)
            mind = np.sqrt(np.min(d2, axis=1))
            return mind

    min_dists = compute_min_dists(points_f)

    # Normalize components to comparable scales
    def safe_norm(arr):
        mx = np.max(arr)
        if mx <= 0:
            return np.zeros_like(arr)
        return arr / mx

    h_norm = safe_norm(single_hvs)
    v_norm = safe_norm(box_vol)
    d_norm = safe_norm(min_dists)

    # Score weights: emphasize hypervolume but add box volume and diversity component
    w_hv = 0.7
    w_vol = 0.2
    w_div = 0.1
    # Combined initial estimate score (different from original single_hvs-only)
    init_scores = w_hv * h_norm + w_vol * v_norm + w_div * d_norm

    # Map score into estimated gain scale by multiplying with single_hvs max to keep units closer to hv
    est_scale = max(1.0, np.max(single_hvs))
    est_init = init_scores * est_scale

    # Lazy greedy using max-heap of estimated gains (initially est_init)
    selected_mask = np.zeros(n_f, dtype=bool)
    selected_idx = []
    selected_points = np.empty((0, m), dtype=float)
    current_hv = 0.0
    tol = 1e-12

    # Build heap: (-est_gain, idx, est_gain)
    heap = [(-est_init[i], i, est_init[i]) for i in range(n_f)]
    heapq.heapify(heap)

    selections = 0
    target_sel = min(k, n_f)
    while selections < target_sel and heap:
        neg_est, idx, est = heapq.heappop(heap)
        if selected_mask[idx]:
            continue
        cand = points_f[idx:idx+1, :]
        # compute true marginal hv gain
        try:
            if selected_points.size == 0:
                hv_combined = pg.hypervolume(cand).compute(reference_point)
            else:
                hv_combined = pg.hypervolume(np.vstack([selected_points, cand])).compute(reference_point)
        except Exception:
            hv_combined = 0.0
        true_gain = hv_combined - current_hv
        # If estimate is still optimistic, push updated estimate
        if true_gain < est - tol:
            # push updated estimate back
            heapq.heappush(heap, (-true_gain, idx, true_gain))
            continue
        # Accept candidate
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

    # If not enough selected (numerical or heap issues), fill by highest estimated scores
    if len(selected_idx) < target_sel:
        remaining = np.where(~selected_mask)[0]
        order = remaining[np.argsort(-est_init[remaining])]
        for idx in order:
            if len(selected_idx) >= target_sel:
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

    # Local 1-for-1 swap improvement (more aggressive, test top candidates by score)
    max_swap_iters = 10
    improved = True
    swap_iter = 0
    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        # For each selected position, try best replacements (by init estimated score)
        for pos_in_sel, sel_f_idx in enumerate(list(selected_idx)):
            remaining = np.where(~selected_mask)[0]
            if remaining.size == 0:
                break
            # try top candidates ordered by initial estimated score
            cand_order = remaining[np.argsort(-est_init[remaining])]
            # limit number of candidates tested for speed
            cand_order = cand_order[:min(200, cand_order.size)]
            for cand_f_idx in cand_order:
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

    # If fewer than k due to filtering, pad by taking from dominated originals first, ranked by a cheap score
    if subset.shape[0] < k:
        needed = k - subset.shape[0]
        remaining_orig = np.setdiff1d(np.arange(n), final_original_indices, assume_unique=True)
        if remaining_orig.size > 0:
            # compute cheap scores: prefer non-dominated-like using box volume; for filtered ones use single_hvs mapping
            scores = []
            for idx in remaining_orig:
                if idx in orig_idx_map:
                    fid = np.where(orig_idx_map == idx)[0]
                    sc = single_hvs[fid[0]] if fid.size else 0.0
                else:
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
    # If fewer (shouldn't happen), pad with duplicates from original first rows
    if subset.shape[0] < k:
        pad_needed = k - subset.shape[0]
        add = points[:pad_needed, :]
        subset = np.vstack([subset, add])

    return subset[:k, :].copy()


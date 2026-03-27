import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    N, D = pts.shape

    if k <= 0:
        return np.zeros((0, D), dtype=pts.dtype)
    if k >= N:
        return pts.copy()

    # Reference handling
    if reference_point is None:
        ref = pts.max(axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float)
        if ref.ndim == 0:
            ref = np.full((D,), float(ref))
        if ref.shape != (D,):
            raise ValueError("reference_point must have shape (D,)")
        max_p = pts.max(axis=0)
        ref = np.maximum(ref, max_p + 1e-12)

    # Remove points that have any coordinate > ref (they contribute zero rect)
    valid_mask = np.all(pts <= ref + 1e-12, axis=1)
    if not np.all(valid_mask):
        pts = pts[valid_mask]
        N = pts.shape[0]
        if N == 0:
            return np.zeros((0, D), dtype=pts.dtype)
        if k >= N:
            return pts.copy()

    # Pareto (non-dominated) filter for minimization (lower is better)
    # vectorized-ish nondomination (keep simple O(N^2) boolean mask but skip already removed)
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        leq = np.all(pts <= pts[i], axis=1)
        lt = np.any(pts < pts[i], axis=1)
        dominated_by = leq & lt
        dominated_by[i] = False
        if np.any(dominated_by):
            keep[i] = False
    if not np.all(keep):
        pts = pts[keep]
        N = pts.shape[0]
        if N == 0:
            return np.zeros((0, D), dtype=pts.dtype)
        if k >= N:
            return pts.copy()

    # Monte-Carlo sampling region: from lower=min(pts) to upper=ref
    lower = pts.min(axis=0)
    upper = ref.copy()
    zero_range = (upper - lower) <= 0
    if np.any(zero_range):
        upper = upper.copy()
        upper[zero_range] = lower[zero_range] + 1e-6

    vol_box = float(np.prod(upper - lower))

    # Adaptive M depending on k, D, and pool size heuristics (reduced for speed but adaptive)
    M_base = max(800, 120 * min(k, 60))
    M_dim = int(80 * max(0, D - 2))
    M = int(min(20000, max(800, min(M_base + M_dim, 20000))))
    rng = np.random.default_rng(987654321)
    u = rng.random((M, D))
    samples = lower + u * (upper - lower)

    # Precompute dominated masks (sample in rect of point)
    # sample s is dominated by point p if s >= p componentwise (minimization convention)
    dominated = (samples[:, None, :] >= pts[None, :, :]).all(axis=2)  # shape (M, N)

    # Per-point sample counts and rect volumes (singleton rect volume)
    sample_counts = dominated.sum(axis=0)  # length N
    rect_volumes = np.prod(np.maximum(ref - pts, 0.0), axis=1)

    # Diversity surrogate: distance to centroid (cheap and stable)
    centroid = pts.mean(axis=0)
    diversity = np.linalg.norm(pts - centroid, axis=1)

    # Normalize surrogates and blend to score for shortlist
    norm_samples = sample_counts.astype(float) / float(M) if M > 0 else np.zeros_like(sample_counts, dtype=float)
    vmax = rect_volumes.max() if rect_volumes.size > 0 else 1.0
    norm_vols = rect_volumes / vmax if vmax > 0 else np.zeros_like(rect_volumes, dtype=float)
    dmax = diversity.max() if diversity.size > 0 else 1.0
    norm_div = diversity / dmax if dmax > 0 else np.zeros_like(diversity, dtype=float)

    # New blending weights (emphasize diversity more than original)
    w_s = 0.50  # sample-based signal
    w_v = 0.20  # rect-volume signal
    w_d = 0.30  # diversity signal
    scores = w_s * norm_samples + w_v * norm_vols + w_d * norm_div

    # Adaptive shortlist size c * k with smaller c
    c = 3
    shortlist_size = int(min(N, max(30, c * k)))
    order = np.argsort(-scores)
    shortlist_idx = order[:shortlist_size]
    pool_idx_map = np.array(shortlist_idx, dtype=int)
    pool_pts = pts[pool_idx_map]
    pool_dom = dominated[:, pool_idx_map]  # (M, pool_N)
    pool_counts = sample_counts[pool_idx_map]
    pool_rects = rect_volumes[pool_idx_map]
    pool_scores = scores[pool_idx_map]

    pool_N = pool_pts.shape[0]
    if pool_N == 0:
        return np.zeros((0, D), dtype=pts.dtype)
    if k >= pool_N:
        return pool_pts.copy()

    # Lazy-stamped greedy: heap stores (-estimated_count, pidx)
    # Start estimates as score-proportional counts for better seeding
    est_counts = np.clip((pool_scores * float(M)).astype(int), 0, M)
    # ensure at least 1 for candidates with some rect or sample counts
    min_seed = np.minimum(pool_counts, np.ones_like(est_counts, dtype=int))
    est_counts = np.maximum(est_counts, min_seed)

    heap = [(-int(est_counts[i]), int(i)) for i in range(pool_N)]
    heapq.heapify(heap)

    covered = np.zeros(M, dtype=bool)
    in_pool = np.ones(pool_N, dtype=bool)
    selected_pool = []

    def marginal_count(pidx):
        # number of previously uncovered samples this candidate would cover
        return int(np.count_nonzero(pool_dom[:, pidx] & ~covered))

    selects = min(k, pool_N)
    # Lazy greedy: recompute only when popped
    while len(selected_pool) < selects and heap:
        est_neg, pidx = heapq.heappop(heap)
        if not in_pool[pidx]:
            continue
        est = -int(est_neg)
        true_c = marginal_count(pidx)
        if true_c == est:
            # accept
            selected_pool.append(int(pidx))
            in_pool[pidx] = False
            if true_c > 0:
                covered |= pool_dom[:, pidx]
        else:
            # update and re-push with true estimate
            heapq.heappush(heap, (-int(true_c), int(pidx)))

    # Fallback fill if not enough selected
    if len(selected_pool) < k:
        remaining = np.where(in_pool)[0]
        if remaining.size > 0:
            order_rem = remaining[np.argsort(-pool_counts[remaining])]
            need = k - len(selected_pool)
            for idx in order_rem[:need]:
                selected_pool.append(int(idx))
                in_pool[idx] = False
                covered |= pool_dom[:, idx]

    # Bounded local swap (single-swap attempts) using sample-estimated HV (counts -> volume)
    current_selected = list(selected_pool[:k])
    current_covered = np.zeros(M, dtype=bool)
    for s in current_selected:
        current_covered |= pool_dom[:, s]
    current_score = int(np.count_nonzero(current_covered))

    outsiders = np.where(~np.isin(np.arange(pool_N), current_selected))[0]
    outsider_order = outsiders[np.argsort(-pool_counts[outsiders])] if outsiders.size > 0 else np.array([], dtype=int)

    S = min(8, max(1, len(current_selected)))
    T = min(80, outsider_order.size)
    # try swapping each of up to S selected elements with top T outsiders
    for si in range(min(S, len(current_selected))):
        sel_idx = current_selected[si]
        # coverage without this selected element
        if len(current_selected) <= 1:
            base_cov = np.zeros(M, dtype=bool)
        else:
            base_cov = np.zeros(M, dtype=bool)
            for j, sidx in enumerate(current_selected):
                if j == si:
                    continue
                base_cov |= pool_dom[:, sidx]
        improved = False
        for cand in outsider_order[:T]:
            new_cov = base_cov | pool_dom[:, cand]
            new_score = int(np.count_nonzero(new_cov))
            if new_score > current_score:
                # perform swap
                old = current_selected[si]
                current_selected[si] = int(cand)
                current_covered = new_cov
                current_score = new_score
                # update outsider_order: remove cand, add old at the end
                outsider_order = outsider_order[outsider_order != cand]
                outsider_order = np.append(outsider_order, np.array([old], dtype=int))
                improved = True
                break
        if not improved:
            continue

    # Ensure exactly k items (pad if needed)
    if len(current_selected) < k:
        remaining = np.setdiff1d(np.arange(pool_N), current_selected, assume_unique=True)
        if remaining.size > 0:
            order_rem = remaining[np.argsort(-pool_counts[remaining])]
            need = k - len(current_selected)
            current_selected.extend(order_rem[:need].tolist())

    current_selected = current_selected[:k]
    final_indices = pool_idx_map[np.array(current_selected, dtype=int)]
    result = pts[final_indices].copy()
    return result


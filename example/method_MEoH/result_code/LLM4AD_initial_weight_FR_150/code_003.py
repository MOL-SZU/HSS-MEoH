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

    # reference handling and clamp
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

    # remove points with any coordinate > ref (they contribute nothing)
    valid_mask = np.all(pts <= ref + 1e-12, axis=1)
    if not np.all(valid_mask):
        idx_valid = np.nonzero(valid_mask)[0]
        pts = pts[idx_valid]
        N = pts.shape[0]
        if N == 0:
            return np.zeros((0, D), dtype=pts.dtype)
        if k >= N:
            return pts.copy()
        # keep mapping back to original indices
        orig_indices = idx_valid
    else:
        orig_indices = np.arange(N)

    # Pareto (non-dominated) prefilter (minimization)
    def pareto_filter(X):
        n = X.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            # any j that dominates i?
            leq = np.all(X <= X[i], axis=1)
            lt = np.any(X < X[i], axis=1)
            dominated_by = leq & lt
            dominated_by[i] = False
            if np.any(dominated_by):
                keep[i] = False
        return keep

    nd_mask = pareto_filter(pts)
    if not np.all(nd_mask):
        idx_nd = np.nonzero(nd_mask)[0]
        pts = pts[idx_nd]
        orig_indices = orig_indices[idx_nd]
        N = pts.shape[0]
        if N == 0:
            return np.zeros((0, D), dtype=pts.dtype)
        if k >= N:
            return pts.copy()

    # compute rectangular volumes (single-point surrogate)
    rect_sizes = np.prod(np.clip(ref - pts, a_min=0.0, a_max=None), axis=1)

    # adaptive Monte-Carlo sample size (balance quality & runtime)
    M = int(min(20000, max(2000, 300 * min(k, 40) + 200 * max(0, D - 3))))
    rng = np.random.default_rng(42)
    lower = pts.min(axis=0)
    upper = ref.copy()
    zero_range = (upper - lower) <= 0
    if np.any(zero_range):
        upper = upper.copy()
        upper[zero_range] = lower[zero_range] + 1e-6

    # generate shared samples in box [lower, upper]
    samples = rng.uniform(low=lower, high=upper, size=(M, D))

    # dominated mask: samples x pts -> True if sample >= point componentwise
    # compute in chunks if memory heavy
    try:
        dominated = np.all(samples[:, None, :] >= pts[None, :, :], axis=2)  # shape (M, N)
    except MemoryError:
        dominated = np.zeros((M, N), dtype=bool)
        chunk = max(1000, M // 10)
        for s in range(0, M, chunk):
            e = min(M, s + chunk)
            dominated[s:e] = np.all(samples[s:e, None, :] >= pts[None, :, :], axis=2)

    sample_counts = dominated.sum(axis=0)  # per-point dominated sample counts
    # diversity surrogate (distance to centroid)
    centroid = pts.mean(axis=0)
    diversity = np.linalg.norm(pts - centroid, axis=1)

    # normalize surrogates and blend into score
    s_counts = sample_counts.astype(float) / float(M) if M > 0 else np.zeros_like(sample_counts, dtype=float)
    vmax = rect_sizes.max() if rect_sizes.size > 0 else 1.0
    s_rect = rect_sizes / vmax if vmax > 0 else np.zeros_like(rect_sizes, dtype=float)
    dmax = diversity.max() if diversity.size > 0 else 1.0
    s_div = diversity / dmax if dmax > 0 else np.zeros_like(diversity, dtype=float)

    # weights
    w_s, w_v, w_d = 0.55, 0.30, 0.15
    scores = w_s * s_counts + w_v * s_rect + w_d * s_div

    # adaptive shortlist c * k
    c = 4
    min_pool = 80
    pool_size = int(min(max(min_pool, c * k), N))
    order = np.argsort(-scores)
    pool_idx = order[:pool_size]
    pool_pts = pts[pool_idx]
    pool_rect = rect_sizes[pool_idx]
    pool_dom = dominated[:, pool_idx]  # (M, pool_N)
    pool_counts = sample_counts[pool_idx]
    pool_scores = scores[pool_idx]
    pool_N = pool_pts.shape[0]

    if pool_N == 0:
        return pts[np.argsort(-rect_sizes)[:k]].copy()
    if k >= pool_N:
        return pool_pts.copy()

    # seeded lazy greedy heap: estimates scaled from pool_scores
    est_counts = np.clip((pool_scores * float(M)).astype(int), 0, M)
    est_counts = np.maximum(est_counts, np.minimum(pool_counts, np.ones_like(est_counts, dtype=int)))
    heap = [(-int(est_counts[i]), int(i)) for i in range(pool_N)]
    heapq.heapify(heap)

    selected_pool = []
    in_pool = np.ones(pool_N, dtype=bool)
    covered = np.zeros(M, dtype=bool)

    def marginal_count(pidx):
        # number of previously uncovered samples this candidate would newly cover
        return int(np.count_nonzero(pool_dom[:, pidx] & ~covered))

    selects = min(k, pool_N)
    while len(selected_pool) < selects and heap:
        est_neg, pidx = heapq.heappop(heap)
        if not in_pool[pidx]:
            continue
        est = -int(est_neg)
        true_gain = marginal_count(pidx)
        if true_gain == est:
            # accept
            selected_pool.append(int(pidx))
            in_pool[pidx] = False
            if true_gain > 0:
                covered |= pool_dom[:, pidx]
        else:
            # update estimate and re-push
            heapq.heappush(heap, (-int(true_gain), int(pidx)))

    # fallback fill if not enough selected
    if len(selected_pool) < k:
        remaining = np.where(in_pool)[0]
        if remaining.size > 0:
            order_rem = remaining[np.argsort(-pool_counts[remaining])]
            need = k - len(selected_pool)
            for idx in order_rem[:need]:
                selected_pool.append(int(idx))
                in_pool[idx] = False
                covered |= pool_dom[:, idx]

    selected_pool = selected_pool[:k]
    # sample-based covered mask for current selection
    current_covered = np.zeros(M, dtype=bool)
    for sidx in selected_pool:
        current_covered |= pool_dom[:, sidx]
    current_score = int(np.count_nonzero(current_covered))

    # bounded local swap (sample-driven) to improve coverage
    # consider S selected slots and top T outsiders by pool_counts
    S = min(max(1, k // 10), len(selected_pool))
    S = min(S, len(selected_pool), 8)
    outsiders = np.where(~np.isin(np.arange(pool_N), selected_pool))[0]
    outsider_order = outsiders[np.argsort(-pool_counts[outsiders])] if outsiders.size > 0 else np.array([], dtype=int)
    T = min(200, outsider_order.size)

    # try swapping first S selected positions (choose largest rect ones first)
    sel_by_rect_order = np.array(selected_pool)[np.argsort(-pool_rect[selected_pool])]
    sel_try = sel_by_rect_order[:S] if sel_by_rect_order.size > 0 else np.array([], dtype=int)

    for sel_idx in sel_try:
        # compute coverage without this selected element
        if len(selected_pool) <= 1:
            base_cov = np.zeros(M, dtype=bool)
        else:
            base_cov = np.zeros(M, dtype=bool)
            for s in selected_pool:
                if s == sel_idx:
                    continue
                base_cov |= pool_dom[:, s]
        improved = False
        # consider top T outsiders
        for cand in outsider_order[:T]:
            new_cov = base_cov | pool_dom[:, cand]
            new_score = int(np.count_nonzero(new_cov))
            if new_score > current_score:
                # perform swap
                # replace in selected_pool
                for ii, vv in enumerate(selected_pool):
                    if vv == sel_idx:
                        selected_pool[ii] = int(cand)
                        break
                current_covered = new_cov
                current_score = new_score
                # update outsider order (cheap re-eval: remove candidate)
                outsider_order = outsider_order[outsider_order != cand]
                outsider_order = np.append(outsider_order, np.array([sel_idx], dtype=int))
                improved = True
                break
        if improved:
            # continue with next selected slot
            continue

    # if still less than k (rare), pad by pool rect sizes
    if len(selected_pool) < k:
        remaining = np.setdiff1d(np.arange(pool_N), np.array(selected_pool, dtype=int), assume_unique=True)
        if remaining.size > 0:
            order_rem = remaining[np.argsort(-pool_rect[remaining])]
            need = k - len(selected_pool)
            for idx in order_rem[:need]:
                selected_pool.append(int(idx))

    selected_pool = selected_pool[:k]

    # map pool indices back to original indices in pts (which are after prefilter)
    final_rel = pool_idx[np.array(selected_pool, dtype=int)]
    final_orig = orig_indices[final_rel]  # map back to original indexing prior to pareto/valid filtering

    # ensure exact length k; if mapping produced duplicates or shortfall, fill from remaining by rect size over original pts
    final_orig = np.unique(final_orig, return_index=False)
    if final_orig.size < k:
        remaining_all = np.setdiff1d(orig_indices, final_orig, assume_unique=True)
        if remaining_all.size > 0:
            rem_rects = np.prod(np.clip(ref - points[remaining_all], a_min=0.0, a_max=None), axis=1)
            add = remaining_all[np.argsort(-rem_rects)][: (k - final_orig.size)]
            final_orig = np.concatenate([final_orig, add])

    final_orig = final_orig[:k]
    return np.asarray(points)[np.array(final_orig, dtype=int)].copy()


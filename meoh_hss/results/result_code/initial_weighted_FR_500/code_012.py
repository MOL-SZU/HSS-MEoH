import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return np.empty((0, D), dtype=float)

    # deterministic RNG
    rng = np.random.default_rng(123456)

    # handle empty input
    if N == 0:
        return np.zeros((k, D), dtype=float)

    # reference point
    if reference_point is None:
        ref = np.max(pts, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).reshape(-1)
        if ref.size != D:
            raise ValueError("reference_point must have same dimensionality as points")

    eps = 1e-12

    # vectorized nondominated filter (minimization)
    def nondominated_indices(X: np.ndarray):
        nloc = X.shape[0]
        if nloc == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(nloc, dtype=bool)
        # Use pairwise comparisons in blocks to reduce Python overhead
        for i in range(nloc):
            if dominated[i]:
                continue
            # j dominates i if X[j] <= X[i] for all dims and < in some
            le = np.all(X <= X[i], axis=1)
            lt = np.any(X < X[i], axis=1)
            dom = le & lt
            dom[i] = False
            if np.any(dom):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = nondominated_indices(pts)
    # shortlist pool: prefer Pareto if nontrivial
    if nd_idx.size >= min(3, k):
        pool_idx = nd_idx
    else:
        pool_idx = np.arange(N)
    pool = pts[pool_idx]
    P = pool.shape[0]

    # cheap box-volume proxy
    diffs_pool = np.maximum(ref - pool, eps)
    box_vol_pool = np.prod(diffs_pool, axis=1)

    # few weight vectors scalarization (deterministic)
    M = min(max(2, 2 * D), 6)  # keep small for speed
    raw = rng.random((M, D)) + 1e-9
    weights = raw / raw.sum(axis=1, keepdims=True)
    pool_ref_minus = np.maximum(ref - pool, 0.0)
    scalarized = pool_ref_minus.dot(weights.T)  # (P, M)
    mean_scalar = np.mean(scalarized, axis=1)
    ms_max = float(np.max(mean_scalar)) if mean_scalar.size > 0 else 1.0
    ms_max = max(ms_max, 1e-12)
    mean_scalar_norm = mean_scalar / ms_max
    score_pool = box_vol_pool * (0.2 + 0.8 * mean_scalar_norm)

    # order pool deterministically: nondominated first (they already in pool if chosen), then by score, tie by index
    order_pool = np.lexsort((pool_idx, -score_pool))
    pareto_size = nd_idx.size
    # adaptive shortlist size: scale with k and pareto_size but keep small for speed
    shortlist_size = int(min(P, max(3 * k, k + 20, int(1.0 * pareto_size))))
    shortlist_size = max(3, min(shortlist_size, P))
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()

    # if shortlist too small to fill k, expand deterministically using global box proxy
    if len(shortlist_idx) < k:
        diffs_all = np.maximum(ref - pts, eps)
        approx_all = np.prod(diffs_all, axis=1)
        order_all = np.argsort(-approx_all, kind='stable')
        take = min(max(50, k), N)
        shortlist_idx = order_all[:take].tolist()
        shortlist_idx = list(dict.fromkeys(shortlist_idx))  # uniq preserve order

    # Hypervolume helper (minimization)
    def hv_of_set(arr: np.ndarray) -> float:
        if arr is None or len(arr) == 0:
            return 0.0
        hv = pg.hypervolume(np.asarray(arr, dtype=float))
        return float(hv.compute(ref))

    # compute singleton hv for shortlist (cache)
    hv_single = {}
    for idx in shortlist_idx:
        hv_single[int(idx)] = hv_of_set(pts[[int(idx)]])

    # optional pairwise cache only when shortlist small
    hv_pair = {}
    if len(shortlist_idx) <= 30:
        ss = shortlist_idx
        for i in range(len(ss)):
            a = int(ss[i])
            for j in range(i + 1, len(ss)):
                b = int(ss[j])
                hv_pair[(a, b)] = hv_of_set(pts[[a, b]])
                hv_pair[(b, a)] = hv_pair[(a, b)]

    # CELF-style lazy greedy using cached singletons and timestamps
    heap = []
    for idx in shortlist_idx:
        gain = float(hv_single[int(idx)])  # marginal from empty
        heapq.heappush(heap, (-gain, int(idx), 0))

    selected_idx = []
    selected_set = set()
    selected_points = []
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))
    tol = 1e-12

    # Main CELF loop
    while len(selected_idx) < target and heap:
        neg_gain, idx, ts = heapq.heappop(heap)
        if idx in selected_set:
            continue
        if ts != len(selected_idx):
            # stale: recompute true marginal
            if len(selected_points) == 0:
                true_hv = hv_single[int(idx)]
            elif len(selected_points) == 1 and ((int(selected_idx[0]), int(idx)) in hv_pair):
                a, b = int(selected_idx[0]), int(idx)
                true_hv = hv_pair.get((a, b), hv_of_set(np.vstack([selected_points[0], pts[int(idx)]])))
            else:
                arr = np.vstack([np.asarray(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                true_hv = hv_of_set(arr)
            true_gain = float(true_hv - hv_selected)
            if true_gain < tol:
                true_gain = 0.0
            heapq.heappush(heap, (-true_gain, int(idx), len(selected_idx)))
            continue
        # accept
        selected_idx.append(int(idx))
        selected_set.add(int(idx))
        selected_points.append(pts[int(idx)].copy())
        hv_selected = hv_of_set(np.vstack(selected_points))

    # pad deterministically if needed using global box proxy
    if len(selected_idx) < k:
        diffs_un = np.maximum(ref - pts, eps)
        approx_un = np.prod(diffs_un, axis=1)
        order_un = np.argsort(-approx_un, kind='stable')
        for idx in order_un:
            if int(idx) in selected_set:
                continue
            selected_idx.append(int(idx))
            selected_set.add(int(idx))
            selected_points.append(pts[int(idx)].copy())
            if len(selected_idx) >= k:
                break
        hv_selected = hv_of_set(np.vstack(selected_points)) if selected_points else 0.0

    # prepare remaining candidates ordered by box proxy
    remaining_candidates = [int(i) for i in range(N) if i not in selected_set]
    if remaining_candidates:
        diffs_rem = np.maximum(ref - pts[remaining_candidates], eps)
        approx_rem = np.prod(diffs_rem, axis=1)
        rem_order = np.argsort(-approx_rem, kind='stable')
        remaining_candidates = [remaining_candidates[i] for i in rem_order]

    # bounded 1-for-1 deterministic swap refinement with incremental checks
    max_swap_iters = min(80, max(10, 6 * k))
    iter_count = 0
    improved = True
    max_cands_per_pos = min(20, max(5, k))

    while improved and iter_count < max_swap_iters:
        iter_count += 1
        improved = False
        # deterministic shuffle of positions to explore different slots
        positions = list(range(len(selected_idx)))
        rng.shuffle(positions)
        # deterministic limited candidate set
        cand_pool = [c for c in remaining_candidates if c not in selected_set]
        for si in positions:
            if improved:
                break
            old_idx = selected_idx[si]
            cands = cand_pool[:max_cands_per_pos]
            for cand in cands:
                # quick upper bound prune using singleton hv
                ub = hv_single.get(int(cand), None)
                if ub is None:
                    ub = hv_of_set(pts[[int(cand)]])
                    hv_single[int(cand)] = ub
                # if ub is too small compared to tiny improvement threshold, skip direct HV call
                if ub <= tol and ub <= hv_selected + tol:
                    continue
                temp_idx = selected_idx.copy()
                temp_idx[si] = int(cand)
                temp_arr = pts[np.array(temp_idx)]
                temp_hv = hv_of_set(temp_arr)
                if temp_hv > hv_selected + 1e-12:
                    # commit swap
                    selected_set.remove(int(old_idx))
                    selected_set.add(int(cand))
                    selected_idx[si] = int(cand)
                    selected_points[si] = pts[int(cand)].copy()
                    hv_selected = temp_hv
                    # update remaining_candidates
                    if int(old_idx) not in remaining_candidates:
                        remaining_candidates.insert(0, int(old_idx))
                    if int(cand) in remaining_candidates:
                        remaining_candidates.remove(int(cand))
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # ensure exactly k unique points preserving selection order
    final_idx = []
    seen = set()
    for idx in selected_idx:
        if idx not in seen:
            final_idx.append(int(idx))
            seen.add(int(idx))
        if len(final_idx) == k:
            break

    if len(final_idx) < k:
        # pad from global box proxy
        diffs_all = np.maximum(ref - pts, eps)
        approx_all = np.prod(diffs_all, axis=1)
        order_all = np.argsort(-approx_all, kind='stable')
        for idx in order_all:
            if int(idx) in seen:
                continue
            final_idx.append(int(idx))
            seen.add(int(idx))
            if len(final_idx) == k:
                break

    if len(final_idx) < k:
        # last resort: repeat best by box proxy
        best = int(np.argmax(np.prod(np.maximum(ref - pts, eps), axis=1)))
        while len(final_idx) < k:
            final_idx.append(best)

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


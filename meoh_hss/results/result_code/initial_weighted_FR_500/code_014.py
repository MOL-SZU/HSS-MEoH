import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    # determinism
    seed = 1
    rng = np.random.default_rng(seed)

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = pts.shape
    if not (1 <= k <= n):
        raise ValueError("k must be between 1 and number of points")

    # reference point
    if reference_point is None:
        ref = np.max(pts, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).reshape(-1)
        if ref.size != d:
            raise ValueError("reference_point must have same dimensionality as points")

    # fast nondominated filter (minimization assumed: smaller is better)
    def nondominated_indices(X: np.ndarray):
        N = X.shape[0]
        if N == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(N, dtype=bool)
        for i in range(N):
            if dominated[i]:
                continue
            # j dominates i if X[j] <= X[i] all dims and < in some
            comp_le = np.all(X <= X[i], axis=1)
            comp_lt = np.any(X < X[i], axis=1)
            comp = comp_le & comp_lt
            comp[i] = False
            if np.any(comp):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = nondominated_indices(pts)
    if nd_idx.size >= k:
        pool_idx = nd_idx
    else:
        pool_idx = np.arange(n)

    pool = pts[pool_idx]

    # cheap per-point box volume proxy
    eps = 1e-12
    diffs = np.maximum(ref - pool, eps)
    rect_vol = np.prod(diffs, axis=1)

    # deterministic small set of weight vectors for scalarization
    M = min(max(2, 2 * d), 8)
    # deterministic dirichlet-like weights via seeded RNG
    raw = rng.random((M, d)) + 1e-6
    weights = raw / raw.sum(axis=1, keepdims=True)  # shape (M, d)

    ref_minus_all = np.maximum(ref - pts, 0.0)  # (n, d)
    # scalarized scores: for pool entries only (vectorized)
    pool_ref_minus = np.maximum(ref - pool, 0.0)  # (len(pool), d)
    scalarized = pool_ref_minus.dot(weights.T)  # (len(pool), M)
    mean_scalar = np.mean(scalarized, axis=1)
    # normalize
    ms_max = max(1e-12, float(np.max(mean_scalar)))
    mean_scalar_norm = mean_scalar / ms_max
    score_pool = rect_vol * (0.2 + 0.8 * mean_scalar_norm)

    # Build ordered list: prioritize nondominated pool members by score, then remaining by score
    # Map pool_idx back to original indices for lexsort tie-breaking
    order_pool = np.lexsort((pool_idx, -score_pool))
    # shortlist size proportional to k but bounded
    shortlist_size = int(min(pool.shape[0], max(4 * k, k + 40)))
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()

    # if shortlist smaller than k, extend to all points (rare)
    if len(shortlist_idx) < k:
        # build a global score for all points to deterministically pad
        diffs_global = np.maximum(ref - pts, eps)
        approx_vol_global = np.prod(diffs_global, axis=1)
        # deterministic order by approx_vol
        order_global = np.argsort(-approx_vol_global, kind='stable')
        shortlist_idx = order_global[:min(max(50, k), n)].tolist()

    # helper hv function (minimization)
    def hv_of_set(arr: np.ndarray) -> float:
        if arr is None or len(arr) == 0:
            return 0.0
        hv = pg.hypervolume(np.asarray(arr, dtype=float))
        return float(hv.compute(ref))

    # precompute singleton HVs for shortlist (cached)
    hv_single = {}
    for idx in shortlist_idx:
        hv_single[int(idx)] = hv_of_set(pts[[int(idx)]])

    # CELF-style lazy greedy using cached singleton HVs and timestamps
    heap = []
    for idx in shortlist_idx:
        gain = float(hv_single[int(idx)])  # when selected is empty
        # push negative gain for max-heap behavior
        heapq.heappush(heap, (-gain, int(idx), 0))

    selected_idx = []
    selected_set = set()
    selected_points = []
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))
    # main CELF loop
    while len(selected_idx) < target and heap:
        neg_gain, idx, ts = heapq.heappop(heap)
        if idx in selected_set:
            continue
        if ts != len(selected_idx):
            # stale gain; recompute exact marginal
            if len(selected_points) == 0:
                true_hv = hv_single[int(idx)]
            else:
                arr = np.vstack([np.array(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                true_hv = hv_of_set(arr)
            true_gain = float(true_hv - hv_selected)
            heapq.heappush(heap, (-true_gain, int(idx), len(selected_idx)))
            continue
        # accept
        selected_idx.append(int(idx))
        selected_set.add(int(idx))
        selected_points.append(pts[int(idx)].reshape(1, -1)[0])
        hv_selected = hv_of_set(np.vstack(selected_points))

    # if still fewer than k, extend deterministically from ordered by global approx_vol
    if len(selected_idx) < k:
        diffs_un = np.maximum(ref - pts, eps)
        approx_un = np.prod(diffs_un, axis=1)
        order_un = np.argsort(-approx_un, kind='stable')
        for idx in order_un:
            if int(idx) in selected_set:
                continue
            selected_idx.append(int(idx))
            selected_set.add(int(idx))
            selected_points.append(pts[int(idx)])
            if len(selected_idx) >= k:
                break
        hv_selected = hv_of_set(np.vstack(selected_points)) if selected_points else 0.0

    # bounded 1-for-1 local swaps (only recompute HV for the swapped set)
    max_swap_iters = min(200, max(20, 8 * k))
    iter_count = 0
    improved = True
    # candidate pool: deterministically order remaining indices by score (use global approx if not in pool)
    remaining_candidates = [int(i) for i in range(n) if i not in selected_set]
    # deterministic shuffle but reproducible
    while improved and iter_count < max_swap_iters:
        improved = False
        iter_count += 1
        sel_positions = list(range(len(selected_idx)))
        rng.shuffle(sel_positions)
        rng.shuffle(remaining_candidates)
        for si in sel_positions:
            if improved:
                break
            old_idx = selected_idx[si]
            for cand in remaining_candidates:
                if cand in selected_set:
                    continue
                # build swapped set
                temp_idx = selected_idx.copy()
                temp_idx[si] = int(cand)
                temp_arr = pts[np.array(temp_idx)]
                temp_hv = hv_of_set(temp_arr)
                if temp_hv > hv_selected + 1e-12:
                    # commit swap
                    selected_idx[si] = int(cand)
                    selected_set.remove(int(old_idx))
                    selected_set.add(int(cand))
                    selected_points[si] = pts[int(cand)]
                    hv_selected = temp_hv
                    # update remaining_candidates: add old back, remove cand
                    if int(old_idx) not in remaining_candidates:
                        remaining_candidates.append(int(old_idx))
                    if int(cand) in remaining_candidates:
                        remaining_candidates.remove(int(cand))
                    improved = True
                    break
        if not improved:
            break

    # Ensure exactly k unique points: dedupe preserving order, then pad if needed
    final_idx = []
    seen = set()
    for idx in selected_idx:
        if idx not in seen:
            final_idx.append(int(idx))
            seen.add(int(idx))
        if len(final_idx) == k:
            break

    if len(final_idx) < k:
        # pad deterministically from remaining by approx global box volume
        remaining = [i for i in range(n) if i not in seen]
        if remaining:
            diffs_rem = np.maximum(ref - pts[remaining], eps)
            approx_rem = np.prod(diffs_rem, axis=1)
            order_rem = np.argsort(-approx_rem, kind='stable')
            for pos in order_rem:
                final_idx.append(int(remaining[int(pos)]))
                if len(final_idx) == k:
                    break
        # as last resort, repeat last selected
        if len(final_idx) < k:
            if len(final_idx) == 0:
                # create k copies of mean clipped to ref
                fallback = np.minimum(np.mean(pts, axis=0), ref)
                return np.tile(fallback, (k, 1)).astype(float)
            last = pts[final_idx[-1]]
            while len(final_idx) < k:
                final_idx.append(final_idx[-1])

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


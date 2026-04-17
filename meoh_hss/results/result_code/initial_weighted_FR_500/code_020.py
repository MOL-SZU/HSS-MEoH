import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    # Deterministic RNG seed (different from original)
    seed = 12345
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

    # fast nondominated filter (minimization assumed)
    def nondominated_indices(X: np.ndarray):
        N = X.shape[0]
        if N == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(N, dtype=bool)
        # simple O(N^2) but deterministic and fine for modest N
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

    # cheap per-point box volume proxy (use slightly larger eps)
    eps = 1e-12
    diffs = np.maximum(ref - pool, eps)
    rect_vol = np.prod(diffs, axis=1)  # box volume proxy

    # deterministic small set of weight vectors for scalarization (tuned differently)
    M = min(max(3, d + 1), 12)  # a few more directions when d is small, but bounded
    raw = rng.random((M, d)) + 1e-8
    weights = raw / raw.sum(axis=1, keepdims=True)  # shape (M, d)

    # scalarized scores: for pool entries only (vectorized)
    pool_ref_minus = np.maximum(ref - pool, 0.0)  # (len(pool), d)
    scalarized = pool_ref_minus.dot(weights.T)  # (len(pool), M)
    mean_scalar = np.mean(scalarized, axis=1)
    # normalize scalarized to [0,1]
    ms_max = max(1e-12, float(np.max(mean_scalar)))
    mean_scalar_norm = mean_scalar / ms_max

    # New score: stronger emphasis on box volume, mild multiplicative boost from scalarization
    # Use exponent to reduce influence of very large boxes and avoid overflow
    rect_power = np.power(np.maximum(rect_vol, 1e-300), 0.85)  # soften large volumes
    score_pool = rect_power * (0.6 + 0.4 * mean_scalar_norm)  # blend with scalarization

    # deterministic stable ordering: higher score first, break ties by smaller original index
    # lexsort uses last key as primary, so use (-score, pool_idx)
    order_pool = np.lexsort((pool_idx, -score_pool))
    # shortlist size: tighter than original to reduce exact HV calls
    shortlist_size = int(min(pool.shape[0], max(3 * k, k + 20)))
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()

    # if shortlist smaller than k, extend to a deterministic global shortlist by approx box volume
    if len(shortlist_idx) < k:
        diffs_global = np.maximum(ref - pts, eps)
        approx_vol_global = np.prod(diffs_global, axis=1)
        order_global = np.argsort(-approx_vol_global, kind='stable')
        shortlist_idx = order_global[:min(max(40, k), n)].tolist()

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

    # If shortlist is small, precompute pairwise HVs to get slightly better initial ordering
    pair_hv = {}
    if len(shortlist_idx) <= 40:
        # compute hv of pairs (i,j) and store hv({i,j}) for quick evaluation in local steps
        sidx = shortlist_idx
        for i in range(len(sidx)):
            ii = int(sidx[i])
            for j in range(i + 1, len(sidx)):
                jj = int(sidx[j])
                hv_ij = hv_of_set(pts[[ii, jj]])
                pair_hv[(ii, jj)] = hv_ij
                pair_hv[(jj, ii)] = hv_ij

    # Initialize CELF-style max-heap using singleton hv as initial marginal when selected is empty
    heap = []
    for idx in shortlist_idx:
        gain = float(hv_single[int(idx)])  # marginal gain from empty set
        heapq.heappush(heap, (-gain, int(idx), 0))

    selected_idx = []
    selected_set = set()
    selected_points = []
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))

    # CELF loop with cached singleton hv and lazy timestamps
    while len(selected_idx) < target and heap:
        neg_gain, idx, ts = heapq.heappop(heap)
        if idx in selected_set:
            continue
        if ts != len(selected_idx):
            # stale entry: recompute true marginal with current selected set
            if len(selected_points) == 0:
                true_hv = hv_single[int(idx)]
            else:
                arr = np.vstack([np.array(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                true_hv = hv_of_set(arr)
            true_gain = float(true_hv - hv_selected)
            # push updated with current timestamp
            heapq.heappush(heap, (-true_gain, int(idx), len(selected_idx)))
            continue
        # accept the candidate
        selected_idx.append(int(idx))
        selected_set.add(int(idx))
        selected_points.append(pts[int(idx)].reshape(1, -1)[0])
        hv_selected = hv_of_set(np.vstack(selected_points))

    # If still fewer than k, pad deterministically by approx box volume ordering
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

    # Bounded 1-for-1 local swaps with limited iterations; use pair_hv if available to speed small computations
    max_swap_iters = min(150, max(10, 6 * k))
    iter_count = 0
    improved = True
    remaining_candidates = [int(i) for i in range(n) if i not in selected_set]
    # deterministic reproducible shuffles
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
                # build swapped set indices
                temp_idx = selected_idx.copy()
                temp_idx[si] = int(cand)
                temp_arr = pts[np.array(temp_idx)]
                # if we have pairwise cached and only two changed relative to others, we could use it,
                # but here we recompute the HV for the full swapped set (bounded by swaps)
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
        # as last resort, repeat last selected or fallback mean clipped
        if len(final_idx) < k:
            if len(final_idx) == 0:
                fallback = np.minimum(np.mean(pts, axis=0), ref)
                return np.tile(fallback, (k, 1)).astype(float)
            last = final_idx[-1]
            while len(final_idx) < k:
                final_idx.append(last)

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


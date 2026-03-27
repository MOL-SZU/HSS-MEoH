import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    pts = np.asarray(points, dtype=float)
    # normalize shapes: allow 1D single point input
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return np.empty((0, D), dtype=float)
    if N == 0:
        return np.zeros((k, D), dtype=float)

    # deterministic RNG
    seed = 424242
    rng = np.random.default_rng(seed)

    # reference point
    if reference_point is None:
        ref = np.max(pts, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).reshape(-1)
        if ref.size != D:
            raise ValueError("reference_point must have same dimensionality as points")

    eps = 1e-12
    tol = 1e-12

    # vectorized nondominated filter (minimization assumed)
    def nondominated_indices(X: np.ndarray):
        M = X.shape[0]
        if M == 0:
            return np.array([], dtype=int)
        A = X[:, None, :]  # (M,1,D)
        B = X[None, :, :]  # (1,M,D)
        le = np.all(A <= B, axis=2)  # (M,M)
        lt = np.any(A < B, axis=2)   # (M,M)
        dominates = le & lt
        dominated = np.any(dominates, axis=0)
        return np.where(~dominated)[0]

    nd_idx = nondominated_indices(pts)
    pareto_size = nd_idx.size

    # pool selection: prefer Pareto if useful but keep pool small
    if pareto_size >= min(3, max(1, k // 2)):
        pool_idx = nd_idx
    else:
        pool_idx = np.arange(N)
    pool = pts[pool_idx]
    P = pool.shape[0]

    # cheap proxies: box volume and few-weight scalarization (deterministic)
    diffs_pool = np.maximum(ref - pool, eps)
    box_vol_pool = np.prod(diffs_pool, axis=1)

    Mw = min(max(2, 2 * D), 6)
    raw = rng.random((Mw, D)) + 1e-9
    weights = raw / raw.sum(axis=1, keepdims=True)
    pool_ref_minus = np.maximum(ref - pool, 0.0)
    scalarized = pool_ref_minus.dot(weights.T)  # (P, Mw)
    mean_scalar = np.mean(scalarized, axis=1)
    ms_max = float(np.max(mean_scalar)) if mean_scalar.size > 0 else 1.0
    ms_max = max(ms_max, 1e-12)
    mean_scalar_norm = mean_scalar / ms_max

    alpha = 0.25
    score_pool = (box_vol_pool ** (1.0 - 0.05)) * (alpha + (1.0 - alpha) * mean_scalar_norm)

    # deterministic ordering: score desc then original index asc
    order_pool = np.lexsort((pool_idx, -score_pool))

    # adaptive O(k) shortlist (compact to speed up)
    shortlist_size = int(min(P, max(3, 3 * k, k + 12, min(60, 4 * pareto_size))))
    shortlist_size = max(3, min(shortlist_size, P))
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()

    # ensure capacity to fill k: expand by global box proxy if needed
    if len(shortlist_idx) < k:
        diffs_all = np.maximum(ref - pts, eps)
        approx_all = np.prod(diffs_all, axis=1)
        order_all = np.argsort(-approx_all, kind='stable')
        take = min(max(30, k), N)
        shortlist_idx = order_all[:take].tolist()

    # unique deterministic shortlist preserving first occurrence
    seen_tmp = set()
    uniq_shortlist_idx = []
    for ii in shortlist_idx:
        i = int(ii)
        if i not in seen_tmp:
            uniq_shortlist_idx.append(i)
            seen_tmp.add(i)
    shortlist_idx = uniq_shortlist_idx
    L = len(shortlist_idx)

    # precompute global box proxy for fallback and ordering
    diffs_all = np.maximum(ref - pts, eps)
    box_vol_all = np.prod(diffs_all, axis=1)

    # hv call budget scaled by k and shortlist size
    hv_calls = 0
    max_hv_calls = max(120, int(20 * k) + 5 * L)

    def hv_of_set(arr: np.ndarray) -> float:
        nonlocal hv_calls
        if arr is None or len(arr) == 0:
            return 0.0
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        hv_calls += 1
        return float(pg.hypervolume(a).compute(ref))

    # singleton cache: compute until budget reached; if budget exceeded, use box proxy fallback
    hv_single = {}
    box_proxy = {int(i): float(box_vol_all[int(i)]) for i in shortlist_idx}
    for idx in shortlist_idx:
        ii = int(idx)
        if hv_calls >= max_hv_calls:
            hv_single[ii] = float(box_proxy.get(ii, 0.0))
        else:
            hv_single[ii] = hv_of_set(pts[[ii]])

    # selective pairwise cache if shortlist is tiny
    pair_hv = {}
    if L <= 30:
        ss = shortlist_idx
        for i in range(len(ss)):
            a = int(ss[i])
            for j in range(i + 1, len(ss)):
                b = int(ss[j])
                if hv_calls >= max_hv_calls:
                    pair_hv[(a, b)] = hv_single.get(a, 0.0) + hv_single.get(b, 0.0)
                    pair_hv[(b, a)] = pair_hv[(a, b)]
                else:
                    v = hv_of_set(pts[[a, b]])
                    pair_hv[(a, b)] = v
                    pair_hv[(b, a)] = v

    # CELF lazy greedy initialization
    heap = []
    for idx in shortlist_idx:
        ii = int(idx)
        g = float(hv_single.get(ii, 0.0))
        heapq.heappush(heap, (-g, ii, 0))

    selected_idx = []
    selected_set = set()
    selected_points = []  # list of np arrays
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))

    # Main CELF loop with pruning and budget awareness
    while len(selected_idx) < target and heap:
        if hv_calls >= max_hv_calls:
            break
        neg_gain, idx, ts = heapq.heappop(heap)
        idx = int(idx)
        if idx in selected_set:
            continue
        # tight upper bound: either singleton hv cached or box proxy
        ub = max(hv_single.get(idx, 0.0), box_proxy.get(idx, 0.0))
        if ub <= hv_selected + tol and ts != len(selected_idx):
            # unlikely to improve
            heapq.heappush(heap, (0.0, idx, len(selected_idx)))
            continue
        if ts != len(selected_idx):
            # stale entry: recompute marginal if budget allows
            if hv_calls >= max_hv_calls:
                est_gain = max(0.0, hv_single.get(idx, 0.0) - hv_selected)
                heapq.heappush(heap, (-est_gain, idx, len(selected_idx)))
                continue
            # compute true hv of selected + candidate (use pair cache when possible)
            if len(selected_points) == 0:
                true_hv = hv_single.get(idx, hv_of_set(pts[[idx]]))
            elif len(selected_points) == 1:
                only = int(selected_idx[0])
                if (only, idx) in pair_hv:
                    true_hv = pair_hv[(only, idx)]
                else:
                    arr = np.vstack([np.asarray(selected_points, dtype=float), pts[idx].reshape(1, -1)])
                    true_hv = hv_of_set(arr)
            else:
                arr = np.vstack([np.asarray(selected_points, dtype=float), pts[idx].reshape(1, -1)])
                true_hv = hv_of_set(arr)
            true_gain = float(max(0.0, true_hv - hv_selected))
            if true_gain < tol:
                true_gain = 0.0
            heapq.heappush(heap, (-true_gain, idx, len(selected_idx)))
            continue
        # accept candidate
        selected_idx.append(idx)
        selected_set.add(idx)
        selected_points.append(pts[idx].copy())
        if hv_calls < max_hv_calls:
            hv_selected = hv_of_set(np.vstack(selected_points))
        else:
            hv_selected = sum(hv_single.get(i, 0.0) for i in selected_idx)

    # pad deterministically by global box proxy if needed
    if len(selected_idx) < k:
        order_un = np.argsort(-box_vol_all, kind='stable')
        for ii in order_un:
            i = int(ii)
            if i in selected_set:
                continue
            selected_idx.append(i)
            selected_set.add(i)
            selected_points.append(pts[i].copy())
            if len(selected_idx) >= k:
                break
        if hv_calls < max_hv_calls and selected_points:
            hv_selected = hv_of_set(np.vstack(selected_points))

    # prepare remaining candidates ordered by box proxy
    remaining_candidates = [int(i) for i in range(N) if int(i) not in selected_set]
    if remaining_candidates:
        diffs_rem = np.maximum(ref - pts[remaining_candidates], eps)
        approx_rem = np.prod(diffs_rem, axis=1)
        rem_order = np.argsort(-approx_rem, kind='stable')
        remaining_candidates = [remaining_candidates[i] for i in rem_order]

    # bounded deterministic 1-for-1 swap refinement (small effort)
    max_swap_iters = min(30, max(4, 4 * k))
    iter_count = 0
    improved = True
    max_cands_per_pos = min(12, max(4, k))

    while improved and iter_count < max_swap_iters and hv_calls < max_hv_calls:
        iter_count += 1
        improved = False
        # deterministic traversal of selected positions
        for si in range(len(selected_idx)):
            if hv_calls >= max_hv_calls:
                break
            old_idx = selected_idx[si]
            for cand in remaining_candidates[:max_cands_per_pos]:
                cand = int(cand)
                if cand in selected_set:
                    continue
                # cheap singleton bound (compute if missing but budget-aware)
                ub = hv_single.get(cand, None)
                if ub is None:
                    if hv_calls >= max_hv_calls:
                        ub = box_proxy.get(cand, 0.0)
                    else:
                        ub = hv_of_set(pts[[cand]])
                        hv_single[cand] = ub
                if ub <= hv_selected + tol:
                    continue
                temp_idx = selected_idx.copy()
                temp_idx[si] = cand
                if hv_calls >= max_hv_calls:
                    continue
                temp_arr = pts[np.array(temp_idx)]
                temp_hv = hv_of_set(temp_arr)
                if temp_hv > hv_selected + 1e-12:
                    # commit swap
                    selected_idx[si] = cand
                    selected_set.remove(old_idx)
                    selected_set.add(cand)
                    selected_points[si] = pts[cand].copy()
                    hv_selected = temp_hv
                    # update remaining_candidates deterministically
                    if old_idx not in remaining_candidates:
                        remaining_candidates.insert(0, old_idx)
                    if cand in remaining_candidates:
                        remaining_candidates.remove(cand)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # final unique order-preserving selection and deterministic padding to exactly k
    final_idx = []
    seen = set()
    for idx in selected_idx:
        if idx not in seen:
            final_idx.append(int(idx))
            seen.add(idx)
        if len(final_idx) == k:
            break

    if len(final_idx) < k:
        # take from remaining_candidates first
        for idx in remaining_candidates:
            if idx in seen:
                continue
            final_idx.append(int(idx))
            seen.add(idx)
            if len(final_idx) == k:
                break

    if len(final_idx) < k:
        # fallback to global box proxy ordering
        order_all = np.argsort(-box_vol_all, kind='stable')
        for ii in order_all:
            i = int(ii)
            if i in seen:
                continue
            final_idx.append(i)
            seen.add(i)
            if len(final_idx) == k:
                break

    if len(final_idx) < k:
        best = int(np.argmax(box_vol_all)) if N > 0 else 0
        while len(final_idx) < k:
            final_idx.append(best)

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


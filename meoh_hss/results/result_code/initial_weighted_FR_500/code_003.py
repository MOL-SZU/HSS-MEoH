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
    if N == 0:
        # nothing to pick from
        return np.zeros((k, D), dtype=float)

    # deterministic RNG
    seed = 2026
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

    # vectorized nondominated filter (minimization)
    def nondominated_indices(X: np.ndarray):
        M = X.shape[0]
        if M == 0:
            return np.array([], dtype=int)
        # A dominates B if all(A <= B) and any(A < B)
        A = X[:, None, :]  # (M,1,D)
        B = X[None, :, :]  # (1,M,D)
        le = np.all(A <= B, axis=2)  # (M,M)
        lt = np.any(A < B, axis=2)   # (M,M)
        dominates = le & lt
        dominated = np.any(dominates, axis=0)
        return np.where(~dominated)[0]

    nd_idx = nondominated_indices(pts)
    pareto_size = nd_idx.size

    # pool: prefer Pareto when meaningful
    if pareto_size >= min(4, k):
        pool_idx = nd_idx
    else:
        pool_idx = np.arange(N)
    pool = pts[pool_idx]
    P = pool.shape[0]

    # vectorized proxies: rect (box) volume and few-weight scalarization
    diffs_pool = np.maximum(ref - pool, eps)  # (P,D)
    box_vol_pool = np.prod(diffs_pool, axis=1)

    diffs_all = np.maximum(ref - pts, eps)
    box_vol_all = np.prod(diffs_all, axis=1)

    # few deterministic weight probes
    Mw = min(max(2, 2 * D), 6)
    raw = rng.random((Mw, D)) + 1e-9
    weights = raw / raw.sum(axis=1, keepdims=True)
    pool_ref_minus = np.maximum(ref - pool, 0.0)
    scalarized = pool_ref_minus.dot(weights.T)  # (P, Mw)
    mean_scalar = np.mean(scalarized, axis=1)
    ms_max = float(np.max(mean_scalar)) if mean_scalar.size > 0 else 1.0
    ms_max = max(ms_max, 1e-12)
    mean_scalar_norm = mean_scalar / ms_max

    alpha = 0.20
    score_pool = box_vol_pool * (alpha + (1.0 - alpha) * mean_scalar_norm)

    # deterministic ordering: desc score, tie-break by original index asc
    order_pool = np.lexsort((pool_idx, -score_pool))

    # compact O(k) shortlist (keeps runtime small)
    shortlist_size = int(min(P, max(3 * k, k + 12, 40)))
    shortlist_size = max(3, shortlist_size)
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()

    # ensure we can fill k by expanding via global box proxy if needed
    if len(shortlist_idx) < k:
        order_all = np.argsort(-box_vol_all, kind='stable')
        take = min(max(40, k), N)
        shortlist_idx = order_all[:take].tolist()

    # unique deterministic shortlist preserving order
    seen_tmp = set()
    uniq_shortlist_idx = []
    for ii in shortlist_idx:
        i = int(ii)
        if i not in seen_tmp:
            uniq_shortlist_idx.append(i)
            seen_tmp.add(i)
    shortlist_idx = uniq_shortlist_idx
    L = len(shortlist_idx)

    # HV-call budget: kept modest to favor speed
    hv_calls = 0
    max_hv_calls = max(120, 25 * k)

    def hv_of_set(arr: np.ndarray) -> float:
        nonlocal hv_calls
        if arr is None or len(arr) == 0:
            return 0.0
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        hv_calls += 1
        return float(pg.hypervolume(a).compute(ref))

    # precompute singleton HVs (cache), fallback to box proxy if budget exceeded
    hv_single = {}
    for idx in shortlist_idx:
        if hv_calls >= max_hv_calls:
            hv_single[int(idx)] = float(box_vol_all[int(idx)])
        else:
            hv_single[int(idx)] = hv_of_set(pts[[int(idx)]])

    # small pairwise cache only when shortlist very small
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

    # CELF lazy greedy initialization with timestamps
    heap = []
    for idx in shortlist_idx:
        gain = float(hv_single.get(int(idx), 0.0))
        heapq.heappush(heap, (-gain, int(idx), 0))

    selected_idx = []
    selected_mask = np.zeros(N, dtype=bool)
    selected_points = []
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))

    # main CELF loop: recompute marginal only when promising and within budget
    while len(selected_idx) < target and heap:
        if hv_calls >= max_hv_calls:
            break
        neg_gain, idx, ts = heapq.heappop(heap)
        if selected_mask[int(idx)]:
            continue
        # if stale entry, recompute true marginal if budget allows
        if ts != len(selected_idx):
            if hv_calls >= max_hv_calls:
                est_gain = max(0.0, hv_single.get(int(idx), 0.0) - hv_selected)
                heapq.heappush(heap, (-est_gain, int(idx), len(selected_idx)))
                continue
            # compute true hv of selected + candidate using caches when possible
            if len(selected_points) == 0:
                true_hv = hv_single.get(int(idx), hv_of_set(pts[[int(idx)]]))
            elif len(selected_points) == 1:
                only = int(selected_idx[0])
                if (only, int(idx)) in pair_hv:
                    true_hv = pair_hv[(only, int(idx))]
                else:
                    arr = np.vstack([np.asarray(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                    true_hv = hv_of_set(arr)
            else:
                arr = np.vstack([np.asarray(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                true_hv = hv_of_set(arr)
            true_gain = float(true_hv - hv_selected)
            if true_gain < tol:
                true_gain = 0.0
            heapq.heappush(heap, (-true_gain, int(idx), len(selected_idx)))
            continue
        # accept candidate
        selected_idx.append(int(idx))
        selected_mask[int(idx)] = True
        selected_points.append(pts[int(idx)].copy())
        if hv_calls < max_hv_calls:
            hv_selected = hv_of_set(np.vstack(selected_points))
        else:
            hv_selected = sum(hv_single.get(i, 0.0) for i in selected_idx)

    # pad deterministically if needed using global box proxy (fast)
    if len(selected_idx) < k:
        order_all = np.argsort(-box_vol_all, kind='stable')
        for idx in order_all:
            if selected_mask[int(idx)]:
                continue
            selected_idx.append(int(idx))
            selected_mask[int(idx)] = True
            selected_points.append(pts[int(idx)].copy())
            if len(selected_idx) >= k:
                break
        if hv_calls < max_hv_calls and selected_points:
            hv_selected = hv_of_set(np.vstack(selected_points))

    # prepare remaining candidates ordered by box proxy (deterministic)
    remaining_candidates = np.where(~selected_mask)[0].tolist()
    remaining_candidates.sort(key=lambda x: -box_vol_all[int(x)])

    # tiny bounded deterministic swap pass to catch easy improvements
    if hv_calls < max_hv_calls and len(selected_idx) > 0:
        max_swap_iters = min(30, max(4, 4 * k))
        iter_count = 0
        improved = True
        max_cands_per_pos = min(12, max(3, k))
        top_remaining = remaining_candidates[:min(len(remaining_candidates), 6 * k + 20)]
        while improved and iter_count < max_swap_iters and hv_calls < max_hv_calls:
            iter_count += 1
            improved = False
            for si in range(len(selected_idx)):
                if hv_calls >= max_hv_calls:
                    break
                old_idx = selected_idx[si]
                for cand in top_remaining[:max_cands_per_pos]:
                    if selected_mask[int(cand)]:
                        continue
                    # lazy singleton upper bound (already cached if possible)
                    ub = hv_single.get(int(cand), None)
                    if ub is None:
                        if hv_calls >= max_hv_calls:
                            ub = box_vol_all[int(cand)]
                        else:
                            ub = hv_of_set(pts[[int(cand)]])
                            hv_single[int(cand)] = ub
                    if ub <= hv_selected + tol:
                        continue
                    # compute HV after swap
                    temp_idx = selected_idx.copy()
                    temp_idx[si] = int(cand)
                    if hv_calls >= max_hv_calls:
                        continue
                    temp_arr = pts[np.array(temp_idx)]
                    temp_hv = hv_of_set(temp_arr)
                    if temp_hv > hv_selected + 1e-12:
                        # commit swap
                        selected_idx[si] = int(cand)
                        selected_mask[int(old_idx)] = False
                        selected_mask[int(cand)] = True
                        selected_points[si] = pts[int(cand)].copy()
                        hv_selected = temp_hv
                        # update top_remaining deterministically
                        if int(old_idx) not in top_remaining:
                            top_remaining.append(int(old_idx))
                        if int(cand) in top_remaining:
                            top_remaining.remove(int(cand))
                        top_remaining.sort(key=lambda x: -box_vol_all[int(x)])
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # final deterministic dedupe and pad to exactly k
    final_idx = []
    seen = set()
    for idx in selected_idx:
        if idx not in seen:
            final_idx.append(int(idx))
            seen.add(idx)
        if len(final_idx) == k:
            break

    if len(final_idx) < k:
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
        for idx in order_all:
            i = int(idx)
            if i in seen:
                continue
            final_idx.append(i)
            seen.add(i)
            if len(final_idx) == k:
                break

    if len(final_idx) < k:
        # last resort: repeat best proxy point
        best = int(np.argmax(box_vol_all)) if N > 0 else 0
        while len(final_idx) < k:
            final_idx.append(best)

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


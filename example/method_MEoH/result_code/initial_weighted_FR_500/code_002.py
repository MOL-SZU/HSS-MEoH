import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    pts = np.asarray(points, dtype=float)
    # allow 1D single point
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

    # deterministic RNG for scalar probes
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

    # vectorized nondominated filter (assume minimization)
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

    # build pool: prefer pareto when meaningful
    if pareto_size >= min(3, max(1, k // 2)):
        pool_idx = nd_idx
    else:
        pool_idx = np.arange(N)
    pool = pts[pool_idx]
    P = pool.shape[0]

    # precompute diffs and box volumes (vectorized)
    diffs_all = np.maximum(ref - pts, eps)    # (N, D)
    box_vol_all = np.prod(diffs_all, axis=1)

    diffs_pool = np.maximum(ref - pool, eps)
    box_vol_pool = np.prod(diffs_pool, axis=1)

    # few deterministic weight probes (cheap and informative)
    Mw = min(max(2, 2 * D), 6)
    raw = rng.random((Mw, D)) + 1e-12
    weights = raw / raw.sum(axis=1, keepdims=True)
    scalarized = np.maximum(ref - pool, 0.0).dot(weights.T)  # (P, Mw)
    scalar_score = np.mean(scalarized, axis=1)
    max_ss = float(np.max(scalar_score)) if scalar_score.size > 0 else 1.0
    max_ss = max(max_ss, 1e-12)
    scalar_norm = scalar_score / max_ss

    # blended compact score to form shortlist
    alpha = 0.25
    score_pool = box_vol_pool * (alpha + (1.0 - alpha) * scalar_norm)

    # deterministic ordering: score desc then original index asc
    order_pool = np.lexsort((pool_idx, -score_pool))

    # adaptive O(k) shortlist (compact, tuned for speed)
    shortlist_size = int(min(P, max(3, 2 * k, k + 10, 30)))
    shortlist_size = max(3, min(shortlist_size, P))
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()

    # ensure can fill k by expanding via global box proxy if needed
    if len(shortlist_idx) < k:
        order_all = np.argsort(-box_vol_all, kind='stable')
        take = min(max(20, k), N)
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

    # hv-call budget tuned for speed while retaining quality
    hv_calls = 0
    max_hv_calls = max(80, int(12 * k) + 4 * L)

    def hv_of_set_by_idx(idxs):
        nonlocal hv_calls
        if len(idxs) == 0:
            return 0.0
        a = pts[np.asarray(idxs, dtype=int)]
        hv_calls += 1
        return float(pg.hypervolume(a).compute(ref))

    # singleton cache (compute until budget, else fallback to box proxy)
    hv_single = {}
    for idx in shortlist_idx:
        ii = int(idx)
        if hv_calls >= max_hv_calls:
            hv_single[ii] = float(box_vol_all[ii])
        else:
            hv_single[ii] = hv_of_set_by_idx([ii])

    # selective pairwise cache only for tiny shortlist to reduce repeated calls
    pair_hv = {}
    if L <= 22:
        ss = shortlist_idx
        for i in range(len(ss)):
            a = int(ss[i])
            for j in range(i + 1, len(ss)):
                b = int(ss[j])
                if hv_calls >= max_hv_calls:
                    pair_hv[(a, b)] = hv_single.get(a, 0.0) + hv_single.get(b, 0.0)
                    pair_hv[(b, a)] = pair_hv[(a, b)]
                else:
                    v = hv_of_set_by_idx([a, b])
                    pair_hv[(a, b)] = v
                    pair_hv[(b, a)] = v

    # CELF lazy greedy init: use heap of (-gain, idx, timestamp)
    heap = []
    for idx in shortlist_idx:
        ii = int(idx)
        g = float(hv_single.get(ii, 0.0))
        heapq.heappush(heap, (-g, ii, 0))

    selected_idx = []
    selected_mask = np.zeros(N, dtype=bool)
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))

    # main CELF loop with box-upper pruning and hv budget awareness
    while len(selected_idx) < target and heap:
        if hv_calls >= max_hv_calls:
            break
        neg_gain, idx, ts = heapq.heappop(heap)
        idx = int(idx)
        if selected_mask[idx]:
            continue
        # cheap upper bound: singleton hv cache or box proxy
        ub = max(hv_single.get(idx, 0.0), float(box_vol_all[idx]))
        if ub <= hv_selected + tol and ts != len(selected_idx):
            # unlikely to improve
            heapq.heappush(heap, (0.0, idx, len(selected_idx)))
            continue
        if ts != len(selected_idx):
            # stale entry: recompute marginal if budget permits
            if hv_calls >= max_hv_calls:
                est_gain = max(0.0, hv_single.get(idx, 0.0) - hv_selected)
                heapq.heappush(heap, (-est_gain, idx, len(selected_idx)))
                continue
            # compute true hv of selected + candidate using pair cache when possible
            if len(selected_idx) == 0:
                true_hv = hv_single.get(idx, hv_of_set_by_idx([idx]))
            elif len(selected_idx) == 1:
                only = int(selected_idx[0])
                if (only, idx) in pair_hv:
                    true_hv = pair_hv[(only, idx)]
                else:
                    true_hv = hv_of_set_by_idx([only, idx])
            else:
                true_hv = hv_of_set_by_idx(selected_idx + [idx])
            true_gain = float(max(0.0, true_hv - hv_selected))
            if true_gain < tol:
                true_gain = 0.0
            heapq.heappush(heap, (-true_gain, idx, len(selected_idx)))
            continue
        # accept candidate
        selected_idx.append(idx)
        selected_mask[idx] = True
        if hv_calls < max_hv_calls:
            hv_selected = hv_of_set_by_idx(selected_idx)
        else:
            hv_selected = sum(hv_single.get(i, 0.0) for i in selected_idx)

    # pad deterministically by box proxy if needed
    if len(selected_idx) < k:
        order_all = np.argsort(-box_vol_all, kind='stable')
        for ii in order_all:
            i = int(ii)
            if selected_mask[i]:
                continue
            selected_idx.append(i)
            selected_mask[i] = True
            if len(selected_idx) >= k:
                break
        if hv_calls < max_hv_calls:
            hv_selected = hv_of_set_by_idx(selected_idx)

    # prepare remaining candidates ordered by box proxy (fast)
    remaining_candidates = np.where(~selected_mask)[0].tolist()
    remaining_candidates.sort(key=lambda x: -box_vol_all[int(x)])

    # tiny bounded deterministic 1-for-1 swap pass to recover local improvements
    if hv_calls < max_hv_calls and len(selected_idx) > 0:
        max_swap_iters = min(10, max(2, 2 * k))
        iter_count = 0
        improved = True
        top_remaining = remaining_candidates[:min(len(remaining_candidates), 6 * k + 10)]
        while improved and iter_count < max_swap_iters and hv_calls < max_hv_calls:
            iter_count += 1
            improved = False
            # deterministic traverse selected positions
            for si in range(len(selected_idx)):
                if hv_calls >= max_hv_calls:
                    break
                old = selected_idx[si]
                # check a few top candidates
                for cand in top_remaining[:min(len(top_remaining), max(6, 2 * k))]:
                    cand = int(cand)
                    if selected_mask[cand]:
                        continue
                    ub = hv_single.get(cand, None)
                    if ub is None:
                        if hv_calls >= max_hv_calls:
                            ub = float(box_vol_all[cand])
                        else:
                            ub = hv_of_set_by_idx([cand])
                            hv_single[cand] = ub
                    if ub <= hv_selected + tol:
                        continue
                    if hv_calls >= max_hv_calls:
                        continue
                    temp = selected_idx.copy()
                    temp[si] = cand
                    temp_hv = hv_of_set_by_idx(temp)
                    if temp_hv > hv_selected + 1e-12:
                        # commit swap
                        selected_mask[old] = False
                        selected_mask[cand] = True
                        selected_idx[si] = cand
                        hv_selected = temp_hv
                        # update top_remaining deterministically
                        if old not in top_remaining:
                            top_remaining.append(old)
                        if cand in top_remaining:
                            top_remaining.remove(cand)
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
        best = int(np.argmax(box_vol_all))
        while len(final_idx) < k:
            final_idx.append(best)

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


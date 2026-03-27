import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return np.empty((0, D), dtype=float)

    # deterministic RNG
    seed = 98765
    rng = np.random.default_rng(seed)

    # reference point
    if reference_point is None:
        ref = np.max(pts, axis=0) * 1.1 if N > 0 else np.ones(D, dtype=float)
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

    # Build initial pool: prefer Pareto but fallback to all if tiny pareto set
    if pareto_size >= min(3, max(1, k // 2)):
        pool_idx = nd_idx.copy()
    else:
        pool_idx = np.arange(N)
    pool = pts[pool_idx]
    P = pool.shape[0]

    # Precompute diffs and box volumes (vectorized)
    diffs_pool = np.maximum(ref - pool, eps)  # (P, D)
    diffs_all = np.maximum(ref - pts, eps)    # (N, D)

    # Score blend parameters (different than provided implementation)
    beta = 0.8         # box exponent
    alpha_scalar = 0.5 # blend weight toward scalarized score

    box_vol_pool = np.prod(diffs_pool, axis=1) ** beta  # (P,)
    box_vol_all = np.prod(diffs_all, axis=1) ** beta   # (N,)

    # scalar probes (few deterministic weight vectors)
    M_weights = min(max(3, 2 * D), 10)
    raw = rng.random((M_weights, D)) + 1e-12
    weights = raw / raw.sum(axis=1, keepdims=True)  # (M_weights, D)
    scalarized = np.maximum(ref - pool, 0.0).dot(weights.T)  # (P, M_weights)
    scalar_score = np.mean(scalarized, axis=1)  # (P,)
    max_ss = float(np.max(scalar_score)) if scalar_score.size > 0 else 1.0
    max_ss = max(max_ss, 1e-12)
    scalar_norm = scalar_score / max_ss

    # Combine via multiplicative blending to differ from additive blend
    combined_score = (box_vol_pool ** (1.0 - alpha_scalar)) * ((scalar_norm + 1e-12) ** alpha_scalar)

    # deterministic ordering: by combined_score desc, tie-break by original pool_idx asc
    order_pool = np.lexsort((pool_idx, -combined_score))

    # adaptive O(k) shortlist (different sizing heuristic)
    shortlist_size = int(min(P, max(3, int(1.5 * k) + 15, min(120, max(30, 4 * pareto_size)))))
    shortlist_size = max(3, min(shortlist_size, P))
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()

    # ensure we can fill k: fallback expand by global box proxy if needed
    if len(shortlist_idx) < k:
        order_all = np.argsort(-box_vol_all, kind='stable')
        take = min(max(40, k), N)
        shortlist_idx = order_all[:take].tolist()

    # make unique & deterministic preserving first occurrences
    seen_tmp = set()
    uniq_shortlist_idx = []
    for ii in shortlist_idx:
        i = int(ii)
        if i not in seen_tmp:
            uniq_shortlist_idx.append(i)
            seen_tmp.add(i)
    shortlist_idx = uniq_shortlist_idx
    L = len(shortlist_idx)

    # HV-call budget (different cap)
    hv_calls = 0
    max_hv_calls = max(200, 30 * k)

    def hv_of_set(arr: np.ndarray) -> float:
        nonlocal hv_calls
        if arr is None or len(arr) == 0:
            return 0.0
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        hv_calls += 1
        return float(pg.hypervolume(a).compute(ref))

    # caches
    hv_single = {}
    box_proxy = {int(i): float(box_vol_all[int(i)]) for i in shortlist_idx}

    # precompute singleton HVs until budget, else use box proxy
    for idx in shortlist_idx:
        idx = int(idx)
        if hv_calls >= max_hv_calls:
            hv_single[idx] = float(box_proxy.get(idx, 0.0))
        else:
            hv_single[idx] = hv_of_set(pts[[idx]])

    # small pairwise cache if shortlist is small
    pair_hv = {}
    pair_threshold = 30
    if L <= pair_threshold:
        for i in range(L):
            a = int(shortlist_idx[i])
            for j in range(i + 1, L):
                b = int(shortlist_idx[j])
                if hv_calls >= max_hv_calls:
                    # conservative additive estimate
                    est = hv_single.get(a, 0.0) + hv_single.get(b, 0.0)
                    pair_hv[(a, b)] = est
                    pair_hv[(b, a)] = est
                else:
                    v = hv_of_set(pts[[a, b]])
                    pair_hv[(a, b)] = v
                    pair_hv[(b, a)] = v

    # CELF lazy greedy
    heap = []
    for idx in shortlist_idx:
        g = float(hv_single.get(int(idx), 0.0))
        # store (-gain, idx, timestamp)
        heapq.heappush(heap, (-g, int(idx), 0))

    selected_idx = []
    selected_mask = np.zeros(N, dtype=bool)
    selected_points = []
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))

    # CELF loop
    while len(selected_idx) < target and heap:
        neg_gain, idx, ts = heapq.heappop(heap)
        idx = int(idx)
        if selected_mask[idx]:
            continue
        # cheap upper bound: box proxy or singleton hv
        ub = max(hv_single.get(idx, 0.0), box_proxy.get(idx, 0.0))
        if ub <= hv_selected + tol and ts != len(selected_idx):
            # cannot improve
            heapq.heappush(heap, (0.0, idx, len(selected_idx)))
            continue

        if ts != len(selected_idx):
            # stale entry: recompute marginal gain if budget allows
            if hv_calls >= max_hv_calls:
                est_gain = max(0.0, hv_single.get(idx, 0.0) - hv_selected)
                heapq.heappush(heap, (-est_gain, idx, len(selected_idx)))
                continue
            # compute true hv of selected + candidate
            if len(selected_points) == 0:
                true_hv = hv_single.get(idx, hv_of_set(pts[[idx]]))
            elif len(selected_points) == 1:
                only = selected_idx[0]
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
        selected_mask[idx] = True
        selected_points.append(pts[idx].copy())
        if hv_calls < max_hv_calls:
            hv_selected = hv_of_set(np.vstack(selected_points))
        else:
            hv_selected = sum(hv_single.get(i, 0.0) for i in selected_idx)

    # If we didn't fill k, pad deterministically by box proxy from all points
    if len(selected_idx) < k:
        order_un = np.argsort(-box_vol_all, kind='stable')
        for idx in order_un:
            idx = int(idx)
            if selected_mask[idx]:
                continue
            selected_idx.append(idx)
            selected_mask[idx] = True
            selected_points.append(pts[idx].copy())
            if len(selected_idx) >= k:
                break
        if hv_calls < max_hv_calls and selected_points:
            hv_selected = hv_of_set(np.vstack(selected_points))

    # small deterministic swap/local improvement pass
    if hv_calls < max_hv_calls and len(selected_idx) > 0:
        remaining = np.where(~selected_mask)[0].tolist()
        remaining.sort(key=lambda x: -box_vol_all[int(x)])
        top_remaining = remaining[:min(len(remaining), max(80, 6 * k))]
        max_swap_iters = min(30, max(6, 4 * k))
        iters = 0
        improved = True
        max_cands_per_pos = min(10, max(4, k))
        while improved and iters < max_swap_iters and hv_calls < max_hv_calls:
            iters += 1
            improved = False
            # deterministic traverse selected positions
            for si in range(len(selected_idx)):
                if hv_calls >= max_hv_calls:
                    break
                old_idx = selected_idx[si]
                for cand in top_remaining[:max_cands_per_pos]:
                    cand = int(cand)
                    if selected_mask[cand]:
                        continue
                    # cheap singleton bound
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
                    temp_arr = pts[np.array(temp_idx)]
                    if hv_calls >= max_hv_calls:
                        continue
                    temp_hv = hv_of_set(temp_arr)
                    if temp_hv > hv_selected + 1e-12:
                        # commit swap
                        selected_idx[si] = cand
                        selected_mask[old_idx] = False
                        selected_mask[cand] = True
                        selected_points[si] = pts[cand].copy()
                        hv_selected = temp_hv
                        # update top_remaining (old_idx becomes candidate)
                        if old_idx not in top_remaining:
                            top_remaining.append(old_idx)
                        if cand in top_remaining:
                            top_remaining.remove(cand)
                        top_remaining.sort(key=lambda x: -box_vol_all[int(x)])
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # deterministic dedupe and final pad to exactly k
    final_idx = []
    seen = set()
    for idx in selected_idx:
        if idx not in seen:
            final_idx.append(int(idx))
            seen.add(idx)
        if len(final_idx) == k:
            break

    if len(final_idx) < k:
        # prefer remaining by box proxy
        remaining_all = np.argsort(-box_vol_all, kind='stable')
        for idx in remaining_all:
            idx = int(idx)
            if idx in seen:
                continue
            final_idx.append(idx)
            seen.add(idx)
            if len(final_idx) == k:
                break

    if len(final_idx) < k:
        # repeat best proxy point as last resort
        if N > 0:
            best = int(np.argmax(np.prod(np.maximum(ref - pts, eps), axis=1)))
        else:
            best = 0
        while len(final_idx) < k:
            final_idx.append(best)

    final_arr = pts[np.array(final_idx, dtype=int)][:k].astype(float)
    return final_arr


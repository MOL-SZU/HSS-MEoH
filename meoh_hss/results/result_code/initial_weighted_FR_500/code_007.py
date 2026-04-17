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
    seed = 2028
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
        # Efficient vectorized dominance: compare all pairs
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

    # Build pool: prefer Pareto front if reasonably sized (vectorized)
    if pareto_size >= min(3, max(1, k // 2)):
        pool_idx = nd_idx
    else:
        pool_idx = np.arange(N)
    pool = pts[pool_idx]
    P = pool.shape[0]

    # Precompute diffs and box volumes (vectorized)
    diffs_pool = np.maximum(ref - pool, eps)  # (P, D)
    diffs_all = np.maximum(ref - pts, eps)    # (N, D)

    # Score function parameters (different settings):
    # - emphasize scalarized spread more (higher alpha_scalar)
    # - softer box exponent (lower beta) to reduce dominance of large box points
    beta = 0.75         # was 0.9 -> more exploratory
    alpha_scalar = 0.65 # blend factor towards scalarized score (was 0.20 originally)
    # compute rect (box) volumes
    box_vol_pool = np.prod(diffs_pool, axis=1) ** beta  # (P,)
    box_vol_all = np.prod(diffs_all, axis=1) ** beta   # (N,)

    # few deterministic weight vectors for scalar probes: slightly more probes
    M_weights = min(max(4, 3 * D), 12)
    # deterministic quasi-random like probes (but still RNG seeded)
    raw = rng.random((M_weights, D)) + 1e-9
    weights = raw / raw.sum(axis=1, keepdims=True)  # (M_weights, D)
    # scalarized scores: higher means more room between point and reference
    scalarized = np.maximum(ref - pool, 0.0).dot(weights.T)  # (P, M_weights)
    scalar_score = np.mean(scalarized, axis=1)  # (P,)
    max_ss = float(np.max(scalar_score)) if scalar_score.size > 0 else 1.0
    max_ss = max(max_ss, 1e-12)
    scalar_norm = scalar_score / max_ss

    # Combined score: stronger emphasis on scalar_norm
    combined_score = box_vol_pool * ((1.0 - alpha_scalar) + alpha_scalar * scalar_norm)

    # deterministic ordering: by combined_score desc, tie-break by original index asc
    # lexsort with tie-breaker: first key is ascending idx, second is descending score
    order_pool = np.lexsort((pool_idx, -combined_score))

    # adaptive O(k) shortlist: more conservative scaling but capped
    shortlist_size = int(min(P, max(2 * k + 10, k + 30, min(120, max(30, 3 * pareto_size)))))
    shortlist_size = max(3, min(shortlist_size, P))
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()
    # ensure we can fill k: fallback expand by global box proxy if needed
    if len(shortlist_idx) < k:
        order_all = np.argsort(-box_vol_all, kind='stable')
        take = min(max(40, k), N)
        shortlist_idx = order_all[:take].tolist()

    # unique deterministic shortlist preserving order (use set for fast checks)
    seen_tmp = set()
    uniq_shortlist_idx = []
    for ii in shortlist_idx:
        i = int(ii)
        if i not in seen_tmp:
            uniq_shortlist_idx.append(i)
            seen_tmp.add(i)
    shortlist_idx = uniq_shortlist_idx
    L = len(shortlist_idx)

    # HV-call budget (stricter than original)
    hv_calls = 0
    max_hv_calls = max(150, 25 * k)

    def hv_of_set(arr: np.ndarray) -> float:
        nonlocal hv_calls
        if arr is None or len(arr) == 0:
            return 0.0
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        hv_calls += 1
        return float(pg.hypervolume(a).compute(ref))

    # precompute singleton HVs (cached). If budget exhausted use box proxy fallback
    hv_single = {}
    box_proxy = {int(i): float(box_vol_all[int(i)]) for i in shortlist_idx}

    for idx in shortlist_idx:
        if hv_calls >= max_hv_calls:
            hv_single[int(idx)] = float(box_proxy.get(int(idx), 0.0))
        else:
            hv_single[int(idx)] = hv_of_set(pts[[int(idx)]])

    # pairwise cache only when shortlist small
    pair_hv = {}
    pair_threshold = 40
    if L <= pair_threshold:
        for i in range(L):
            a = int(shortlist_idx[i])
            for j in range(i + 1, L):
                b = int(shortlist_idx[j])
                if hv_calls >= max_hv_calls:
                    est = hv_single.get(a, 0.0) + hv_single.get(b, 0.0)
                    pair_hv[(a, b)] = est
                    pair_hv[(b, a)] = est
                else:
                    v = hv_of_set(pts[[a, b]])
                    pair_hv[(a, b)] = v
                    pair_hv[(b, a)] = v

    # CELF lazy greedy: initialize heap with singleton gains
    heap = []
    for idx in shortlist_idx:
        gain = float(hv_single.get(int(idx), 0.0))
        heapq.heappush(heap, (-gain, int(idx), 0))

    selected_idx = []
    selected_mask = np.zeros(N, dtype=bool)  # boolean mask for membership checks
    selected_points = []  # list of point arrays
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))

    # main CELF loop with tight box upper bound pruning and HV budget awareness
    while len(selected_idx) < target and heap:
        neg_gain, idx, ts = heapq.heappop(heap)
        if selected_mask[int(idx)]:
            continue
        # cheap upper bound: singleton hv or box proxy
        ub = min(hv_single.get(int(idx), float(box_proxy.get(int(idx), 0.0))),
                 float(box_proxy.get(int(idx), 0.0)))
        if ub <= hv_selected + tol and ts != len(selected_idx):
            # this candidate cannot improve current solution
            heapq.heappush(heap, (0.0, int(idx), len(selected_idx)))
            continue

        if ts != len(selected_idx):
            # stale: recompute if budget allows, else push conservative estimate
            if hv_calls >= max_hv_calls:
                est_gain = max(0.0, hv_single.get(int(idx), 0.0) - hv_selected)
                heapq.heappush(heap, (-est_gain, int(idx), len(selected_idx)))
                continue
            # compute true hypervolume of selected + candidate efficiently
            if len(selected_points) == 0:
                true_hv = hv_single.get(int(idx), hv_of_set(pts[[int(idx)]]))
            elif len(selected_points) == 1:
                only = int(selected_idx[0])
                if (only, int(idx)) in pair_hv:
                    true_hv = pair_hv[(only, int(idx))]
                else:
                    # batch vstack once
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
        # update hv_selected if budget allows else fallback to sum singletons
        if hv_calls < max_hv_calls:
            hv_selected = hv_of_set(np.vstack(selected_points))
        else:
            hv_selected = sum(hv_single.get(i, 0.0) for i in selected_idx)

    # pad deterministically if needed using global box proxy
    if len(selected_idx) < k:
        order_un = np.argsort(-box_vol_all, kind='stable')
        for idx in order_un:
            if selected_mask[int(idx)]:
                continue
            selected_idx.append(int(idx))
            selected_mask[int(idx)] = True
            selected_points.append(pts[int(idx)].copy())
            if len(selected_idx) >= k:
                break
        if hv_calls < max_hv_calls and selected_points:
            hv_selected = hv_of_set(np.vstack(selected_points))

    # targeted deterministic swap pass on a small candidate pool if budget remains
    if hv_calls < max_hv_calls and len(selected_idx) > 0:
        remaining_candidates = np.where(~selected_mask)[0].tolist()
        # rank remaining by box proxy deterministically
        remaining_candidates.sort(key=lambda x: -box_vol_all[int(x)])
        max_swap_iters = min(40, max(6, 5 * k))
        iter_count = 0
        improved = True
        max_cands_per_pos = min(12, max(6, k))
        # restrict candidate pool to top M to keep swaps cheap
        top_remaining = remaining_candidates[:min(len(remaining_candidates), max(150, 8 * k))]
        while improved and iter_count < max_swap_iters and hv_calls < max_hv_calls:
            iter_count += 1
            improved = False
            # deterministic order of selected positions
            for si in range(len(selected_idx)):
                if hv_calls >= max_hv_calls:
                    break
                old_idx = selected_idx[si]
                # try best remaining candidates by box proxy
                for cand in top_remaining[:max_cands_per_pos]:
                    if selected_mask[int(cand)]:
                        continue
                    # cheap singleton upper bound prune
                    ub = hv_single.get(int(cand), None)
                    if ub is None:
                        if hv_calls >= max_hv_calls:
                            ub = box_proxy.get(int(cand), 0.0)
                        else:
                            ub = hv_of_set(pts[[int(cand)]])
                            hv_single[int(cand)] = ub
                    if ub <= hv_selected + tol:
                        continue
                    temp_idx = selected_idx.copy()
                    temp_idx[si] = int(cand)
                    temp_arr = pts[np.array(temp_idx)]
                    if hv_calls >= max_hv_calls:
                        continue
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
        # prefer remaining_candidates first (already sorted by box proxy)
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
        best = int(np.argmax(np.prod(np.maximum(ref - pts, eps), axis=1))) if N > 0 else 0
        while len(final_idx) < k:
            final_idx.append(best)

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


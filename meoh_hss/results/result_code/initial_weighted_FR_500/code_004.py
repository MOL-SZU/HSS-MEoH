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
        return np.zeros((k, D), dtype=float)

    # deterministic RNG
    seed = 42
    rng = np.random.default_rng(seed)

    # reference point
    if reference_point is None:
        ref = np.max(pts, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).reshape(-1)
        if ref.size != D:
            raise ValueError("reference_point must have same dimensionality as points")

    # fast nondominated filter (minimization)
    def nondominated_indices(X: np.ndarray):
        Nloc = X.shape[0]
        if Nloc == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(Nloc, dtype=bool)
        for i in range(Nloc):
            if dominated[i]:
                continue
            comp_le = np.all(X <= X[i], axis=1)
            comp_lt = np.any(X < X[i], axis=1)
            comp = comp_le & comp_lt
            comp[i] = False
            if np.any(comp):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = nondominated_indices(pts)

    # pool: use Pareto front if large enough, else full set
    if nd_idx.size >= min(4, k):  # prefer pareto if non-trivial
        pool_idx = nd_idx
    else:
        pool_idx = np.arange(N)

    pool = pts[pool_idx]

    # cheap proxies
    eps = 1e-12
    diffs_pool = np.maximum(ref - pool, eps)
    box_vol_pool = np.prod(diffs_pool, axis=1)

    # deterministic few weight scalarizations
    M = min(max(2, 2 * D), 8)
    # deterministic "random" by seeded RNG
    raw = rng.random((M, D)) + 1e-9
    weights = raw / raw.sum(axis=1, keepdims=True)
    pool_ref_minus = np.maximum(ref - pool, 0.0)
    scalarized = pool_ref_minus.dot(weights.T)
    mean_scalar = np.mean(scalarized, axis=1)
    ms_max = float(np.max(mean_scalar)) if mean_scalar.size > 0 else 1.0
    ms_max = max(ms_max, 1e-12)
    mean_scalar_norm = mean_scalar / ms_max
    score_pool = box_vol_pool * (0.15 + 0.85 * mean_scalar_norm)

    # deterministic ordering: descending score, tie-break by original index
    order_pool = np.lexsort((pool_idx, -score_pool))
    P = pool_idx.size

    # adaptive shortlist size: scale with k and Pareto size but cap
    shortlist_size = int(min(P, max(3 * k, k + 20, min(120, int(2.5 * k + 0.2 * P)))))
    shortlist_size = max(3, shortlist_size)
    shortlist_in_pool_pos = order_pool[:shortlist_size]
    shortlist_idx = pool_idx[shortlist_in_pool_pos].tolist()

    # if shortlist too small to fill k, expand deterministically using global box proxy
    if len(shortlist_idx) < k:
        diffs_all = np.maximum(ref - pts, eps)
        approx_all = np.prod(diffs_all, axis=1)
        order_all = np.argsort(-approx_all, kind='stable')
        take = min(max(50, k), N)
        shortlist_idx = order_all[:take].tolist()

    # hypervolume helper (minimization)
    def hv_of_set(arr: np.ndarray) -> float:
        if arr is None or len(arr) == 0:
            return 0.0
        hv = pg.hypervolume(np.asarray(arr, dtype=float))
        return float(hv.compute(ref))

    # compute singleton hv for shortlist (cache)
    hv_single = {}
    for idx in shortlist_idx:
        hv_single[int(idx)] = hv_of_set(pts[[int(idx)]])

    # optionally precompute pairwise hv if shortlist small
    pair_hv = {}
    if len(shortlist_idx) <= 40:
        sids = shortlist_idx
        for i in range(len(sids)):
            a = int(sids[i])
            for j in range(i + 1, len(sids)):
                b = int(sids[j])
                pair_hv[(a, b)] = hv_of_set(pts[[a, b]])
                pair_hv[(b, a)] = pair_hv[(a, b)]

    # CELF lazy greedy over shortlist using cached singletons and timestamps
    heap = []
    for idx in shortlist_idx:
        gain = float(hv_single[int(idx)])  # initial marginal when set empty
        heapq.heappush(heap, (-gain, int(idx), 0))

    selected_idx = []
    selected_set = set()
    selected_points = []
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))

    # To reduce hv calls, maintain best_known_gain for quick pruning (singleton upper bound)
    while len(selected_idx) < target and heap:
        neg_gain, idx, ts = heapq.heappop(heap)
        if idx in selected_set:
            continue
        # if timestamp stale, recompute exact marginal and reinsert
        if ts != len(selected_idx):
            # compute true hv of selected + idx
            if len(selected_points) == 0:
                true_hv = hv_single[int(idx)]
            elif len(selected_points) == 1:
                # use pair cache if available
                only = int(selected_idx[0])
                if (only, int(idx)) in pair_hv:
                    true_hv = pair_hv[(only, int(idx))]
                else:
                    arr = np.vstack([np.array(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                    true_hv = hv_of_set(arr)
            else:
                arr = np.vstack([np.array(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                true_hv = hv_of_set(arr)
            true_gain = float(true_hv - hv_selected)
            # conservative prune: if true_gain is extremely small, skip pushing many times
            heapq.heappush(heap, (-true_gain, int(idx), len(selected_idx)))
            continue
        # accept candidate
        selected_idx.append(int(idx))
        selected_set.add(int(idx))
        selected_points.append(pts[int(idx)].reshape(1, -1)[0])
        hv_selected = hv_of_set(np.vstack(selected_points))

    # if still fewer than k, deterministically pad using global box proxy
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

    # prepare remaining candidates sorted by global box proxy deterministically
    remaining_candidates = [int(i) for i in range(N) if int(i) not in selected_set]
    if remaining_candidates:
        diffs_rem = np.maximum(ref - pts[remaining_candidates], eps)
        approx_rem = np.prod(diffs_rem, axis=1)
        rem_order = np.argsort(-approx_rem, kind='stable')
        remaining_candidates = [remaining_candidates[i] for i in rem_order]

    # Bounded targeted deterministic 1-for-1 swaps with incremental checks
    max_swap_iters = min(100, max(10, 6 * k))
    iter_count = 0
    improved = True
    max_cands_per_pos = min(25, max(6, k))

    while improved and iter_count < max_swap_iters:
        iter_count += 1
        improved = False
        # deterministic shuffle of positions via RNG but seeded for determinism
        positions = list(range(len(selected_idx)))
        rng.shuffle(positions)
        for si in positions:
            if improved:
                break
            old_idx = selected_idx[si]
            # candidate pool = top remaining not in selected_set
            cands = [c for c in remaining_candidates if c not in selected_set]
            if not cands:
                break
            cands = cands[:max_cands_per_pos]
            for cand in cands:
                # fast upper bound: singleton hv of candidate <= possible marginal, skip if not promising
                possible_upper = hv_single.get(int(cand), None)
                if possible_upper is None:
                    # compute singleton Hv lazily
                    possible_upper = hv_of_set(pts[[int(cand)]])
                    hv_single[int(cand)] = possible_upper
                # trivial pruning: if candidate singleton hv smaller than tiny fraction of current hv selected, allow check but don't expect large gains
                # compute hv of swapped set
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
                    # update remaining_candidates deterministically: put old_idx at front
                    if int(old_idx) not in remaining_candidates:
                        remaining_candidates.insert(0, int(old_idx))
                    if int(cand) in remaining_candidates:
                        remaining_candidates.remove(int(cand))
                    improved = True
                    break
        if not improved:
            break

    # final dedupe preserving order and deterministic padding to exactly k
    final_idx = []
    seen = set()
    for idx in selected_idx:
        if idx not in seen:
            final_idx.append(int(idx))
            seen.add(int(idx))
        if len(final_idx) == k:
            break

    if len(final_idx) < k:
        remaining = [i for i in range(N) if i not in seen]
        if remaining:
            diffs_rem = np.maximum(ref - pts[remaining], eps)
            approx_rem = np.prod(diffs_rem, axis=1)
            order_rem = np.argsort(-approx_rem, kind='stable')
            for pos in order_rem:
                final_idx.append(int(remaining[int(pos)]))
                seen.add(int(remaining[int(pos)]))
                if len(final_idx) == k:
                    break
        if len(final_idx) < k:
            # pad by repeating best global box proxy point
            diffs_all = np.maximum(ref - pts, eps)
            approx_all = np.prod(diffs_all, axis=1)
            best = int(np.argmax(approx_all))
            while len(final_idx) < k:
                final_idx.append(best)

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


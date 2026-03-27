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
    if N == 0:
        raise ValueError("points must contain at least one point")
    if k < 1:
        raise ValueError("k must be >= 1")

    rng = np.random.default_rng(2026)

    # reference point (minimization assumed)
    if reference_point is None:
        ref = np.max(pts, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).reshape(-1)
        if ref.size != D:
            raise ValueError("reference_point must have same dimensionality as points")

    eps = 1e-12

    # vectorized nondominated filter (minimization)
    def nondominated_indices(X: np.ndarray):
        M = X.shape[0]
        if M == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(M, dtype=bool)
        # efficient pairwise check using broadcasting when moderate M
        for i in range(M):
            if dominated[i]:
                continue
            le = np.all(X <= X[i], axis=1)
            lt = np.any(X < X[i], axis=1)
            dom = le & lt
            dom[i] = False
            if np.any(dom):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = nondominated_indices(pts)
    pareto_size = len(nd_idx)

    # cheap box-volume upper bound (singleton upper bound) and scalar proxies
    diffs = np.maximum(ref - pts, eps)
    box_vol = np.prod(diffs, axis=1)

    M_w = min(max(3, 2 * D), 8)
    raw = rng.random((M_w, D)) + 1e-9
    weights = raw / raw.sum(axis=1, keepdims=True)
    ref_minus = np.maximum(ref - pts, 0.0)
    scalarized = ref_minus.dot(weights.T)  # (N, M_w)
    mean_scalar = np.mean(scalarized, axis=1)
    ms_max = max(1e-12, float(np.max(mean_scalar)))
    mean_scalar_norm = mean_scalar / ms_max

    # combined proxy (hybrid) for deterministic ranking
    score = box_vol * (0.2 + 0.8 * mean_scalar_norm)

    # ordering: Pareto first (by score), then dominated
    nondom_mask = np.zeros(N, dtype=bool)
    nondom_mask[nd_idx] = True
    nondom_indices = np.where(nondom_mask)[0]
    dom_indices = np.where(~nondom_mask)[0]

    order_nd = nondom_indices[np.argsort(-score[nondom_indices], kind="stable")]
    order_dom = dom_indices[np.argsort(-score[dom_indices], kind="stable")]
    ordered_all_idx = np.concatenate([order_nd, order_dom])

    # shortlist size adaptive: proportional to k but bounded, scale with pareto_size
    shortlist_size = int(min(N, max(4 * k, pareto_size + 6, min(80, 6 * k))))
    shortlist_size = max(5, shortlist_size)
    shortlist_idx = ordered_all_idx[:shortlist_size].tolist()

    # ensure shortlist large enough to fill k if possible
    if len(shortlist_idx) < min(k, N):
        take = min(N, max(k, 2 * shortlist_size))
        shortlist_idx = ordered_all_idx[:take].tolist()

    # hypervolume helper
    def hv_of(arr: np.ndarray) -> float:
        if arr is None:
            return 0.0
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return 0.0
        # ensure 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return float(pg.hypervolume(arr).compute(ref))

    # Precompute singleton HV for shortlist
    hv_single = {}
    for idx in shortlist_idx:
        i = int(idx)
        hv_single[i] = hv_of(pts[[i]])

    # selective pairwise caching when shortlist small
    pair_hv = {}
    if len(shortlist_idx) <= 40:
        sids = shortlist_idx
        for i in range(len(sids)):
            a = int(sids[i])
            for j in range(i + 1, len(sids)):
                b = int(sids[j])
                hvv = hv_of(pts[[a, b]])
                pair_hv[(a, b)] = hvv
                pair_hv[(b, a)] = hvv

    # CELF lazy greedy initialization
    heap = []
    for idx in shortlist_idx:
        ii = int(idx)
        gain = float(hv_single.get(ii, 0.0))
        # push (negative gain, index, timestamp)
        heapq.heappush(heap, (-gain, ii, 0))

    selected_idx = []
    selected_set = set()
    selected_points = []
    hv_selected = 0.0
    target = min(k, len(shortlist_idx))
    tol = 1e-12

    # CELF loop with upper-bound pruning using box_vol
    while len(selected_idx) < target and heap:
        neg_gain, idx, ts = heapq.heappop(heap)
        if idx in selected_set:
            continue
        # quick prune: if even the box upper bound cannot improve over tiny threshold relative to hv_selected
        ub = box_vol[int(idx)]
        if hv_selected > 0 and ub <= 1e-16 * hv_selected:
            # negligible contribution, skip permanently
            continue
        if ts != len(selected_idx):
            # stale entry: recompute true marginal
            if len(selected_points) == 0:
                true_hv = hv_single.get(int(idx), hv_of(pts[[int(idx)]]))
            elif len(selected_points) == 1:
                only = int(selected_idx[0])
                if (only, int(idx)) in pair_hv:
                    true_hv = pair_hv[(only, int(idx))]
                else:
                    arr = np.vstack([np.array(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                    true_hv = hv_of(arr)
            else:
                # tight upper bound: sum of box_vol of candidate and currently selected is not additive,
                # but use box_vol[idx] as a cheap upper bound; if not promising compute exact HV
                if hv_selected > 0 and ub <= 1e-14 * hv_selected:
                    true_gain = 0.0
                    heapq.heappush(heap, (-true_gain, int(idx), len(selected_idx)))
                    continue
                arr = np.vstack([np.array(selected_points, dtype=float), pts[int(idx)].reshape(1, -1)])
                true_hv = hv_of(arr)
            true_gain = float(true_hv - hv_selected)
            if true_gain < 0 and true_gain > -1e-14:
                true_gain = 0.0
            heapq.heappush(heap, (-true_gain, int(idx), len(selected_idx)))
            continue
        # accept candidate
        selected_idx.append(int(idx))
        selected_set.add(int(idx))
        selected_points.append(pts[int(idx)].copy())
        hv_selected = hv_of(np.vstack(selected_points))

    # pad deterministically if needed
    if len(selected_idx) < k:
        for idx in ordered_all_idx:
            ii = int(idx)
            if ii in selected_set:
                continue
            selected_idx.append(ii)
            selected_set.add(ii)
            selected_points.append(pts[ii].copy())
            if len(selected_idx) >= k or len(selected_idx) >= N:
                break
        hv_selected = hv_of(np.vstack(selected_points)) if selected_points else 0.0

    # build remaining candidates list maintaining deterministic order
    remaining_candidates = [int(i) for i in ordered_all_idx if int(i) not in selected_set]

    # bounded deterministic 1-for-1 swap local search
    max_swap_iters = min(160, max(16, 10 * k))
    iter_count = 0
    improved = True
    cand_cap = min(32, max(8, 3 * k))

    # deterministic permutations based on RNG (seeded)
    while improved and iter_count < max_swap_iters:
        iter_count += 1
        improved = False
        sel_positions = list(range(len(selected_idx)))
        sel_positions = list(rng.permutation(sel_positions))
        rem_order = list(remaining_candidates)
        if rem_order:
            rem_order = [rem_order[i] for i in rng.permutation(len(rem_order))]
        for si in sel_positions:
            if improved:
                break
            old_idx = selected_idx[si]
            # consider top candidates not selected up to cap
            to_check = []
            for c in rem_order:
                if c not in selected_set:
                    to_check.append(c)
                    if len(to_check) >= cand_cap:
                        break
            if not to_check:
                continue
            for cand in to_check:
                # cheap prune using singleton upper bound
                ub_single = hv_single.get(int(cand), None)
                if ub_single is None:
                    ub_single = hv_of(pts[[int(cand)]])
                    hv_single[int(cand)] = ub_single
                if hv_selected > 0 and ub_single <= 1e-14 * hv_selected:
                    continue
                # Evaluate swapped set HV
                temp_idx = selected_idx.copy()
                temp_idx[si] = int(cand)
                temp_arr = pts[np.array(temp_idx)]
                temp_hv = hv_of(temp_arr)
                if temp_hv > hv_selected + tol:
                    # commit swap
                    selected_idx[si] = int(cand)
                    selected_set.remove(int(old_idx))
                    selected_set.add(int(cand))
                    selected_points[si] = pts[int(cand)].copy()
                    hv_selected = temp_hv
                    # update remaining_candidates deterministically: move old to front
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

    # final dedupe preserving selection order and deterministic padding to exactly k
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
        for idx in ordered_all_idx:
            ii = int(idx)
            if ii in seen:
                continue
            final_idx.append(ii)
            seen.add(ii)
            if len(final_idx) == k:
                break

    # last resort: repeat best point deterministically
    if len(final_idx) < k:
        if len(ordered_all_idx) == 0:
            fallback = np.minimum(np.mean(pts, axis=0), ref)
            return np.tile(fallback, (k, 1)).astype(float)
        best = int(ordered_all_idx[0])
        while len(final_idx) < k:
            final_idx.append(best)

    final_arr = pts[np.array(final_idx, dtype=int)][:k]
    return np.asarray(final_arr, dtype=float)


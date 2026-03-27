import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    # Optional KDTree for faster distance queries in diversity stage
    try:
        from scipy.spatial import cKDTree as KDTree
    except Exception:
        KDTree = None

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D).")
    N, D = points.shape

    if not (isinstance(k, int) and k > 0 and k <= N):
        raise ValueError("k must be an integer in 1..N (number of points).")

    # reference_point default
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)
    if reference_point.shape[0] != D:
        raise ValueError("reference_point must have dimension D = points.shape[1].")

    # -----------------------------
    # 1) Pareto (nondominated) filtering (assume minimization)
    # -----------------------------
    def nondominated_indices(vals):
        n = vals.shape[0]
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # j dominates i if j <= i in all and < in at least one
            mask = np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1)
            if np.any(mask):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = nondominated_indices(points)
    if nd_idx.size == 0:
        candidate_idx = np.arange(N)
    else:
        candidate_idx = nd_idx

    cand_points = points[candidate_idx]

    # -----------------------------
    # 2) Modified Surrogate ranking
    #    - different exponent beta (more smoothing)
    #    - use sqrt-range weighting and sigmoid normalization per-dim
    #    - scale final scores to [0,1] for stability
    # -----------------------------
    eps = 1e-12
    diffs = np.maximum(reference_point - cand_points, eps)  # positive distances

    # compute ranges and apply sqrt dampening to reduce extreme dim dominance
    ranges = np.maximum(np.ptp(points, axis=0), eps)
    gamma = 0.5  # different parameterization: take ranges^(gamma)
    raw_weights = 1.0 / (ranges ** gamma)
    # sigmoid-normalize weights for stability and to avoid one-dim dominance
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    w_sig = _sigmoid(raw_weights - np.median(raw_weights))
    dim_weights = w_sig / (np.sum(w_sig) + eps)

    # new exponent
    beta = 0.6  # different from original 0.85: smoother emphasis
    with np.errstate(divide='ignore', invalid='ignore'):
        # use log of diffs, raise abs(log) to beta but preserve sign (logs are <= large)
        logd = np.log(diffs)
        # to keep monotonicity, use (-logd) since larger diffs -> larger contribution
        # compute per-dim contributions
        contrib = ( (-logd) ** beta ) * dim_weights.reshape(1, -1)
        surrogate_scores = np.sum(contrib, axis=1)

    # stabilise and scale to [0,1]
    surrogate_scores = np.nan_to_num(surrogate_scores, neginf=-1e300, posinf=1e300)
    if surrogate_scores.size > 0:
        mn, mx = np.min(surrogate_scores), np.max(surrogate_scores)
        if mx - mn > 0:
            surrogate_scores = (surrogate_scores - mn) / (mx - mn)
        else:
            surrogate_scores = np.zeros_like(surrogate_scores)

    # sort descending by surrogate (higher better); stable for determinism
    order = np.argsort(-surrogate_scores, kind="stable")
    ordered_idx = candidate_idx[order]

    # -----------------------------
    # 3) Adaptive shortlist + diversity-aware farthest-first (KDTree accelerated)
    # -----------------------------
    M = len(ordered_idx)
    # adaptive pool and shortlist caps (more aggressive shrink)
    pool_top = min(M, max(2 * k, int(k + 20)))
    pool_idx = ordered_idx[:pool_top]
    L = min(len(pool_idx), max(4 * k, int(k * np.log1p(N) + 10)))
    if L <= 0:
        shortlisted_idx = np.array([], dtype=int)
    elif pool_idx.size <= L:
        shortlisted_idx = pool_idx.copy()
    else:
        # farthest-first: seed with best surrogate (first element)
        sel = [int(pool_idx[0])]
        remaining = [int(x) for x in pool_idx[1:]]
        coords = points  # alias
        if KDTree is not None and len(remaining) > 40:
            rem_coords = coords[remaining]
            tree = KDTree(rem_coords)
            seed_pt = coords[sel[0]]
            # initial min distances
            dists, _ = tree.query(seed_pt.reshape(1, -1), k=len(remaining))
            min_dists = dists.ravel()
            for _ in range(1, L):
                if len(remaining) == 0:
                    break
                max_pos = int(np.argmax(min_dists))
                chosen = remaining.pop(max_pos)
                min_dists = np.delete(min_dists, max_pos)
                sel.append(chosen)
                if len(remaining) == 0:
                    break
                new_pt = coords[chosen]
                d_new, _ = tree.query(new_pt.reshape(1, -1), k=len(remaining))
                d_new = d_new.ravel()
                min_dists = np.minimum(min_dists, d_new)
        else:
            rem_arr = coords[remaining]
            seed_pt = coords[sel[0]]
            min_dists = np.linalg.norm(rem_arr - seed_pt.reshape(1, -1), axis=1)
            for _ in range(1, L):
                if not remaining:
                    break
                max_pos = int(np.argmax(min_dists))
                chosen = remaining.pop(max_pos)
                min_dists = np.delete(min_dists, max_pos)
                sel.append(chosen)
                if not remaining:
                    break
                new_pt = coords[chosen]
                rem_arr = coords[remaining]
                d_new = np.linalg.norm(rem_arr - new_pt.reshape(1, -1), axis=1)
                min_dists = np.minimum(min_dists, d_new)
        shortlisted_idx = np.array(sel, dtype=int)

    if shortlisted_idx.size < min(k, 1):
        shortlisted_idx = ordered_idx[:min(len(ordered_idx), max(1, L))]

    candidates = [int(i) for i in shortlisted_idx.tolist()]

    # -----------------------------
    # 4) Precompute singleton HVs, use canonical tuple keys, CELF lazy greedy with early stop
    # -----------------------------
    hv_cache = {}  # key: tuple(sorted indices) -> hv value

    def hv_of_set(idx_set):
        # canonical key: sorted tuple of ints
        key = tuple(sorted(int(i) for i in idx_set))
        if not key:
            return 0.0
        if key in hv_cache:
            return hv_cache[key]
        arr = points[list(key)]
        try:
            hv_obj = pg.hypervolume(arr)
            val = float(hv_obj.compute(reference_point))
        except Exception:
            val = 0.0
        hv_cache[key] = val
        return val

    # precompute singleton HVs for shortlist candidates
    singleton_hv = {}
    for idx in candidates:
        singleton_hv[idx] = hv_of_set([idx])

    # initialize CELF heap with marginal gains w.r.t empty set (singleton HVs)
    heap = []
    # heap entries: (-gain, tie_idx, idx, last_updated_round)
    for idx in candidates:
        gain = float(singleton_hv.get(idx, 0.0))
        heap.append((-gain, int(idx), int(idx), 0))
    heapq.heapify(heap)

    selected = []
    S_set = set()
    current_round = 1
    current_hv = 0.0
    hv_call_count = 0
    tol = 1e-12  # early stop tolerance
    max_hv_calls = max(2000, 200 * k)  # safety cap to avoid runaway hv calls

    while len(selected) < k and heap:
        neg_gain, tie, idx, last_round = heapq.heappop(heap)
        stored_gain = -neg_gain
        # If last updated wrt previous round, accept
        if last_round == current_round - 1:
            selected.append(int(idx))
            S_set.add(int(idx))
            current_hv = hv_of_set(S_set)
            hv_call_count += 1
            current_round += 1
            # early stop if we've reached k or no more positive marginal
            continue
        else:
            # recompute true marginal gain w.r.t current S_set
            hv_with = hv_of_set(S_set | {int(idx)})
            hv_call_count += 1
            true_gain = hv_with - current_hv
            # early pruning: if nonpositive and we already have at least one selected, drop it
            if true_gain <= tol and len(selected) > 0:
                # skip considering this idx further
                if hv_call_count > max_hv_calls:
                    break
                continue
            # push back with updated marginal and mark as updated for current_round
            heapq.heappush(heap, (-float(true_gain), int(idx), int(idx), current_round - 1))
            if hv_call_count > max_hv_calls:
                break
            continue

    # Supplement if fewer than k selected: use global surrogate over all points deterministically
    if len(selected) < k:
        need = k - len(selected)
        already = set(selected)
        # compute global surrogate using same formula as above but over all points
        all_diffs = np.maximum(reference_point - points, eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            logd_all = np.log(all_diffs)
            contrib_all = ((-logd_all) ** beta) * dim_weights.reshape(1, -1)
            all_surrogate = np.sum(contrib_all, axis=1)
        all_surrogate = np.nan_to_num(all_surrogate, neginf=-1e300, posinf=1e300)
        # scale to [0,1]
        if all_surrogate.size > 0:
            mn, mx = np.min(all_surrogate), np.max(all_surrogate)
            if mx - mn > 0:
                all_surrogate = (all_surrogate - mn) / (mx - mn)
            else:
                all_surrogate = np.zeros_like(all_surrogate)
        remaining_indices = [i for i in range(N) if i not in already]
        extras = []
        if remaining_indices:
            rem_scores = all_surrogate[remaining_indices]
            pick_order = np.argsort(-rem_scores, kind="stable")[:need]
            extras = [remaining_indices[i] for i in pick_order]
        else:
            if selected:
                extras = [selected[-1]] * need
            else:
                extras = [None] * need
        for ex in extras:
            if ex is None:
                selected.append(None)
            else:
                selected.append(int(ex))

    # Build final selected array, handling None padding
    final = []
    for idx in selected[:k]:
        if idx is None:
            final.append(reference_point.copy())
        else:
            final.append(points[int(idx)])
    final = np.asarray(final, dtype=float)

    # ensure shape (k, D)
    if final.shape[0] < k:
        if final.size == 0:
            pad_row = reference_point.reshape(1, -1)
        else:
            pad_row = final[-1].reshape(1, -1)
        pad = np.vstack([pad_row] * (k - final.shape[0]))
        final = np.vstack([final, pad])
    elif final.shape[0] > k:
        final = final[:k]

    return final.reshape(k, D)


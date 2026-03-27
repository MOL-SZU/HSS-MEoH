import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    # hypervolume dependency
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    # optional KDTree for diversity acceleration
    try:
        from scipy.spatial import cKDTree as KDTree
    except Exception:
        KDTree = None

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 0))
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D).")
    N, D = pts.shape

    if not isinstance(k, int):
        raise ValueError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be > 0")

    pad_if_needed = False
    if k > N:
        pad_if_needed = True
        k_eff = N
    else:
        k_eff = k

    # reference point
    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape((D,))
    if reference_point.shape[0] != D:
        raise ValueError("reference_point must have dimension D = points.shape[1].")

    # Fast nondominated filter (assume minimization)
    def nondominated_indices(arr: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # j dominates i if j <= i in all dims and < in at least one
            mask = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
            if np.any(mask):
                dominated[i] = True
        return np.where(~dominated)[0]

    # small N edge cases
    if N == 0:
        return np.empty((0, D))
    if N == 1:
        row = pts[0].reshape(1, D)
        if pad_if_needed:
            out = np.vstack([row] * k)
            return out.reshape(k, D)
        else:
            return row.reshape(k_eff, D)

    nd_idx = nondominated_indices(pts)
    if nd_idx.size == 0:
        cand_idx = np.arange(N, dtype=int)
    else:
        cand_idx = nd_idx
    cand_pts = pts[cand_idx]
    m = cand_pts.shape[0]

    # Surrogate: weighted log-product + max gap (vectorized), normalized
    eps = 1e-12
    diffs = np.maximum(reference_point.reshape(1, -1) - cand_pts, eps)  # shape (m, D)

    data_min = np.min(pts, axis=0)
    data_max = np.max(pts, axis=0)
    ranges = np.maximum(data_max - data_min, eps)
    dim_weights = 1.0 / ranges
    dim_weights = dim_weights / np.sum(dim_weights)

    with np.errstate(divide='ignore', invalid='ignore'):
        logprod = np.sum(dim_weights.reshape(1, -1) * np.log(diffs + 1e-12), axis=1)
    maxgap = np.max(diffs, axis=1)

    # normalize terms robustly
    lp_min, lp_max = np.min(logprod), np.max(logprod)
    if lp_max - lp_min > 0:
        logprod_norm = (logprod - lp_min) / (lp_max - lp_min + eps)
    else:
        logprod_norm = np.zeros_like(logprod)
    if np.max(maxgap) > 0:
        maxgap_norm = maxgap / (np.max(maxgap) + eps)
    else:
        maxgap_norm = np.zeros_like(maxgap)

    gamma = 0.6
    surrogate = gamma * logprod_norm + (1.0 - gamma) * maxgap_norm

    # order by surrogate descending, stable
    order_sur = np.argsort(-surrogate, kind='stable')
    ordered_local = order_sur  # indices into cand_idx

    # Adaptive shortlist size: small O(k) with modest floor to cut HV calls
    candidate_count = min(m, max(4 * k_eff, 16))
    candidate_count = max(candidate_count, k_eff)

    # form pool_top (a little larger than candidate_count) then diversify
    pool_top = min(m, max(candidate_count * 2, int(10 + k_eff)))
    pool_local = ordered_local[:pool_top]
    pool_global = cand_idx[pool_local]  # global indices

    if pool_global.size <= candidate_count:
        candidates_global = list(map(int, pool_global.tolist()))
    else:
        # farthest-first diversity seeded by best surrogate
        sel_globals = [int(pool_global[0])]
        remaining_globals = [int(x) for x in pool_global[1:]]
        if KDTree is not None and len(remaining_globals) > 50:
            rem_coords = pts[remaining_globals]
            tree = KDTree(rem_coords)
            seed_pt = pts[sel_globals[0]]
            dists, _ = tree.query(seed_pt.reshape(1, -1), k=len(remaining_globals))
            min_dists = dists.ravel()
            for _ in range(1, candidate_count):
                if len(remaining_globals) == 0:
                    break
                max_pos = int(np.argmax(min_dists))
                chosen = remaining_globals.pop(max_pos)
                min_dists = np.delete(min_dists, max_pos)
                sel_globals.append(chosen)
                if len(remaining_globals) == 0:
                    break
                new_pt = pts[chosen]
                d_new, _ = tree.query(new_pt.reshape(1, -1), k=len(remaining_globals))
                d_new = d_new.ravel()
                min_dists = np.minimum(min_dists, d_new)
        else:
            rem_arr = pts[remaining_globals]
            seed_pt = pts[sel_globals[0]]
            min_dists = np.linalg.norm(rem_arr - seed_pt.reshape(1, -1), axis=1)
            for _ in range(1, candidate_count):
                if not remaining_globals:
                    break
                max_pos = int(np.argmax(min_dists))
                chosen = remaining_globals.pop(max_pos)
                min_dists = np.delete(min_dists, max_pos)
                sel_globals.append(chosen)
                if not remaining_globals:
                    break
                rem_arr = pts[remaining_globals]
                new_pt = pts[chosen]
                d_new = np.linalg.norm(rem_arr - new_pt.reshape(1, -1), axis=1)
                min_dists = np.minimum(min_dists, d_new)
        # ensure order uniqueness and deterministic
        seen = set()
        candidates_global = []
        for g in sel_globals:
            if g not in seen:
                candidates_global.append(int(g))
                seen.add(g)
        candidates_global = candidates_global[:candidate_count]

    if len(candidates_global) == 0:
        # fallback to top surrogate among all candidates
        top_locals = ordered_local[:max(1, candidate_count)]
        candidates_global = [int(cand_idx[i]) for i in top_locals]

    # HV cache with canonical sorted tuple keys
    hv_cache = {}

    def hv_of_indices(index_iterable):
        key = tuple(sorted(int(i) for i in index_iterable))
        if key in hv_cache:
            return hv_cache[key]
        if len(key) == 0:
            val = 0.0
        else:
            arr = pts[list(key)]
            val = float(pg.hypervolume(arr).compute(reference_point))
        hv_cache[key] = val
        return val

    # precompute singleton HVs for shortlist candidates
    singleton_hv = {}
    for c in candidates_global:
        singleton_hv[c] = hv_of_indices((c,))

    # initialize CELF heap: (-estimate_gain, tie, idx, last_eval_round)
    heap = []
    for tie_order, c in enumerate(candidates_global):
        g = float(singleton_hv.get(c, 0.0))
        heap.append((-g, int(tie_order), int(c), 0))
    heapq.heapify(heap)

    selected_list = []
    selected_set = set()
    hv_S = 0.0
    tol = 1e-12

    # CELF main loop (lazy greedy) with early stop
    while len(selected_list) < k_eff and heap:
        neg_est, tie, idx_c, last_eval = heapq.heappop(heap)
        est = -neg_est
        if idx_c in selected_set:
            continue
        if last_eval == len(selected_list):
            # up-to-date, accept
            selected_list.append(int(idx_c))
            selected_set.add(int(idx_c))
            hv_S = hv_of_indices(selected_set)
            continue
        # recompute true marginal
        hv_with = hv_of_indices(selected_set | {int(idx_c)})
        true_gain = hv_with - hv_S
        # push back updated
        heapq.heappush(heap, (-float(true_gain), int(tie), int(idx_c), len(selected_list)))
        # early stopping: if current recomputed top and this gain nonpositive
        top_neg = heap[0][0] if heap else 0.0
        if true_gain <= tol and -top_neg <= tol:
            break

    # Supplement deterministically by surrogate if not enough selected
    if len(selected_list) < k_eff:
        needed = k_eff - len(selected_list)
        already = set(selected_list)
        # compute surrogate on all points to pick remaining deterministically
        all_diffs = np.maximum(reference_point.reshape(1, -1) - pts, eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            all_logprod = np.sum(dim_weights.reshape(1, -1) * np.log(all_diffs + 1e-12), axis=1)
        # combine with maxgap
        all_maxgap = np.max(all_diffs, axis=1)
        lp_min, lp_max = np.min(all_logprod), np.max(all_logprod)
        if lp_max - lp_min > 0:
            all_lp_norm = (all_logprod - lp_min) / (lp_max - lp_min + eps)
        else:
            all_lp_norm = np.zeros_like(all_logprod)
        if np.max(all_maxgap) > 0:
            all_mg_norm = all_maxgap / (np.max(all_maxgap) + eps)
        else:
            all_mg_norm = np.zeros_like(all_maxgap)
        all_scores = gamma * all_lp_norm + (1.0 - gamma) * all_mg_norm
        remaining_indices = [i for i in range(N) if i not in already]
        if remaining_indices:
            rem_scores = all_scores[remaining_indices]
            order_rem = np.argsort(-rem_scores, kind='stable')
            for idx in order_rem:
                if len(selected_list) >= k_eff:
                    break
                selected_list.append(int(remaining_indices[int(idx)]))
        # fill by repeating last or using lowest index deterministically
        while len(selected_list) < k_eff:
            if selected_list:
                selected_list.append(int(selected_list[-1]))
            else:
                selected_list.append(0)

    # ensure uniqueness while preserving order; if duplicates dropped, fill with leftovers
    final_sel = selected_list[:k_eff]
    unique_final = []
    seen2 = set()
    for idx in final_sel:
        if idx not in seen2:
            unique_final.append(int(idx))
            seen2.add(idx)
    if len(unique_final) < k_eff:
        leftover = [i for i in range(N) if i not in seen2]
        for idx in leftover:
            if len(unique_final) >= k_eff:
                break
            unique_final.append(int(idx))
            seen2.add(idx)
        while len(unique_final) < k_eff:
            if unique_final:
                unique_final.append(int(unique_final[-1]))
            else:
                unique_final.append(0)
    final_sel = np.array(unique_final[:k_eff], dtype=int).tolist()

    # pad to k if needed by repeating last selected
    if pad_if_needed:
        while len(final_sel) < k:
            final_sel.append(final_sel[-1])

    final_sel = np.array(final_sel, dtype=int)[:k]
    subset = pts[final_sel]
    return subset.reshape((k, D))


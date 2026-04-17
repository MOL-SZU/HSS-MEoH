import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    # hypervolume dependency
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    # optional kd-tree for diversity acceleration
    try:
        from scipy.spatial import cKDTree as KDTree
    except Exception:
        KDTree = None

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        # return empty shaped (k, D) not well-defined; mimic prior behavior: return empty (0,0)
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

    # -------------------------
    # 1) Pareto (nondominated) filtering (assume minimization)
    # -------------------------
    def nondominated_indices(arr: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # j dominates i if j <= i in all and < in at least one
            mask = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
            if np.any(mask):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = nondominated_indices(pts)
    if nd_idx.size == 0:
        # fallback: use all points
        cand_idx = np.arange(N, dtype=int)
    else:
        cand_idx = nd_idx

    cand_pts = pts[cand_idx]
    m = cand_pts.shape[0]

    # trivial cases
    if N == 0:
        return np.empty((0, D))
    if N == 1:
        # pad deterministically if needed
        if pad_if_needed and k > 1:
            return np.vstack([pts[0]] * k).reshape(k, D)
        else:
            return pts[0].reshape(k, D)

    # -------------------------
    # 2) Surrogate ranking: weighted log-product + max-gap blending
    # -------------------------
    eps = 1e-12
    diffs = np.maximum(reference_point.reshape(1, -1) - cand_pts, eps)  # shape (m, D)

    # dimension weights: inverse of range to emphasize tight dims
    data_min = np.min(pts, axis=0)
    data_max = np.max(pts, axis=0)
    ranges = np.maximum(data_max - data_min, eps)
    dim_weights = 1.0 / ranges
    dim_weights = dim_weights / np.sum(dim_weights)

    # compute log-product proxy (stable)
    with np.errstate(divide='ignore', invalid='ignore'):
        logprod = np.sum(dim_weights.reshape(1, -1) * np.log(diffs + 1e-12), axis=1)
    maxgap = np.max(diffs, axis=1)
    # normalize terms
    logprod_norm = (logprod - np.min(logprod)) if np.ptp(logprod) > 0 else logprod
    if np.ptp(logprod_norm) > 0:
        logprod_norm = (logprod_norm - np.min(logprod_norm)) / (np.ptp(logprod_norm) + eps)
    maxgap_norm = maxgap / (np.max(maxgap) + eps)

    gamma = 0.55
    surrogate = gamma * logprod_norm + (1.0 - gamma) * maxgap_norm

    # stable ordering by surrogate (descending)
    order_sur = np.argsort(-surrogate, kind='stable')
    ordered_local_idx = order_sur  # indices into cand_idx

    # -------------------------
    # 3) Adaptive shortlist + diversity (farthest-first) with KDTree fallback
    # -------------------------
    candidate_count = min(m, max(3 * k_eff, 12))
    candidate_count = max(candidate_count, k_eff)

    # take a pool of top candidates by surrogate then diversify
    pool_top = min(m, max(candidate_count * 2, int(10 + k_eff)))
    pool_local = ordered_local_idx[:pool_top]
    pool_global = cand_idx[pool_local]  # global indices
    pool_pts = pts[pool_global]

    if pool_pts.shape[0] <= candidate_count:
        shortlisted_global = pool_global.tolist()
    else:
        # farthest-first selection
        sel_globals = [int(pool_global[0])]  # seed with best surrogate
        # remaining as list of globals
        remaining_globals = [int(x) for x in pool_global[1:]]
        if KDTree is not None and len(remaining_globals) > 50:
            rem_coords = pts[remaining_globals]
            tree = KDTree(rem_coords)
            # initial min distances to seed
            seed_pt = pts[sel_globals[0]]
            dists, _ = tree.query(seed_pt.reshape(1, -1), k=len(remaining_globals))
            min_dists = dists.ravel()
            # iterative selection
            for _ in range(1, candidate_count):
                if len(remaining_globals) == 0:
                    break
                max_pos = int(np.argmax(min_dists))
                chosen_global = remaining_globals.pop(max_pos)
                min_dists = np.delete(min_dists, max_pos)
                sel_globals.append(chosen_global)
                if len(remaining_globals) == 0:
                    break
                # update min_dists by distances to new point
                new_pt = pts[chosen_global]
                d_new, _ = tree.query(new_pt.reshape(1, -1), k=len(remaining_globals))
                d_new = d_new.ravel()
                min_dists = np.minimum(min_dists, d_new)
        else:
            # pure numpy incremental distances (squared)
            rem_arr = pts[remaining_globals]
            seed_pt = pts[sel_globals[0]]
            min_dists = np.linalg.norm(rem_arr - seed_pt.reshape(1, -1), axis=1)
            for _ in range(1, candidate_count):
                if not remaining_globals:
                    break
                max_pos = int(np.argmax(min_dists))
                chosen_global = remaining_globals.pop(max_pos)
                min_dists = np.delete(min_dists, max_pos)
                sel_globals.append(chosen_global)
                if not remaining_globals:
                    break
                rem_arr = pts[remaining_globals]
                new_pt = pts[chosen_global]
                d_new = np.linalg.norm(rem_arr - new_pt.reshape(1, -1), axis=1)
                min_dists = np.minimum(min_dists, d_new)
        shortlisted_global = sel_globals

    # ensure deterministic uniqueness and preserve order encountered
    seen = set()
    candidates_global = []
    for idx in shortlisted_global:
        if idx not in seen:
            candidates_global.append(int(idx))
            seen.add(idx)

    # cap to candidate_count
    candidates_global = candidates_global[:candidate_count]

    # If shortlist empty (corner), fall back to best surrogates
    if len(candidates_global) == 0:
        top_local = ordered_local_idx[:max(1, candidate_count)]
        candidates_global = [int(cand_idx[i]) for i in top_local]

    # -------------------------
    # 4) CELF lazy greedy with canonical HV cache (sorted tuple keys)
    # -------------------------
    hv_cache = {}

    def hv_of_indices(index_iterable):
        # canonical key: sorted tuple of ints
        key = tuple(sorted(int(i) for i in index_iterable))
        if key in hv_cache:
            return hv_cache[key]
        if len(key) == 0:
            val = 0.0
        else:
            arr = pts[list(key)]
            # compute hypervolume (pygmo expects minimization points and reference)
            val = float(pg.hypervolume(arr).compute(reference_point))
        hv_cache[key] = val
        return val

    # precompute singleton HVs for shortlist
    singleton_hv = {}
    for c in candidates_global:
        singleton_hv[c] = hv_of_indices((c,))

    # initialize heap with (-estimated_gain, tie, idx, last_eval)
    heap = []
    for tie_order, c in enumerate(candidates_global):
        g = float(singleton_hv.get(c, 0.0))
        heap.append((-g, int(tie_order), int(c), 0))
    heapq.heapify(heap)

    selected_list = []
    selected_set = set()
    hv_S = 0.0
    tol = 1e-12

    # CELF main loop
    while len(selected_list) < k_eff and heap:
        neg_est, tie, idx_c, last_eval = heapq.heappop(heap)
        est = -neg_est
        if idx_c in selected_set:
            continue
        if last_eval == len(selected_list):
            # up-to-date estimate, accept
            selected_list.append(int(idx_c))
            selected_set.add(int(idx_c))
            hv_S = hv_of_indices(selected_set)
            continue
        # recompute true marginal
        hv_with = hv_of_indices(selected_set | {int(idx_c)})
        true_gain = hv_with - hv_S
        # push back updated
        heapq.heappush(heap, (-float(true_gain), int(tie), int(idx_c), len(selected_list)))
        # early stopping: if current recomputed top is nonpositive and top estimate also nonpositive
        top_neg = heap[0][0] if heap else 0.0
        if true_gain <= tol and -top_neg <= tol:
            break

    # If not enough selected, supplement deterministically by surrogate over remaining global pool
    if len(selected_list) < k_eff:
        selected_set_local = set(selected_list)
        remaining_global = [i for i in range(N) if i not in selected_set_local]
        if remaining_global:
            rem_pts = pts[remaining_global]
            rem_diffs = np.maximum(reference_point.reshape(1, -1) - rem_pts, eps)
            with np.errstate(divide='ignore', invalid='ignore'):
                rem_logprod = np.sum(dim_weights.reshape(1, -1) * np.log(rem_diffs + 1e-12), axis=1)
            rem_maxgap = np.max(rem_diffs, axis=1)
            # normalize like before
            rlp = rem_logprod
            if np.ptp(rlp) > 0:
                rlp = (rlp - np.min(rlp)) / (np.ptp(rlp) + eps)
            rmg = rem_maxgap / (np.max(rem_maxgap) + eps)
            rem_scores = gamma * rlp + (1 - gamma) * rmg
            order_rem = np.argsort(-rem_scores, kind='stable')
            for idx in order_rem:
                if len(selected_list) >= k_eff:
                    break
                selected_list.append(int(remaining_global[int(idx)]))
        # if still short, pad deterministically by repeating last selected
        while len(selected_list) < k_eff:
            if selected_list:
                selected_list.append(int(selected_list[-1]))
            else:
                # fallback: choose best global surrogate (first cand_idx by surrogate)
                if m > 0:
                    selected_list.append(int(cand_idx[ordered_local_idx[0]]))
                else:
                    selected_list.append(0)

    # ensure uniqueness while preserving order; if duplicates dropped, fill from leftover
    final_sel = selected_list[:k_eff]
    unique_final = []
    seen2 = set()
    for idx in final_sel:
        if idx not in seen2:
            unique_final.append(int(idx))
            seen2.add(idx)
    if len(unique_final) < k_eff:
        # fill with remaining indices deterministically (lowest index first)
        leftover = [i for i in range(N) if i not in seen2]
        for idx in leftover:
            if len(unique_final) >= k_eff:
                break
            unique_final.append(int(idx))
            seen2.add(idx)
        while len(unique_final) < k_eff:
            # repeat last
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


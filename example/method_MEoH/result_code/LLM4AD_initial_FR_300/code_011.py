import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    pts_all = np.asarray(points, dtype=float)
    if pts_all.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N_all, D = pts_all.shape

    if k <= 0:
        return np.zeros((0, D), dtype=float)
    if k >= N_all:
        return pts_all.copy()

    # reference point
    if reference_point is None:
        ref = pts_all.max(axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # ---------------------------
    # 1) Pareto (nondominated) pruning (minimization)
    # ---------------------------
    def nondominated_idx(arr):
        idx = np.lexsort(arr.T[::-1])
        front = []
        for i in idx:
            p = arr[i]
            if not front:
                front.append(i)
                continue
            front_pts = arr[np.array(front)]
            if np.any((front_pts <= p).all(axis=1)):
                continue
            dom = (p <= front_pts).all(axis=1) & (p < front_pts).any(axis=1)
            if np.any(dom):
                keep = np.where(~dom)[0]
                front = list(np.array(front)[keep])
            front.append(i)
        return np.array(front, dtype=int)

    nd_idx = nondominated_idx(pts_all)
    if nd_idx.size < N_all:
        pts = pts_all[nd_idx]
        orig_idx = nd_idx.copy()
    else:
        pts = pts_all
        orig_idx = np.arange(N_all, dtype=int)

    N = pts.shape[0]
    if N <= k:
        return pts_all[orig_idx].copy()

    # ---------------------------
    # 2) Build MC samples (memory-aware)
    # ---------------------------
    low = pts.min(axis=0)
    high = ref.copy()
    span = high - low
    eps = 1e-12
    span[span <= 0] = eps

    # adaptive sample count: small, deterministic, tuned for speed
    M = int(min(8000, max(1200, 200 * k, 600 * D)))
    rng = np.random.default_rng(0)
    samples = rng.random((M, D)) * span + low  # (M, D)

    # compute domination masks in blocks to limit peak memory touched at once
    max_bool = int(8e7)
    block = max(1, int(max_bool // M))
    masks = np.empty((N, M), dtype=bool)
    for s in range(0, N, block):
        e = min(N, s + block)
        chunk = pts[s:e]  # (b, D)
        masks[s:e] = np.all(chunk[:, None, :] <= samples[None, :, :], axis=2)

    init_gain = masks.sum(axis=1).astype(int)

    # If no sample is covered by any point, fallback to box-volume selection
    if init_gain.sum() == 0:
        box_volumes = np.prod(np.maximum(0.0, ref - pts), axis=1)
        chosen_idx = np.argsort(-box_volumes)[:k]
        return pts_all[orig_idx[chosen_idx]].copy()

    # ---------------------------
    # 3) Sparse index lists if beneficial
    # ---------------------------
    avg_dom = init_gain.mean()
    use_index_lists = (avg_dom < (M / 8))  # if typically sparse, lists speed up counting
    index_lists = None
    if use_index_lists:
        index_lists = [None] * N
        for i in range(N):
            if init_gain[i] == 0:
                index_lists[i] = np.empty(0, dtype=int)
            else:
                index_lists[i] = np.nonzero(masks[i])[0]

    # ---------------------------
    # 4) Lazy greedy selection with heap and cached estimates
    # ---------------------------
    heap = [(-g, i) for i, g in enumerate(init_gain)]
    heapq.heapify(heap)
    selected_local = []
    not_covered = np.ones(M, dtype=bool)

    while len(selected_local) < k and heap:
        neg_est, idx = heapq.heappop(heap)
        est = -neg_est
        if est == 0:
            break
        # compute true gain
        if use_index_lists:
            idxs = index_lists[idx]
            if idxs.size == 0:
                true_gain = 0
            else:
                true_gain = int(np.count_nonzero(not_covered[idxs]))
        else:
            true_gain = int(np.count_nonzero(masks[idx] & not_covered))
        if true_gain == 0:
            continue
        if true_gain != est:
            heapq.heappush(heap, (-true_gain, idx))
            continue
        # accept
        selected_local.append(idx)
        if use_index_lists:
            to_cover = index_lists[idx]
            if to_cover.size:
                not_covered[to_cover] = False
        else:
            newly = masks[idx] & not_covered
            if newly.any():
                not_covered[newly] = False
        if not not_covered.any():
            break

    # fill if needed by box-volume
    if len(selected_local) < k:
        remaining = np.setdiff1d(np.arange(N), selected_local, assume_unique=True)
        box_volumes = np.prod(np.maximum(0.0, ref - pts), axis=1)
        extra = remaining[np.argsort(-box_volumes[remaining])[: k - len(selected_local)]]
        selected_local.extend(extra.tolist())

    selected_local = np.array(selected_local, dtype=int)[:k]

    # ---------------------------
    # 5) Shortlist + bounded exact refinement (tiny pool)
    # ---------------------------
    not_sel = np.setdiff1d(np.arange(N), selected_local, assume_unique=True)
    if not_sel.size > 0:
        if use_index_lists:
            marginal = np.array([int(np.count_nonzero(not_covered[index_lists[i]])) for i in not_sel], dtype=int)
        else:
            marginal = (masks[not_sel] & not_covered).sum(axis=1)
        order = np.argsort(-marginal)
        shortlist_extra = not_sel[order][:min(3 * k, not_sel.size)]
        shortlist_idx = np.unique(np.concatenate([selected_local, shortlist_extra]))
    else:
        shortlist_idx = selected_local.copy()

    shortlist = pts[shortlist_idx]
    # map selection positions inside shortlist
    sel_pos = [int(np.where(shortlist_idx == i)[0][0]) for i in selected_local]

    refined_indices = None
    try:
        import pygmo as pg
        have_pygmo = True
    except Exception:
        have_pygmo = False

    if have_pygmo and shortlist.shape[0] > k and shortlist.shape[0] <= 200:
        def exact_hv_from_array(arr):
            if arr.size == 0:
                return 0.0
            hv = pg.hypervolume(arr)
            return hv.compute(ref)
        current_sel = np.array(sel_pos, dtype=int)
        current_set = shortlist[current_sel]
        best_hv = exact_hv_from_array(current_set)
        improved = True
        max_iters = 30
        it = 0
        while improved and it < max_iters:
            it += 1
            improved = False
            for i_pos in range(current_sel.size):
                for cand in range(shortlist.shape[0]):
                    if cand in current_sel:
                        continue
                    trial = current_sel.copy()
                    trial[i_pos] = cand
                    hv = exact_hv_from_array(shortlist[trial])
                    if hv > best_hv + 1e-12:
                        best_hv = hv
                        current_sel = trial
                        improved = True
                        break
                if improved:
                    break
        refined_indices = shortlist_idx[current_sel]
    else:
        refined_indices = shortlist_idx[np.array(sel_pos, dtype=int)]

    final_idx_in_pruned = np.array(refined_indices, dtype=int)[:k]
    final_orig_idx = orig_idx[final_idx_in_pruned]
    return pts_all[final_orig_idx].copy()


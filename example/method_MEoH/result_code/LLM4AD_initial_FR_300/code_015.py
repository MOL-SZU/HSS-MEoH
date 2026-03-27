import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    pts_all = np.asarray(points, dtype=float)
    if pts_all.ndim != 2:
        raise ValueError("points must be a 2‑D array")
    N_all, D = pts_all.shape
    if k <= 0 or k > N_all:
        raise ValueError("k must satisfy 1 ≤ k ≤ N")

    # reference point
    if reference_point is None:
        reference_point = pts_all.max(axis=0) * 1.1
    ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # -------------------------
    # 1) Pareto (nondominated) pruning (minimization assumed)
    # -------------------------
    def nondominated_idx(arr):
        # returns indices (into arr) that are nondominated (minimize)
        idx = np.lexsort(arr.T[::-1])
        front = []
        for i in idx:
            p = arr[i]
            if not front:
                front.append(i)
                continue
            front_pts = arr[np.array(front)]
            # if some front point <= p (i.e., dominates or equal) -> skip p
            if np.any((front_pts <= p).all(axis=1)):
                continue
            # remove front points dominated by p
            dom = (p <= front_pts).all(axis=1) & (p < front_pts).any(axis=1)
            if np.any(dom):
                keep = np.where(~dom)[0]
                front = list(np.array(front)[keep])
            front.append(i)
        return np.array(front, dtype=int)

    nd_idx = nondominated_idx(pts_all)
    if nd_idx.size < N_all:
        pts = pts_all[nd_idx]
        orig_map = nd_idx  # map from pts indices to original pts_all indices
    else:
        pts = pts_all
        orig_map = np.arange(N_all, dtype=int)
    N = pts.shape[0]

    if N <= k:
        # return corresponding original points
        return pts_all[orig_map].copy()

    # -------------------------
    # 2) Shared Monte-Carlo samples (blockwise)
    # -------------------------
    low = pts.min(axis=0)
    high = ref.copy()
    span = high - low
    eps = 1e-12
    span[span <= 0] = eps
    M = int(min(12000, max(2000, 300 * k, 1000 * D)))
    rng = np.random.default_rng(0)
    samples = rng.random((M, D)) * span + low  # shape (M, D)

    # -------------------------
    # 3) Build sparse per-point sample index lists (memory-friendly, blockwise)
    # -------------------------
    # To avoid a giant boolean matrix, keep for each point the indices of samples it dominates.
    per_point_inds = [None] * N
    max_bool = 8e7
    block = max(1, int(max_bool // M))
    for s in range(0, N, block):
        e = min(N, s + block)
        chunk = pts[s:e]                         # (b, D)
        # comparison: chunk[:, None, :] <= samples[None, :, :]
        comp = np.all(chunk[:, None, :] <= samples[None, :, :], axis=2)  # (b, M)
        for i in range(chunk.shape[0]):
            inds = np.nonzero(comp[i])[0]
            per_point_inds[s + i] = inds.astype(np.int32)

    # -------------------------
    # 4) Precompute box volumes for tie-breaking
    # -------------------------
    box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)

    # -------------------------
    # 5) Lazy-greedy selection with heap (use -gain, -box_vol to tie-break larger volume)
    # -------------------------
    init_gain = np.array([inds.size for inds in per_point_inds], dtype=np.int32)
    heap = [(-int(g), -float(box_vol[i]), int(i)) for i, g in enumerate(init_gain)]
    heapq.heapify(heap)

    selected_idx = []
    covered = np.zeros(M, dtype=bool)

    while len(selected_idx) < k and heap:
        neg_gain, neg_vol, idx = heapq.heappop(heap)
        # compute true gain quickly using the sparse sample indices
        inds = per_point_inds[idx]
        if inds.size == 0:
            true_gain = 0
        else:
            true_gain = int(np.count_nonzero(~covered[inds]))
        if -neg_gain != true_gain:
            # reinsert with updated gain (tie-break by box volume unchanged)
            heapq.heappush(heap, (-true_gain, neg_vol, idx))
            continue
        # select idx
        selected_idx.append(int(idx))
        if true_gain > 0:
            covered[inds] = covered[inds] | True  # mark covered samples

    # If heap exhausted before k (rare), fill with largest box volumes among remaining
    if len(selected_idx) < k:
        remaining = np.setdiff1d(np.arange(N), np.array(selected_idx, dtype=int), assume_unique=True)
        extra = remaining[np.argsort(-box_vol[remaining])[:k - len(selected_idx)]]
        selected_idx.extend(extra.tolist())

    selected_idx = np.array(selected_idx, dtype=int)[:k]

    # -------------------------
    # 6) Build a small shortlist for refinement (top marginal candidates + selected)
    # -------------------------
    not_sel = np.setdiff1d(np.arange(N), selected_idx, assume_unique=True)
    # recompute marginal gains for not selected (cheap via sparse lists)
    marg = []
    for i in not_sel:
        inds = per_point_inds[i]
        marg.append(int(np.count_nonzero(~covered[inds])) if inds.size > 0 else 0)
    marg = np.array(marg, dtype=int)
    order = np.argsort(-marg)
    shortlist_extra = not_sel[order][:min(3 * k, not_sel.size)]
    shortlist_idx = np.unique(np.concatenate([selected_idx, shortlist_extra]))
    shortlist = pts[shortlist_idx]
    # positions of selected inside shortlist
    sel_pos = [int(np.where(shortlist_idx == i)[0][0]) for i in selected_idx]

    # -------------------------
    # 7) Optional exact swap refinement (bounded) using pygmo if available
    # -------------------------
    try:
        import pygmo as pg
        have_pygmo = True
    except Exception:
        have_pygmo = False

    refined_idx_in_pts = np.array(selected_idx, dtype=int)
    if have_pygmo and shortlist.shape[0] > k:
        def exact_hv_array(arr):
            if arr.size == 0:
                return 0.0
            hv = pg.hypervolume(arr)
            return float(hv.compute(ref))

        # current set in terms of shortlist positions
        sel_pos = list(sel_pos)
        current_set = shortlist[sel_pos]
        best_hv = exact_hv_array(current_set)
        improved = True
        max_iters = 60
        it = 0
        # To reduce HV calls, break early when no improvement in full pass
        while improved and it < max_iters:
            improved = False
            it += 1
            # iterate over selected positions and attempt to swap with non-selected shortlist points
            for i_sel_idx, s_pos in enumerate(sel_pos):
                # try candidates ordered by box volume (descending) to reach improvements quickly
                candidates = [c for c in range(shortlist.shape[0]) if c not in sel_pos]
                # sort candidates by local box volume (w.r.t. shortlist point)
                cand_vols = np.prod(np.maximum(0.0, ref - shortlist[candidates]), axis=1)
                cand_order = np.argsort(-cand_vols)
                for c in [candidates[j] for j in cand_order]:
                    trial_pos = sel_pos.copy()
                    trial_pos[i_sel_idx] = c
                    trial_set = shortlist[trial_pos]
                    hv = exact_hv_array(trial_set)
                    if hv > best_hv + 1e-12:
                        best_hv = hv
                        sel_pos = trial_pos
                        improved = True
                        break
                if improved:
                    break
        refined_idx_in_pts = shortlist_idx[np.array(sel_pos, dtype=int)]
    else:
        refined_idx_in_pts = np.array(selected_idx, dtype=int)

    # -------------------------
    # 8) Map back to original points and return exactly k points
    # -------------------------
    refined_idx_in_pts = np.array(refined_idx_in_pts, dtype=int)[:k]
    # map to original indices if pruning happened
    result = pts_all[orig_map[refined_idx_in_pts]]
    return result.copy()


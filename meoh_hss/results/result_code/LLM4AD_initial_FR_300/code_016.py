import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    pts_all = np.asarray(points, dtype=float)
    if pts_all.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N_all, D = pts_all.shape

    if k <= 0:
        return np.zeros((0, D))
    if k >= N_all:
        return pts_all.copy()

    # reference point
    if reference_point is None:
        ref = pts_all.max(axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float)
        if ref.shape != (D,):
            raise ValueError("reference_point must have shape (D,)")

    # ------------------------------------------------------------------
    # 1) Pareto (nondominated) pruning (minimization convention)
    # ------------------------------------------------------------------
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
        orig_map = nd_idx
    else:
        pts = pts_all
        orig_map = np.arange(N_all, dtype=int)
    N = pts.shape[0]
    if N <= k:
        return pts.copy()

    # ------------------------------------------------------------------
    # 2) Monte-Carlo sampling domain and count
    # ------------------------------------------------------------------
    low = pts.min(axis=0)
    high = ref.copy()
    span = high - low
    eps = 1e-12
    span[span <= 0] = eps

    # sample count: scale with k and D, but cap
    M = int(min(12000, max(2000, 350 * k, 1000 * D)))
    rng = np.random.default_rng(0)
    samples = rng.random((M, D)) * span + low
    domain_vol = float(np.prod(span))

    # ------------------------------------------------------------------
    # 3) Build compact per-point dominated sample index lists in memory-safe blocks
    # ------------------------------------------------------------------
    max_bool = int(8e7)  # control block size relative to M
    block = max(1, int(max_bool // M))
    point_sample_lists = [None] * N
    init_gain = np.empty(N, dtype=int)

    for s in range(0, N, block):
        e = min(N, s + block)
        chunk = pts[s:e]                      # (b, D)
        # compute boolean block (b, M)
        bmask = np.all(chunk[:, None, :] <= samples[None, :, :], axis=2)
        for i in range(s, e):
            row = bmask[i - s]
            if row.any():
                idxs = np.nonzero(row)[0].astype(np.int32)
            else:
                idxs = np.empty(0, dtype=np.int32)
            point_sample_lists[i] = idxs
            init_gain[i] = idxs.size

    # ------------------------------------------------------------------
    # 4) Lazy-greedy selection using sparse lists and max-heap
    # ------------------------------------------------------------------
    heap = [(-int(g), int(i)) for i, g in enumerate(init_gain)]
    heapq.heapify(heap)

    selected_local = []
    covered = np.zeros(M, dtype=bool)

    while len(selected_local) < k and heap:
        neg_gain, idx = heapq.heappop(heap)
        lst = point_sample_lists[idx]
        if lst.size == 0:
            true_gain = 0
        else:
            # count uncovered in this list quickly
            true_gain = int(np.count_nonzero(~covered[lst]))
        if true_gain == 0:
            # no contribution left; skip
            continue
        if -neg_gain != true_gain:
            # push corrected value
            heapq.heappush(heap, (-true_gain, idx))
            continue
        # accept
        selected_local.append(int(idx))
        # mark covered
        if lst.size:
            covered[lst] = True

    # If exhausted before k, fill by box-volume heuristic on remaining
    if len(selected_local) < k:
        remaining = np.setdiff1d(np.arange(N, dtype=int), np.array(selected_local, dtype=int), assume_unique=True)
        if remaining.size > 0:
            box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)
            need = k - len(selected_local)
            extra = remaining[np.argsort(-box_vol[remaining])[:need]]
            for x in extra.tolist():
                selected_local.append(int(x))

    selected_local = selected_local[:k]
    selected_local = np.array(selected_local, dtype=int)

    # ------------------------------------------------------------------
    # 5) Build shortlist for bounded exact polish: selected + top marginals
    # ------------------------------------------------------------------
    not_sel = np.setdiff1d(np.arange(N, dtype=int), selected_local, assume_unique=True)
    shortlist_extra = np.array([], dtype=int)
    if not_sel.size > 0:
        # compute marginal uncovered counts relative to currently covered
        marg = np.empty(not_sel.size, dtype=int)
        for i, idx in enumerate(not_sel):
            lst = point_sample_lists[idx]
            if lst.size == 0:
                marg[i] = 0
            else:
                marg[i] = int(np.count_nonzero(~covered[lst]))
        order = np.argsort(-marg)
        extra_count = min(max(2 * k, 3 * k), not_sel.size)
        shortlist_extra = not_sel[order][:extra_count]

    shortlist_idx = np.unique(np.concatenate([selected_local, shortlist_extra])).astype(int)
    shortlist_pts = pts[shortlist_idx]
    # map selected positions into shortlist indices
    sel_pos = [int(np.where(shortlist_idx == i)[0][0]) for i in selected_local]

    # ------------------------------------------------------------------
    # 6) Bounded exact swap refinement (pygmo if available) else mask-based polishing
    # ------------------------------------------------------------------
    try:
        import pygmo as pg
        have_pygmo = True
    except Exception:
        have_pygmo = False

    def exact_hv_of_indices_local(idx_list):
        if len(idx_list) == 0:
            return 0.0
        return float(pg.hypervolume(shortlist_pts[idx_list]).compute(ref))

    def approx_hv_of_indices_local(idx_list):
        if len(idx_list) == 0:
            return 0.0
        # OR the masks rows corresponding to shortlist candidates
        rows = []
        for pos in idx_list:
            global_idx = shortlist_idx[int(pos)]
            lst = point_sample_lists[global_idx]
            rows.append(lst)
        if not rows:
            return 0.0
        # build boolean union efficiently
        covered_local = np.zeros(M, dtype=bool)
        for r in rows:
            if r.size:
                covered_local[r] = True
        frac = float(covered_local.sum()) / float(M)
        return frac * domain_vol

    refined_sel_pos = sel_pos.copy()
    max_iters = 50
    if have_pygmo and shortlist_pts.shape[0] > k:
        current_hv = exact_hv_of_indices_local(refined_sel_pos)
        improved = True
        it = 0
        while improved and it < max_iters:
            it += 1
            improved = False
            for si in range(len(refined_sel_pos)):
                best_swap = None
                best_hv = current_hv
                for cand in range(shortlist_pts.shape[0]):
                    if cand in refined_sel_pos:
                        continue
                    trial = refined_sel_pos.copy()
                    trial[si] = cand
                    hv = exact_hv_of_indices_local(trial)
                    if hv > best_hv + 1e-12:
                        best_hv = hv
                        best_swap = trial
                if best_swap is not None:
                    refined_sel_pos = best_swap
                    current_hv = best_hv
                    improved = True
                    break
    else:
        current_hv = approx_hv_of_indices_local(refined_sel_pos)
        improved = True
        it = 0
        while improved and it < max_iters:
            it += 1
            improved = False
            for si in range(len(refined_sel_pos)):
                best_swap = None
                best_hv = current_hv
                # candidate pool: top by marginal uncovered samples relative to current covered by refined_sel_pos
                chosen_set = set(refined_sel_pos)
                cand_list = [i for i in range(shortlist_pts.shape[0]) if i not in chosen_set]
                if not cand_list:
                    continue
                # compute covered by current refined selection to speed marginals
                cur_rows = [point_sample_lists[int(shortlist_idx[int(p)])] for p in refined_sel_pos]
                cur_cov = np.zeros(M, dtype=bool)
                for r in cur_rows:
                    if r.size:
                        cur_cov[r] = True
                # compute marginals for candidates
                cand_abs = shortlist_idx[cand_list]
                cand_marg = []
                for a in cand_abs:
                    lst = point_sample_lists[a]
                    if lst.size:
                        cand_marg.append(int(np.count_nonzero(~cur_cov[lst])))
                    else:
                        cand_marg.append(0)
                top_k = min(16, len(cand_list))
                top_order = np.argsort(-np.array(cand_marg))[:top_k]
                top_candidates = [cand_list[i] for i in top_order]
                for cand_pos in top_candidates:
                    trial = refined_sel_pos.copy()
                    trial[si] = cand_pos
                    hv = approx_hv_of_indices_local(trial)
                    if hv > best_hv + 1e-12:
                        best_hv = hv
                        best_swap = trial
                if best_swap is not None:
                    refined_sel_pos = best_swap
                    current_hv = best_hv
                    improved = True
                    break

    # map refined positions in shortlist back to global point indices
    final_local_idx = shortlist_idx[np.array(refined_sel_pos, dtype=int)]
    # ensure uniqueness & exact count
    final_local_idx = list(dict.fromkeys([int(i) for i in final_local_idx]))
    if len(final_local_idx) < k:
        already = set(final_local_idx)
        remaining = [i for i in range(N) if i not in already]
        if remaining:
            rem_vols = np.prod(np.maximum(0.0, ref - pts[remaining]), axis=1)
            need = k - len(final_local_idx)
            order = np.argsort(-rem_vols)[:need]
            for o in order:
                final_local_idx.append(int(remaining[o]))
    if len(final_local_idx) > k:
        final_local_idx = final_local_idx[:k]
    final_local_idx = np.array(final_local_idx, dtype=int)

    # map back to original indices
    final_global_idx = orig_map[final_local_idx]

    return pts_all[final_global_idx].copy()


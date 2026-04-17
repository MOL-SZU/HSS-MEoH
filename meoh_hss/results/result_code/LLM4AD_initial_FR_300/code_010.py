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

    # ---------------------------
    # 1) Pareto (nondominated) pruning (minimization assumption)
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
            dominated_mask = (p <= front_pts).all(axis=1) & (p < front_pts).any(axis=1)
            if np.any(dominated_mask):
                keep = np.where(~dominated_mask)[0]
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
        return pts.copy()

    # ---------------------------
    # 2) Build adaptive MC samples (memory-aware) with jittered uniform sampling
    # ---------------------------
    low = pts.min(axis=0)
    span = ref - low
    eps = 1e-12
    span[span <= 0] = eps

    M = int(min(12000, max(1800, 280 * k, 900 * D, 8000 if N > 8000 else 1500)))
    rng = np.random.default_rng(0)  # deterministic
    samples = rng.random((M, D)) * span + low

    # ---------------------------
    # 3) Precompute domination masks in memory‑safe blocks
    # ---------------------------
    max_bool = int(8e7)
    block = max(1, int(max_bool // M))
    masks = np.empty((N, M), dtype=bool)
    for s in range(0, N, block):
        e = min(N, s + block)
        chunk = pts[s:e]  # (b, D)
        masks[s:e] = np.all(chunk[:, None, :] <= samples[None, :, :], axis=2)

    init_gain = masks.sum(axis=1).astype(int)

    # If no sample covered, fallback to box-volume selection
    if init_gain.sum() == 0:
        box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)
        chosen_idx = np.argsort(-box_vol)[:k]
        return pts_all[orig_idx[chosen_idx]].copy()

    # ---------------------------
    # 4) Build sparse index lists when beneficial
    # ---------------------------
    avg_dom = init_gain.mean()
    use_index_lists = (avg_dom < (M / 8)) and (M > 500)
    if use_index_lists:
        index_lists = [None] * N
        for i in range(N):
            if init_gain[i] == 0:
                index_lists[i] = np.empty(0, dtype=int)
            else:
                index_lists[i] = np.nonzero(masks[i])[0]
    else:
        index_lists = None

    # precompute box volumes for tie-breaking
    box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)
    bv_norm = box_vol / (box_vol.max() + 1e-30)

    # ---------------------------
    # 5) Lazy greedy selection with heap and incremental coverage updates
    # ---------------------------
    heap = [(-int(init_gain[i]), -float(bv_norm[i]), int(i)) for i in range(N)]
    heapq.heapify(heap)

    selected_local = []
    covered = np.zeros(M, dtype=bool)

    if use_index_lists:
        def true_gain_of(i):
            idxs = index_lists[i]
            if idxs.size == 0:
                return 0
            return int(np.count_nonzero(~covered[idxs]))
    else:
        def true_gain_of(i):
            return int(np.count_nonzero(masks[i] & ~covered))

    while len(selected_local) < k and heap:
        neg_gain, neg_bv, idx = heapq.heappop(heap)
        est_gain = int(-neg_gain)
        if est_gain == 0:
            break
        tg = true_gain_of(idx)
        if tg == 0:
            continue
        if tg != est_gain:
            heapq.heappush(heap, (-tg, neg_bv, idx))
            continue
        selected_local.append(int(idx))
        if use_index_lists:
            to_cover = index_lists[idx]
            if to_cover.size:
                covered[to_cover] = True
        else:
            newly = masks[idx] & ~covered
            if newly.any():
                covered |= newly
        if not (~covered).any():
            break

    # if insufficient selected, fill by box-volume among remaining
    if len(selected_local) < k:
        remaining = np.setdiff1d(np.arange(N), np.array(selected_local, dtype=int), assume_unique=True)
        if remaining.size > 0:
            extra = remaining[np.argsort(-box_vol[remaining])[: k - len(selected_local)]]
            selected_local.extend(int(x) for x in extra.tolist())

    selected_local = np.array(selected_local, dtype=int)[:k]

    # ---------------------------
    # 6) Shortlist composition with rebalanced score (different parameterisation)
    # ---------------------------
    not_sel = np.setdiff1d(np.arange(N, dtype=int), selected_local, assume_unique=True)
    if not_sel.size > 0:
        if use_index_lists:
            marginal = np.array([int(np.count_nonzero(~covered[index_lists[i]])) for i in not_sel], dtype=int)
        else:
            marginal = (masks[not_sel] & ~covered).sum(axis=1).astype(int)

        # New scoring: normalize marginals and box_vol, combine with equal weight,
        # apply mild power transform to box_vol to favor robust large-volume points
        marg_max = float(marginal.max()) if marginal.size else 0.0
        if marg_max <= 0:
            marginal_norm = np.zeros_like(marginal, dtype=float)
        else:
            marginal_norm = marginal.astype(float) / (marg_max + 1e-30)

        bv_not = box_vol[not_sel]
        bv_max = float(bv_not.max()) if bv_not.size else 0.0
        if bv_max <= 0:
            bv_norm_not = np.zeros_like(bv_not, dtype=float)
        else:
            bv_norm_not = (bv_not / (bv_max + 1e-30)) ** 0.7  # mild power (0.7)

        # combine with equal weighting (0.5/0.5)
        score = 0.5 * marginal_norm + 0.5 * bv_norm_not

        order = np.argsort(-score)
        shortlist_extra = not_sel[order][:min(6 * k, not_sel.size)]
        shortlist_idx = np.unique(np.concatenate([selected_local, shortlist_extra]))
    else:
        shortlist_idx = selected_local.copy()

    shortlist_pts = pts[shortlist_idx]
    sel_pos = [int(np.where(shortlist_idx == i)[0][0]) for i in selected_local]

    # ---------------------------
    # 7) Bounded exact refinement (prefer pygmo); otherwise mask-based approximate local swaps
    # ---------------------------
    refined_positions = np.array(sel_pos, dtype=int)
    try:
        import pygmo as pg
        have_pygmo = True
    except Exception:
        have_pygmo = False

    domain_vol = float(np.prod(span))
    def approx_hv_by_mask_indices(idx_list):
        if idx_list.size == 0:
            return 0.0
        rows = masks[shortlist_idx[idx_list]]
        dom = np.any(rows, axis=0)
        frac = float(dom.sum()) / float(M)
        return frac * domain_vol

    max_iters = 40
    if have_pygmo and shortlist_pts.shape[0] > k and shortlist_pts.shape[0] <= 250:
        def exact_hv_of_positions(pos_list):
            if pos_list.size == 0:
                return 0.0
            arr = shortlist_pts[pos_list]
            return float(pg.hypervolume(arr).compute(ref))

        current_hv = exact_hv_of_positions(refined_positions)
        improved = True
        it = 0
        while improved and it < max_iters:
            it += 1
            improved = False
            for sel_i in range(refined_positions.size):
                best_swap = None
                best_hv = current_hv
                for cand in range(shortlist_pts.shape[0]):
                    if cand in refined_positions:
                        continue
                    trial = refined_positions.copy()
                    trial[sel_i] = cand
                    hv = exact_hv_of_positions(trial)
                    if hv > best_hv + 1e-12:
                        best_hv = hv
                        best_swap = trial
                        break
                if best_swap is not None:
                    refined_positions = best_swap
                    current_hv = best_hv
                    improved = True
                    break
    else:
        current_hv = approx_hv_by_mask_indices(refined_positions)
        improved = True
        it = 0
        while improved and it < max_iters:
            it += 1
            improved = False
            for sel_i in range(refined_positions.size):
                best_swap = None
                best_hv = current_hv
                cand_list = [i for i in range(shortlist_pts.shape[0]) if i not in refined_positions]
                if not cand_list:
                    continue
                cand_abs = shortlist_idx[cand_list]
                cand_marg = (masks[cand_abs] & ~covered).sum(axis=1)
                top_k = min(20, len(cand_list))
                top_order = np.argsort(-cand_marg)[:top_k]
                top_candidates = [cand_list[ii] for ii in top_order]
                for cand_pos in top_candidates:
                    trial = refined_positions.copy()
                    trial[sel_i] = cand_pos
                    hv = approx_hv_by_mask_indices(trial)
                    if hv > best_hv + 1e-12:
                        best_hv = hv
                        best_swap = trial
                        break
                if best_swap is not None:
                    refined_positions = best_swap
                    current_hv = best_hv
                    improved = True
                    break

    # map refined positions inside shortlist back to indices in pts
    final_local_idx = shortlist_idx[np.array(refined_positions, dtype=int)]
    # ensure uniqueness & exact count k
    seen = []
    for i in final_local_idx.tolist():
        if i not in seen:
            seen.append(int(i))
    if len(seen) < k:
        already = set(seen)
        remaining = [i for i in range(N) if i not in already]
        if remaining:
            rem_vols = box_vol[remaining]
            need = k - len(seen)
            order = np.argsort(-rem_vols)[:need]
            for o in order:
                seen.append(int(remaining[o]))
    if len(seen) > k:
        seen = seen[:k]
    final_local_idx = np.array(seen, dtype=int)

    # map back to original indices
    final_global_idx = orig_idx[final_local_idx]

    return pts_all[final_global_idx].copy()


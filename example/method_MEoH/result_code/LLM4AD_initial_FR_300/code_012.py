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
    if N_all == 0:
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

    # Remove points that cannot contribute (any dim > ref)
    valid_mask = np.all(pts_all <= ref, axis=1)
    if not np.any(valid_mask):
        # fallback: choose k nearest to reference
        dists = np.linalg.norm(pts_all - ref, axis=1)
        idx = np.argsort(dists)[:min(k, N_all)]
        return pts_all[idx].copy()
    pts_valid = pts_all[valid_mask]
    orig_idx_map = np.nonzero(valid_mask)[0]
    N = pts_valid.shape[0]
    if k >= N:
        return pts_valid.copy()

    # ---------------------------
    # 1) Nondominated pruning (minimization assumed)
    # ---------------------------
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
            # if some front point dominates p -> skip
            if np.any((front_pts <= p).all(axis=1)):
                continue
            # remove front points dominated by p
            dom = (p <= front_pts).all(axis=1) & (p < front_pts).any(axis=1)
            if np.any(dom):
                keep = np.where(~dom)[0]
                front = list(np.array(front)[keep])
            front.append(i)
        return np.array(front, dtype=int)

    nd_idx = nondominated_idx(pts_valid)
    if nd_idx.size < N:
        pts = pts_valid[nd_idx]
        orig_idx = orig_idx_map[nd_idx]
    else:
        pts = pts_valid
        orig_idx = orig_idx_map.copy()
    N = pts.shape[0]
    if k >= N:
        return pts.copy()

    # ---------------------------
    # 2) Adaptive jittered Monte-Carlo sampling (deterministic)
    # ---------------------------
    low = pts.min(axis=0)
    span = ref - low
    eps = 1e-12
    span[span <= 0] = eps

    M = int(min(12000, max(1800, 300 * k, 900 * D, 1500)))
    rng = np.random.default_rng(0)
    # jittered sampling (simple stratified jitter)
    base = rng.random((M, D))
    jitter = (rng.random((M, D)) - 0.5) * (1.0 / max(1, int(np.sqrt(M))))
    samples = np.clip(base + jitter, 0.0, 1.0) * span + low
    domain_vol = float(np.prod(span))

    # ---------------------------
    # 3) Build masks in memory-safe blocks and optionally sparse index lists
    # ---------------------------
    max_bool = int(8e7)
    block = max(1, int(max_bool // M))
    masks = np.empty((N, M), dtype=bool)
    for s in range(0, N, block):
        e = min(N, s + block)
        chunk = pts[s:e]  # (b, D)
        comp = np.all(chunk[:, None, :] <= samples[None, :, :], axis=2)  # (b, M)
        masks[s:e] = comp

    init_gain = masks.sum(axis=1).astype(int)
    if init_gain.sum() == 0:
        # nothing captured by samples -> fallback by box volume
        box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)
        sel = np.argsort(-box_vol)[:k]
        return pts_all[orig_idx[sel]].copy()

    avg_dom = float(init_gain.mean())
    use_index_lists = (avg_dom < (M / 8)) and (M > 500)
    if use_index_lists:
        index_lists = [None] * N
        for i in range(N):
            if init_gain[i] == 0:
                index_lists[i] = np.empty(0, dtype=np.int32)
            else:
                index_lists[i] = np.nonzero(masks[i])[0].astype(np.int32)
    else:
        index_lists = None

    # precompute box volumes for tie-breaking
    box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)
    bv_norm = box_vol / (box_vol.max() + 1e-30)

    # ---------------------------
    # 4) Lazy greedy (heap) selection with incremental true-gain updates
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
        # accept
        selected_local.append(int(idx))
        # update covered
        if use_index_lists:
            tl = index_lists[idx]
            if tl.size:
                covered[tl] = True
        else:
            newly = masks[idx] & ~covered
            if newly.any():
                covered |= newly
        if not (~covered).any():
            break

    # if insufficient, fill by box volume among remaining
    if len(selected_local) < k:
        remaining = np.setdiff1d(np.arange(N), np.array(selected_local, dtype=int), assume_unique=True)
        if remaining.size > 0:
            extra = remaining[np.argsort(-box_vol[remaining])[: k - len(selected_local)]]
            selected_local.extend(int(x) for x in extra.tolist())

    selected_local = np.array(selected_local, dtype=int)[:k]

    # ---------------------------
    # 5) Shortlist for bounded refinement (combine marginals + box volume)
    # ---------------------------
    not_sel = np.setdiff1d(np.arange(N, dtype=int), selected_local, assume_unique=True)
    if not_sel.size > 0:
        if use_index_lists:
            marginal = np.array([int(np.count_nonzero(~covered[index_lists[i]])) for i in not_sel], dtype=int)
        else:
            marginal = (masks[not_sel] & ~covered).sum(axis=1)
        score = marginal.astype(float) + 0.25 * (box_vol[not_sel] / (box_vol.max() + 1e-30))
        order = np.argsort(-score)
        shortlist_extra = not_sel[order][:min(5 * k, not_sel.size)]
        shortlist_idx = np.unique(np.concatenate([selected_local, shortlist_extra]))
    else:
        shortlist_idx = selected_local.copy()

    shortlist_pts = pts[shortlist_idx]
    sel_pos = [int(np.where(shortlist_idx == i)[0][0]) for i in selected_local]

    # ---------------------------
    # 6) Bounded refinement: exact HV via pygmo if available and small, otherwise mask-based polish
    # ---------------------------
    try:
        import pygmo as pg
        have_pygmo = True
    except Exception:
        have_pygmo = False

    refined_positions = np.array(sel_pos, dtype=int)
    max_iters = 60

    def approx_hv_by_masks(pos_list):
        if pos_list.size == 0:
            return 0.0
        rows = masks[shortlist_idx[pos_list]]
        dom = np.any(rows, axis=0)
        frac = float(dom.sum()) / float(M)
        return frac * domain_vol

    if have_pygmo and shortlist_pts.shape[0] > k and shortlist_pts.shape[0] <= 400:
        def exact_hv_positions(pos_list):
            if pos_list.size == 0:
                return 0.0
            arr = shortlist_pts[pos_list]
            return float(pg.hypervolume(arr).compute(ref))
        current_hv = exact_hv_positions(refined_positions)
        it = 0
        improved = True
        while improved and it < max_iters:
            it += 1
            improved = False
            # try single swaps
            for i_sel in range(refined_positions.size):
                for cand in range(shortlist_pts.shape[0]):
                    if cand in refined_positions:
                        continue
                    trial = refined_positions.copy()
                    trial[i_sel] = cand
                    hv = exact_hv_positions(trial)
                    if hv > current_hv + 1e-12:
                        current_hv = hv
                        refined_positions = trial
                        improved = True
                        break
                if improved:
                    break
    else:
        current_hv = approx_hv_by_masks(refined_positions)
        it = 0
        improved = True
        while improved and it < max_iters:
            it += 1
            improved = False
            for i_sel in range(refined_positions.size):
                cand_list = [i for i in range(shortlist_pts.shape[0]) if i not in refined_positions]
                if not cand_list:
                    continue
                # evaluate top candidates by marginal contribution
                cand_abs = shortlist_idx[cand_list]
                if cand_abs.size == 0:
                    continue
                if use_index_lists:
                    cand_marg = np.array([int(np.count_nonzero(~covered[index_lists[int(a)]])) for a in cand_abs], dtype=int)
                else:
                    cand_marg = (masks[cand_abs] & ~covered).sum(axis=1)
                top_k = min(30, len(cand_list))
                top_order = np.argsort(-cand_marg)[:top_k]
                for idx_in_top in top_order:
                    cand_pos = cand_list[int(idx_in_top)]
                    trial = refined_positions.copy()
                    trial[i_sel] = cand_pos
                    hv = approx_hv_by_masks(trial)
                    if hv > current_hv + 1e-12:
                        current_hv = hv
                        refined_positions = trial
                        improved = True
                        break
                if improved:
                    break

    final_local_idx = shortlist_idx[np.array(refined_positions, dtype=int)]

    # ensure uniqueness and exact count k
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

    # map back to original indices and return
    final_global_idx = orig_idx[final_local_idx]
    return pts_all[final_global_idx].copy()


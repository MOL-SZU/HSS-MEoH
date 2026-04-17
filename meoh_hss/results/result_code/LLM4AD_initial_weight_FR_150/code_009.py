import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    N, D = pts.shape

    if k <= 0:
        return np.zeros((0, D), dtype=pts.dtype)
    if k >= N:
        return pts.copy()

    # reference handling
    if reference_point is None:
        ref = pts.max(axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float)
        if ref.ndim == 0:
            ref = np.full((D,), float(ref))
        if ref.shape != (D,):
            raise ValueError("reference_point must have shape (D,)")
        max_p = pts.max(axis=0)
        ref = np.maximum(ref, max_p + 1e-12)

    # compute per-point rect volumes w.r.t ref, remove zero-volume points
    rect_sizes_all = np.prod(np.clip(ref - pts, a_min=0.0, a_max=None), axis=1)
    meaningful_mask = rect_sizes_all > 0
    if not np.any(meaningful_mask):
        idx = np.arange(N)[:k]
        return pts[idx].copy()
    idx_all = np.nonzero(meaningful_mask)[0]
    pts_m = pts[idx_all]
    rect_sizes = rect_sizes_all[idx_all]
    Nm = pts_m.shape[0]

    # nondominated filtering (minimization)
    def nondominated_mask(X):
        n = X.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            xi = X[i]
            leq = np.all(X <= xi, axis=1)
            lt = np.any(X < xi, axis=1)
            dominated_by = leq & lt
            dominated_by[i] = False
            if np.any(dominated_by):
                keep[i] = False
        return keep

    nd_mask = nondominated_mask(pts_m)
    nd_idx_rel = np.nonzero(nd_mask)[0]
    candidate_idx_rel = nd_idx_rel if nd_idx_rel.size >= max(1, min(Nm, k)) else np.arange(Nm)

    # Trivial if candidates <= k
    if candidate_idx_rel.size <= k:
        sel_rel = candidate_idx_rel.tolist()
        if len(sel_rel) < k:
            remaining = np.setdiff1d(np.arange(Nm), np.array(sel_rel, dtype=int), assume_unique=True)
            add_rel = remaining[np.argsort(-rect_sizes[remaining])][: (k - len(sel_rel))]
            sel_rel.extend(add_rel.tolist())
        selected_rel = np.array(sel_rel, dtype=int)[:k]
        selected_orig = idx_all[selected_rel]
        return pts[selected_orig].copy()

    # shortlist selection proportional to k (c * k cap)
    c = 6
    shortlist_size = int(min(Nm, max(c * k, 50)))
    cand = np.array(candidate_idx_rel, dtype=int)
    order_by_rect = cand[np.argsort(-rect_sizes[cand])]
    shortlist = order_by_rect[:shortlist_size]

    # Prepare HV computation (try pygmo exact, else MC with shared samples)
    hv_cache = {tuple(): 0.0}
    try:
        import pygmo as pg
        def hv_of_set_rel(rel_indices):
            rel_indices = np.asarray(rel_indices, dtype=int)
            if rel_indices.size == 0:
                return 0.0
            data = pts_m[rel_indices]
            hv = pg.hypervolume(data)
            return float(hv.compute(ref))
    except Exception:
        rng_local = np.random.default_rng(1234567)
        lower_glob = pts_m.min(axis=0)
        upper_glob = ref.copy()
        zero_r = (upper_glob - lower_glob) <= 0
        if np.any(zero_r):
            upper_glob[zero_r] = lower_glob[zero_r] + 1e-9
        # adaptive global sample size
        if D <= 3:
            M_global = 6000
        elif D <= 6:
            M_global = 3000
        else:
            M_global = 1500
        # ensure not too small
        M_global = max(1000, int(M_global))
        samples_unit = rng_local.random((M_global, D))
        samples = samples_unit * (upper_glob - lower_glob)[None, :] + lower_glob[None, :]
        box_vol_global = float(np.prod(upper_glob - lower_glob))

        def hv_of_set_rel(rel_indices):
            rel_indices = np.asarray(rel_indices, dtype=int)
            if rel_indices.size == 0:
                return 0.0
            data = pts_m[rel_indices]
            dominated = np.zeros(M_global, dtype=bool)
            # vectorized per point accumulation
            for p in data:
                dominated |= np.all(samples >= p, axis=1)
                if dominated.all():
                    break
            return float(box_vol_global * dominated.mean())

    def cached_hv(rel_tuple):
        key = tuple(sorted(int(x) for x in rel_tuple))
        if key in hv_cache:
            return hv_cache[key]
        v = hv_of_set_rel(np.array(key, dtype=int))
        hv_cache[key] = float(v)
        return float(v)

    # Lazy greedy with stamping (hv_version)
    import heapq
    hv_version = 0
    S = []
    hv_S = 0.0
    # initial estimates: use rect_sizes as cheap surrogate
    heap = []
    in_S_mask = np.zeros(Nm, dtype=bool)
    for idx in shortlist:
        est = float(rect_sizes[int(idx)])
        # push (neg_est, idx, version)
        heapq.heappush(heap, (-est, int(idx), hv_version))

    # If shortlist too small, ensure there are fallback candidates later (remaining_by_rect)
    remaining_candidates = np.setdiff1d(cand, shortlist, assume_unique=True)
    remaining_by_rect = remaining_candidates[np.argsort(-rect_sizes[remaining_candidates])]

    # Greedy select k elements
    while len(S) < k and heap:
        neg_est, idx, item_ver = heapq.heappop(heap)
        if in_S_mask[int(idx)]:
            continue
        # if item_version != current hv_version, recompute true marginal and reinsert with updated stamp
        if item_ver != hv_version:
            # compute true marginal gain
            key_union = tuple(list(S) + [int(idx)])
            hv_union = cached_hv(key_union)
            marginal = hv_union - hv_S
            # If marginal is negligible or negative, skip pushing maybe (but keep for potential later)
            if marginal <= 0:
                # we still might want other items, skip reinserting zero/negatives
                continue
            heapq.heappush(heap, (-float(marginal), int(idx), hv_version))
            continue
        # item is up-to-date, accept it
        key_union = tuple(list(S) + [int(idx)])
        hv_union = cached_hv(key_union)
        marginal = hv_union - hv_S
        if marginal <= 0:
            # nothing gained, skip
            continue
        S.append(int(idx))
        in_S_mask[int(idx)] = True
        hv_S = float(hv_union)
        hv_version += 1
        # optionally expand heap if we are running out and have remaining_by_rect
        if len(heap) < 3 and remaining_by_rect.size > 0:
            to_add = remaining_by_rect[:min( max(10, k), remaining_by_rect.size )]
            remaining_by_rect = remaining_by_rect[np.setdiff1d(np.arange(remaining_by_rect.size), np.arange(len(to_add)), assume_unique=False)]
            for r in to_add:
                if not in_S_mask[int(r)]:
                    est = float(rect_sizes[int(r)])
                    heapq.heappush(heap, (-est, int(r), hv_version))

    # if still not enough (heap exhausted), fill by top rect among remaining candidates
    if len(S) < k:
        remaining_all = np.setdiff1d(cand, np.array(S, dtype=int), assume_unique=True)
        need = k - len(S)
        add_rel = remaining_all[np.argsort(-rect_sizes[remaining_all])][:need]
        for a in add_rel:
            if in_S_mask[int(a)]:
                continue
            S.append(int(a))
            in_S_mask[int(a)] = True
        # recompute hv_S for final S
        hv_S = cached_hv(tuple(S))

    S = list(dict.fromkeys(S))[:k]
    hv_S = cached_hv(tuple(S))

    # Bounded single-swap local refinement (try a limited number of beneficial swaps)
    max_swap_iters = 50
    swap_iter = 0
    improved = True
    top_out = min(Nm, max(10 * k, 200))
    outsider_order = np.argsort(-rect_sizes)  # global by rect
    while improved and swap_iter < max_swap_iters:
        swap_iter += 1
        improved = False
        cur_S = list(S)
        non_selected = np.setdiff1d(np.arange(Nm), np.array(cur_S, dtype=int), assume_unique=True)
        if non_selected.size == 0:
            break
        outsider_cands = outsider_order[np.isin(outsider_order, non_selected)][:top_out]
        # randomized traversal of selection to avoid deterministic cycles
        rng = np.random.default_rng(1000 + swap_iter)
        sel_order = cur_S.copy()
        rng.shuffle(sel_order)
        for si in sel_order:
            for out in outsider_cands:
                if out in cur_S:
                    continue
                trial = list(cur_S)
                try:
                    pos = trial.index(int(si))
                except ValueError:
                    continue
                trial[pos] = int(out)
                key = tuple(sorted(int(x) for x in trial))
                hv_trial = cached_hv(key)
                if hv_trial > hv_S + 1e-12:
                    S = list(key)[:k]
                    hv_S = float(hv_trial)
                    improved = True
                    break
            if improved:
                break

    selected_rel = np.array(S, dtype=int)[:k]
    selected_orig = idx_all[selected_rel]
    # safety fill if too few
    if selected_orig.size < k:
        remaining_orig = np.setdiff1d(np.arange(N), selected_orig, assume_unique=True)
        if remaining_orig.size > 0:
            rem_rects_full = rect_sizes_all[remaining_orig]
            add = remaining_orig[np.argsort(-rem_rects_full)][: (k - selected_orig.size)]
            selected_orig = np.concatenate([selected_orig, add])
    selected_orig = selected_orig[:k]
    return pts[selected_orig].copy()


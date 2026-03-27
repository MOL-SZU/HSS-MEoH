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

    # Remove points that have zero rect volume w.r.t. ref
    rect_sizes_all = np.prod(np.clip(ref - pts, a_min=0.0, a_max=None), axis=1)
    meaningful_mask = rect_sizes_all > 0
    if not np.any(meaningful_mask):
        idx = np.arange(N)[:k]
        return pts[idx].copy()
    idx_all = np.nonzero(meaningful_mask)[0]
    pts_m = pts[idx_all]
    rect_sizes = rect_sizes_all[idx_all]
    Nm = pts_m.shape[0]

    # vectorized nondominated filtering (minimization assumed)
    def nondominated_mask(X):
        n = X.shape[0]
        keep = np.ones(n, dtype=bool)
        # efficient pairwise check in blocks to avoid heavy Python loops if n is large
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
    if nd_idx_rel.size >= max(1, k):
        candidate_idx_rel = nd_idx_rel
    else:
        candidate_idx_rel = np.arange(Nm)

    # adaptive shortlist similar spirit but we will sample from this candidate set
    c = 4
    pool_size = min(candidate_idx_rel.size, max(int(c * k), 100))
    sorted_by_rect = candidate_idx_rel[np.argsort(-rect_sizes[candidate_idx_rel])]
    pool_idx_rel = sorted_by_rect[:pool_size]
    pool_idx_rel = np.unique(pool_idx_rel)
    pool_count = pool_idx_rel.size

    # HV computation: try pygmo for exact HV, otherwise Monte Carlo fallback
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
        rng_local = np.random.default_rng(12345)

        def hv_of_set_rel(rel_indices, M_base=3000):
            rel_indices = np.asarray(rel_indices, dtype=int)
            if rel_indices.size == 0:
                return 0.0
            data = pts_m[rel_indices]
            M = min(6000, max(800, M_base))
            lower = data.min(axis=0)
            upper = ref.copy()
            zero_range = (upper - lower) <= 0
            if np.any(zero_range):
                upper = upper.copy()
                upper[zero_range] = lower[zero_range] + 1e-9
            samples = rng_local.random((M, D)) * (upper - lower)[None, :] + lower[None, :]
            dominated = np.zeros(M, dtype=bool)
            for p in data:
                dominated |= np.all(samples >= p, axis=1)
                if dominated.all():
                    break
            box_vol = np.prod(upper - lower)
            return float(box_vol * dominated.mean())

    # cache hv of frozenset of relative indices
    hv_cache = {frozenset(): 0.0}
    def cached_hv_from_set(rel_set):
        key = frozenset(int(x) for x in rel_set)
        if key in hv_cache:
            return hv_cache[key]
        val = hv_of_set_rel(np.array(list(key), dtype=int))
        hv_cache[key] = float(val)
        return float(val)

    # single-point rect surrogate
    single_rect = np.prod(np.clip(ref - pts_m, a_min=0.0, a_max=None), axis=1)

    # Stochastic greedy selection:
    rng = np.random.default_rng(42)
    S_set = set()
    S_list = []
    hv_S = 0.0

    # Precompute candidate pool indices (operate on pool_idx_rel)
    candidates = np.array(pool_idx_rel, dtype=int)

    # Main loop: sample a small subset each iteration, evaluate only top-tier by surrogate
    for it in range(k):
        remaining = np.setdiff1d(candidates, np.array(list(S_set), dtype=int), assume_unique=True)
        if remaining.size == 0:
            # fallback to entire non-selected universe
            remaining = np.setdiff1d(np.arange(Nm), np.array(list(S_set), dtype=int), assume_unique=True)
            if remaining.size == 0:
                break

        # sample size: balance exploration and cost
        # base sample proportional to pool size but at least 20 and at most remaining
        s = min(remaining.size, max(20, int(max(1, candidates.size // max(1, k)))))
        sample_idx = rng.choice(remaining, size=s, replace=False)

        # further shortlist among the sample using single_rect surrogate
        t = min(12, sample_idx.size)  # evaluate exact HV for top t by surrogate
        # pick top-t by single_rect within sample
        sample_rects = single_rect[sample_idx]
        if t < sample_idx.size:
            top_t_mask = np.argsort(-sample_rects)[:t]
            top_candidates = sample_idx[top_t_mask]
        else:
            top_candidates = sample_idx

        # compute exact marginal hv for top_candidates using cache
        best_c = None
        best_gain = -np.inf
        # Avoid rebuilding S_set list repeatedly
        S_list_local = list(S_set)
        for cidx in top_candidates:
            union_key = frozenset(S_list_local + [int(cidx)])
            if union_key in hv_cache:
                hv_union = hv_cache[union_key]
            else:
                hv_union = hv_of_set_rel(np.array(list(union_key), dtype=int))
                hv_cache[union_key] = float(hv_union)
            gain = hv_union - hv_S
            if gain > best_gain:
                best_gain = gain
                best_c = int(cidx)

        # If all top_candidates had very small gain and there remain other sampled items,
        # check a small secondary batch to avoid missing good non-rect-high points
        if (best_gain <= 0 and sample_idx.size > top_candidates.size):
            remaining_sample = np.setdiff1d(sample_idx, top_candidates, assume_unique=True)
            sec_t = min(6, remaining_sample.size)
            sec_by_rect = remaining_sample[np.argsort(-single_rect[remaining_sample])][:sec_t]
            for cidx in sec_by_rect:
                union_key = frozenset(S_list_local + [int(cidx)])
                if union_key in hv_cache:
                    hv_union = hv_cache[union_key]
                else:
                    hv_union = hv_of_set_rel(np.array(list(union_key), dtype=int))
                    hv_cache[union_key] = float(hv_union)
                gain = hv_union - hv_S
                if gain > best_gain:
                    best_gain = gain
                    best_c = int(cidx)

        # If still no positive gain and remaining is non-empty, pick highest rect among remaining
        if best_c is None:
            # pick by rect surrogate
            cand = remaining[np.argmax(single_rect[remaining])]
            best_c = int(cand)
            union_key = frozenset(S_list_local + [best_c])
            if union_key in hv_cache:
                hv_union = hv_cache[union_key]
            else:
                hv_union = hv_of_set_rel(np.array(list(union_key), dtype=int))
                hv_cache[union_key] = float(hv_union)
            best_gain = hv_union - hv_S

        # Accept best_c
        S_set.add(int(best_c))
        S_list.append(int(best_c))
        hv_S = cached_hv_from_set(S_set)

    # If we selected fewer than k (rare), fill by highest rect from full meaningful set
    if len(S_list) < k:
        remaining_all = np.setdiff1d(np.arange(Nm), np.array(S_list, dtype=int), assume_unique=True)
        need = k - len(S_list)
        if remaining_all.size > 0:
            add_rel = remaining_all[np.argsort(-rect_sizes[remaining_all])][:need]
            for r in add_rel:
                if int(r) in S_set:
                    continue
                S_set.add(int(r))
                S_list.append(int(r))
        hv_S = cached_hv_from_set(S_set)

    # Local bounded swap refinement (single-swap hillclimb)
    max_swaps = 80
    swaps = 0
    improved = True
    while improved and swaps < max_swaps:
        improved = False
        swaps += 1
        cur_S = list(S_list)
        non_selected_rel = np.setdiff1d(np.arange(Nm), np.array(cur_S, dtype=int), assume_unique=True)
        if non_selected_rel.size == 0:
            break
        # prioritize outsiders by rect
        outsider_cands = non_selected_rel[np.argsort(-rect_sizes[non_selected_rel])][:min(250, non_selected_rel.size)]
        for si in cur_S:
            for out in outsider_cands:
                if int(out) in S_set:
                    continue
                trial_set = set(cur_S)
                trial_set.remove(int(si))
                trial_set.add(int(out))
                trial_key = frozenset(trial_set)
                if trial_key in hv_cache:
                    hv_trial = hv_cache[trial_key]
                else:
                    hv_trial = hv_of_set_rel(np.array(list(trial_key), dtype=int))
                    hv_cache[trial_key] = float(hv_trial)
                if hv_trial > hv_S + 1e-12:
                    S_list = list(trial_set)
                    S_set = set(S_list)
                    hv_S = float(hv_trial)
                    improved = True
                    break
            if improved:
                break

    selected_rel = np.array(S_list, dtype=int)[:k]
    # Map back to original indices
    selected_orig = idx_all[selected_rel]
    if selected_orig.size < k:
        remaining_orig = np.setdiff1d(np.arange(N), selected_orig, assume_unique=True)
        if remaining_orig.size > 0:
            rem_rects_full = rect_sizes_all[remaining_orig]
            add = remaining_orig[np.argsort(-rem_rects_full)][: (k - selected_orig.size)]
            selected_orig = np.concatenate([selected_orig, add])
    selected_orig = selected_orig[:k]
    return pts[selected_orig].copy()


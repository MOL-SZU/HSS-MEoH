def HSS(points, k: int, reference_point, c_shortlist=4.657166449523883, min_pool=190.46801481868147, M_base=3565.0134751480023, max_swaps=171.3593778660401, outsider_cands_cap=836.9724916211603) -> __import__('numpy').ndarray:
    import numpy as np
    import heapq
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    N, D = pts.shape
    if k <= 0:
        return np.zeros((0, D), dtype=pts.dtype)
    if k >= N:
        return pts.copy()
    if reference_point is None:
        ref = pts.max(axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float)
        if ref.ndim == 0:
            ref = np.full((D,), float(ref))
        if ref.shape != (D,):
            raise ValueError('reference_point must have shape (D,)')
        max_p = pts.max(axis=0)
        ref = np.maximum(ref, max_p + 1e-12)
    rect_sizes_all = np.prod(np.clip(ref - pts, a_min=0.0, a_max=None), axis=1)
    meaningful_mask = rect_sizes_all > 0
    if not np.any(meaningful_mask):
        idx = np.arange(N)[:k]
        return pts[idx].copy()
    idx_all = np.nonzero(meaningful_mask)[0]
    pts_m = pts[idx_all]
    rect_sizes = rect_sizes_all[idx_all]
    Nm = pts_m.shape[0]

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
    if nd_idx_rel.size >= max(1, k):
        candidate_idx_rel = nd_idx_rel
    else:
        candidate_idx_rel = np.arange(Nm)
    pool_size = min(candidate_idx_rel.size, max(int(c_shortlist * k), min_pool))
    sorted_by_rect = candidate_idx_rel[np.argsort(-rect_sizes[candidate_idx_rel])]
    pool_idx_rel = sorted_by_rect[:pool_size]
    pool_idx_rel = np.unique(pool_idx_rel)
    pool_count = pool_idx_rel.size
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

        def hv_of_set_rel(rel_indices):
            rel_indices = np.asarray(rel_indices, dtype=int)
            if rel_indices.size == 0:
                return 0.0
            data = pts_m[rel_indices]
            if D <= 3:
                M = min(5000, max(1500, M_base * 2))
            elif D <= 6:
                M = min(4000, max(1200, M_base))
            else:
                M = min(2500, max(800, M_base // 2))
            lower = data.min(axis=0)
            upper = ref.copy()
            zero_range = upper - lower <= 0
            if np.any(zero_range):
                upper = upper.copy()
                upper[zero_range] = lower[zero_range] + 1e-09
            samples = rng_local.random((M, D)) * (upper - lower)[None, :] + lower[None, :]
            dominated = np.zeros(M, dtype=bool)
            for p in data:
                dominated |= np.all(samples >= p, axis=1)
                if dominated.all():
                    break
            box_vol = np.prod(upper - lower)
            return float(box_vol * dominated.mean())
    hv_cache = {tuple(): 0.0}

    def cached_hv_from_set_tuple(rel_tuple):
        if rel_tuple in hv_cache:
            return hv_cache[rel_tuple]
        val = hv_of_set_rel(np.array(rel_tuple, dtype=int))
        hv_cache[rel_tuple] = float(val)
        return float(val)
    single_rect = np.prod(np.clip(ref - pts_m, a_min=0.0, a_max=None), axis=1)
    max_rect = single_rect.max() if single_rect.size > 0 else 1.0
    norm_rect = single_rect / max_rect
    rng = np.random.default_rng(2021)
    heap = []
    stamp = 0
    candidates = np.array(pool_idx_rel, dtype=int)
    for idx in candidates:
        score = float(norm_rect[int(idx)] + 1e-06 * rng.random())
        heap.append((-score, 0, int(idx)))
    heapq.heapify(heap)
    S_set = set()
    S_list = []
    hv_S = 0.0
    hv_cache[tuple()] = 0.0
    while len(S_list) < k and heap:
        neg_est, s_stamp, cand = heapq.heappop(heap)
        if s_stamp != stamp:
            S_plus = S_list + [int(cand)]
            key = tuple(sorted(S_plus))
            hv_union = cached_hv_from_set_tuple(key)
            gain = hv_union - hv_S
            heapq.heappush(heap, (-float(gain), stamp, int(cand)))
            continue
        else:
            key = tuple(sorted(S_list + [int(cand)]))
            if key in hv_cache:
                hv_union = hv_cache[key]
            else:
                hv_union = hv_of_set_rel(np.array(key, dtype=int))
                hv_cache[key] = float(hv_union)
            gain = hv_union - hv_S
            S_list.append(int(cand))
            S_set.add(int(cand))
            hv_S = float(hv_union)
            stamp += 1
            if not heap and len(S_list) < k:
                remaining = np.setdiff1d(np.arange(Nm), np.array(S_list, dtype=int), assume_unique=True)
                for idx in remaining:
                    if int(idx) in S_set:
                        continue
                    score = float(norm_rect[int(idx)] + 1e-06 * rng.random())
                    heapq.heappush(heap, (-score, stamp, int(idx)))
                if not heap:
                    break
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
        hv_S = cached_hv_from_set_tuple(tuple(sorted(S_list)))
    swaps = 0
    improved = True
    while improved and swaps < max_swaps:
        improved = False
        swaps += 1
        cur_S = list(S_list)
        non_selected_rel = np.setdiff1d(np.arange(Nm), np.array(cur_S, dtype=int), assume_unique=True)
        if non_selected_rel.size == 0:
            break
        outsider_cands = non_selected_rel[np.argsort(-rect_sizes[non_selected_rel])][:min(outsider_cands_cap, non_selected_rel.size)]
        for si in list(cur_S):
            for out in outsider_cands:
                if int(out) in S_set:
                    continue
                trial = set(cur_S)
                trial.remove(int(si))
                trial.add(int(out))
                trial_key = tuple(sorted(trial))
                if trial_key in hv_cache:
                    hv_trial = hv_cache[trial_key]
                else:
                    hv_trial = hv_of_set_rel(np.array(trial_key, dtype=int))
                    hv_cache[trial_key] = float(hv_trial)
                if hv_trial > hv_S + 1e-12:
                    S_list = list(trial_key)
                    S_set = set(S_list)
                    hv_S = float(hv_trial)
                    improved = True
                    break
            if improved:
                break
    selected_rel = np.array(S_list, dtype=int)[:k]
    selected_orig = idx_all[selected_rel]
    if selected_orig.size < k:
        remaining_orig = np.setdiff1d(np.arange(N), selected_orig, assume_unique=True)
        if remaining_orig.size > 0:
            rem_rects_full = rect_sizes_all[remaining_orig]
            add = remaining_orig[np.argsort(-rem_rects_full)][:k - selected_orig.size]
            selected_orig = np.concatenate([selected_orig, add])
    selected_orig = selected_orig[:k]
    return pts[selected_orig].copy()
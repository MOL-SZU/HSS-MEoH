import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for exact hypervolume computation") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k <= 0:
        return np.empty((0, D), dtype=pts.dtype)
    if N == 0:
        return np.empty((0, D), dtype=pts.dtype)
    if k >= N:
        return pts.copy()

    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # compute simple box volumes as a quick proxy (non-negative dims only)
    delta = np.clip(reference_point - pts, a_min=0.0, a_max=None)
    with np.errstate(invalid='ignore'):
        box_vol = np.prod(delta, axis=1)
    box_vol = np.clip(box_vol, a_min=0.0, a_max=None)

    # fast nondominated extraction (smaller coordinates are better for HV)
    def nondominated_indices(idx_list):
        arr = pts[np.array(idx_list, dtype=int), :]
        order = np.argsort(np.sum(arr, axis=1), kind='mergesort')
        ordered = [idx_list[i] for i in order]
        kept = []
        for idx in ordered:
            p = pts[idx]
            dominated = False
            for j in kept:
                q = pts[j]
                if np.all(q <= p) and np.any(q < p):
                    dominated = True
                    break
            if not dominated:
                to_remove = []
                for j in kept:
                    q = pts[j]
                    if np.all(p <= q) and np.any(p < q):
                        to_remove.append(j)
                for tr in to_remove:
                    kept.remove(tr)
                kept.append(idx)
        return kept

    # candidate pool parameters tuned differently: somewhat smaller pools for speed
    candidate_pool_size = int(min(N, max(6 * k, 300)))
    candidate_pool_size = max(candidate_pool_size, k)
    order_by_box = np.argsort(-box_vol, kind='mergesort')
    initial_candidates = order_by_box[:candidate_pool_size].tolist()

    nd = nondominated_indices(initial_candidates)
    if len(nd) < k:
        candidates = initial_candidates
    else:
        cap = int(min(len(nd), max(4 * k, 500)))
        candidates = nd[:cap]

    # diversity sampling using farthest-point sampling on objective-space (normalized)
    def farthest_point_sampling(indices, m, rng):
        if len(indices) <= m:
            return list(indices)
        P = pts[np.array(indices, dtype=int), :]
        mins = P.min(axis=0)
        maxs = P.max(axis=0)
        rngange = maxs - mins
        rngange[rngange == 0] = 1.0
        normP = (P - mins) / rngange
        chosen = []
        # start with a random high-box-vol point to increase diversity across runs
        idx0 = max(indices, key=lambda i: box_vol[i])
        if rng.random() < 0.5:
            # sometimes choose a random high-box-vol among top 5
            top5 = sorted(indices, key=lambda i: -box_vol[i])[:min(5, len(indices))]
            idx0 = rng.choice(top5)
        chosen.append(idx0)
        chosen_idx_in_list = [indices.index(idx0)]
        dists = np.linalg.norm(normP - normP[chosen_idx_in_list[0]], axis=1)
        for _ in range(1, m):
            far_i = int(np.argmax(dists))
            chosen.append(indices[far_i])
            chosen_idx_in_list.append(far_i)
            newd = np.linalg.norm(normP - normP[far_i], axis=1)
            dists = np.minimum(dists, newd)
        return chosen

    rng = np.random.RandomState(42)
    cand_cap = int(min(len(candidates), max(3 * k, 300)))
    diverse_candidates = farthest_point_sampling(candidates, cand_cap, rng)

    # prepare exact hypervolume computation with caching
    hv_cache = {}

    def hv_of_indices(global_idxs):
        if not global_idxs:
            return 0.0
        key = tuple(sorted(int(i) for i in global_idxs))
        if key in hv_cache:
            return hv_cache[key]
        data = pts[np.array(key, dtype=int), :]
        hv = pg.hypervolume(data)
        val = float(hv.compute(reference_point))
        hv_cache[key] = val
        return val

    # precompute singletons for candidates (exact)
    hv_single = {}
    for idx in diverse_candidates:
        hv_single[idx] = hv_of_indices([idx])

    # build combined score: normalized box_vol and normalized hv_single
    alpha = 0.35  # different weighting favoring exact hv a bit more
    bvals = np.array([box_vol[i] for i in diverse_candidates], dtype=float)
    hvals = np.array([hv_single[i] for i in diverse_candidates], dtype=float)
    # avoid degenerate normalization
    if bvals.max() > bvals.min():
        bnorm = (bvals - bvals.min()) / (bvals.max() - bvals.min())
    else:
        bnorm = np.ones_like(bvals)
    if hvals.max() > hvals.min():
        hnorm = (hvals - hvals.min()) / (hvals.max() - hvals.min())
    else:
        hnorm = np.ones_like(hvals)
    combo_score = {}
    for ii, idx in enumerate(diverse_candidates):
        combo_score[idx] = float(alpha * bnorm[ii] + (1.0 - alpha) * hnorm[ii])

    # quick path if candidate size equals k
    if len(diverse_candidates) == k:
        return pts[np.array(diverse_candidates, dtype=int), :].copy()

    # main multi-start lazy greedy with randomized restarts (more restarts but smaller per-run effort)
    best_sel = None
    best_hv = -np.inf
    num_restarts = int(min(12, max(2, k // 2)))
    base_candidates = list(diverse_candidates)

    for restart in range(num_restarts):
        selected = []
        selected_set = set()
        # randomized seeding: mixture of top combo scores and random pick
        if restart == 0:
            # deterministic start: choose best combo seed
            seed0 = max(base_candidates, key=lambda x: combo_score.get(x, 0.0))
            selected.append(int(seed0))
            selected_set.add(int(seed0))
        else:
            # seed with 0-2 items depending on k
            seeds = []
            if rng.random() < 0.6:
                # take top by combo and one far
                top1 = max(base_candidates, key=lambda x: combo_score.get(x, 0.0))
                seeds.append(top1)
                if k > 1 and len(base_candidates) > 1:
                    rem = [c for c in base_candidates if c != top1]
                    seed2 = max(rem, key=lambda i: np.linalg.norm(pts[i] - pts[top1]))
                    seeds.append(seed2)
            else:
                # random seeds
                seeds = list(rng.choice(base_candidates, size=min(2, len(base_candidates)), replace=False))
            for s in seeds[:k]:
                if s not in selected_set:
                    selected.append(int(s))
                    selected_set.add(int(s))

        hv_current = hv_of_indices(selected)

        # lazy heap: use combo_score as initial estimate of marginal gain (scaled)
        heap = []
        version = 0
        est = {}
        # normalize combo scores to positive scale
        cs_vals = np.array([combo_score[i] for i in base_candidates], dtype=float)
        if cs_vals.max() > cs_vals.min():
            cs_norm_map = {i: (combo_score[i] - cs_vals.min()) / (cs_vals.max() - cs_vals.min()) for i in base_candidates}
        else:
            cs_norm_map = {i: 1.0 for i in base_candidates}
        for idx in base_candidates:
            if idx in selected_set:
                continue
            g = cs_norm_map.get(idx, 0.0)
            est[idx] = g
            heap.append((-g, 0, int(idx)))
        heapq.heapify(heap)

        # greedy selection with lazy exact re-evaluation
        while len(selected) < k and heap:
            neg_g, ver, idx = heapq.heappop(heap)
            if idx in selected_set:
                continue
            # if estimate may be stale, recompute true marginal hv
            if ver < version:
                with_idx = hv_of_indices(selected + [idx])
                gain = with_idx - hv_current
                if gain < 0:
                    gain = 0.0
                est[idx] = gain
                heapq.heappush(heap, (-gain, version, int(idx)))
                continue
            # accept idx
            selected.append(int(idx))
            selected_set.add(int(idx))
            hv_current = hv_of_indices(selected)
            version += 1

        # if not enough selected, fill by remaining top combo score then box_vol
        if len(selected) < k:
            rem_candidates = [i for i in base_candidates if i not in selected_set]
            rem_sorted = sorted(rem_candidates, key=lambda x: (-combo_score.get(x, 0.0), -box_vol[x]))
            for i in rem_sorted:
                selected.append(int(i))
                selected_set.add(int(i))
                if len(selected) >= k:
                    break
            hv_current = hv_of_indices(selected)

        # bounded 1-swap local improvement (try best candidates first)
        max_swap_iters = 50
        swap_iter = 0
        improved = True
        tol = 1e-12
        candidate_explore = sorted([i for i in base_candidates if i not in selected_set],
                                   key=lambda x: -combo_score.get(x, 0.0))[:min(400, len(base_candidates))]
        while improved and swap_iter < max_swap_iters:
            improved = False
            swap_iter += 1
            for sel_pos, sel_idx in enumerate(list(selected)):
                if not candidate_explore:
                    break
                base_without = [x for x in selected if x != sel_idx]
                best_local_gain = 0.0
                best_u = None
                # try limited top candidates to bound time
                for u in candidate_explore:
                    if u in selected_set:
                        continue
                    swapped_hv = hv_of_indices(base_without + [u])
                    gain = swapped_hv - hv_current
                    if gain > best_local_gain + tol:
                        best_local_gain = gain
                        best_u = u
                if best_u is not None:
                    # perform swap
                    selected[sel_pos] = int(best_u)
                    selected_set.remove(sel_idx)
                    selected_set.add(int(best_u))
                    hv_current += best_local_gain
                    improved = True
                    candidate_explore = sorted([i for i in base_candidates if i not in selected_set],
                                               key=lambda x: -combo_score.get(x, 0.0))[:min(400, len(base_candidates))]
                    break

        if hv_current > best_hv + 1e-15:
            best_hv = hv_current
            best_sel = list(selected)

    # fallback if nothing found
    if best_sel is None:
        best_sel = order_by_box[:k].tolist()
    if len(best_sel) < k:
        remaining = [i for i in order_by_box if i not in best_sel]
        need = k - len(best_sel)
        best_sel.extend(remaining[:need])
    best_sel = best_sel[:k]

    subset = pts[np.array(best_sel, dtype=int), :].copy()
    return subset


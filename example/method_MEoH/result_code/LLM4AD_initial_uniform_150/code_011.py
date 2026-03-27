import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    HSS (Hypervolume-based Subset Selection): 贪心选择超体积贡献最大的 k 个点

    参数:
        points: np.ndarray
            所有候选点，形状为 (N, D)，其中 N 是点数，D 是目标维度
        k: int
            需要选择的点数
        reference_point: np.ndarray, optional
            参考点，形状为 (D,)。如果为 None，则使用 points 的最大值 * 1.1

    返回:
        subset: np.ndarray
            选出的子集，形状为 (k, D)
    """
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

    # candidate pool parameters (different from original): medium pool, bias to more candidates for diversity
    candidate_pool_size = int(min(N, max(8 * k, 500)))
    candidate_pool_size = max(candidate_pool_size, k)
    order_by_box = np.argsort(-box_vol, kind='mergesort')
    initial_candidates = order_by_box[:candidate_pool_size].tolist()

    nd = nondominated_indices(initial_candidates)
    if len(nd) < k:
        candidates = initial_candidates
    else:
        cap = int(min(len(nd), max(6 * k, 800)))
        candidates = nd[:cap]

    # diversity / downsampling using farthest-point sampling on objective-space (normalized)
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
        # deterministic start: highest box volume
        idx0 = max(indices, key=lambda i: box_vol[i])
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

    rng = np.random.RandomState(123)
    cand_cap = int(min(len(candidates), max(4 * k, 500)))
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

    # build combined score: use geometric mean of normalized box_vol and normalized hv_single,
    # and add a density-based diversity bonus (lower density -> higher bonus)
    beta = 0.65  # weight towards box volume in geometric combination (exponent)
    lambda_div = 0.20  # weight of diversity term
    # prepare arrays
    bvals = np.array([box_vol[i] for i in diverse_candidates], dtype=float)
    hvals = np.array([hv_single[i] for i in diverse_candidates], dtype=float)
    # normalize to [0,1]
    if bvals.max() > bvals.min():
        bnorm = (bvals - bvals.min()) / (bvals.max() - bvals.min())
    else:
        bnorm = np.full_like(bvals, 1.0)
    if hvals.max() > hvals.min():
        hnorm = (hvals - hvals.min()) / (hvals.max() - hvals.min())
    else:
        hnorm = np.full_like(hvals, 1.0)
    # avoid zeros for geometric mean by small epsilon
    eps = 1e-12
    bnorm = np.clip(bnorm, eps, 1.0)
    hnorm = np.clip(hnorm, eps, 1.0)
    # compute geometric-like score via weighted power means
    geom_scores = (bnorm ** beta) * (hnorm ** (1.0 - beta))

    # density estimation (local crowding): average normalized distance to nearest neighbors among candidates
    P = pts[np.array(diverse_candidates, dtype=int), :]
    # use Euclidean distances; compute pairwise distances in a memory-conscious way if necessary
    # small candidate sets typically, so compute full distance matrix
    if P.shape[0] > 1:
        dif = P[:, None, :] - P[None, :, :]
        dmat = np.linalg.norm(dif, axis=2)
        # set self-distance to large
        np.fill_diagonal(dmat, dmat.max() if dmat.size > 1 else 1.0)
        # use average distance to up to m_nn nearest neighbors
        m_nn = min(6, P.shape[0] - 1)
        nn_mean = np.mean(np.sort(dmat, axis=1)[:, :m_nn], axis=1)
        # normalize density: larger nn_mean => less crowded => higher diversity score
        if nn_mean.max() > nn_mean.min():
            diversity_score = (nn_mean - nn_mean.min()) / (nn_mean.max() - nn_mean.min())
        else:
            diversity_score = np.ones_like(nn_mean)
    else:
        diversity_score = np.ones(len(diverse_candidates))

    # final combo: mostly geom_scores plus diversity bonus
    final_score = (1.0 - lambda_div) * (geom_scores - geom_scores.min()) / max(1e-12, (geom_scores.max() - geom_scores.min())) \
                  + lambda_div * diversity_score
    combo_score = {int(diverse_candidates[i]): float(final_score[i]) for i in range(len(diverse_candidates))}

    # quick path
    if len(diverse_candidates) == k:
        return pts[np.array(diverse_candidates, dtype=int), :].copy()

    # main multi-start lazy greedy with randomized restarts (different parameters)
    best_sel = None
    best_hv = -np.inf
    num_restarts = int(min(10, max(2, k)))  # a bit more restarts for smaller k
    base_candidates = list(diverse_candidates)

    for restart in range(num_restarts):
        selected = []
        selected_set = set()
        # seeding strategy: deterministic best-first on first restart, otherwise mix
        if restart == 0:
            seed0 = max(base_candidates, key=lambda x: combo_score.get(x, 0.0))
            selected.append(int(seed0))
            selected_set.add(int(seed0))
        else:
            # choose 1-3 seeds depending on k
            nseed = 1 if k <= 4 else (2 if k <= 12 else 3)
            # sample top by score but introduce randomness
            top_pool = sorted(base_candidates, key=lambda x: -combo_score.get(x, 0.0))[:min(10, len(base_candidates))]
            seeds = list(rng.choice(top_pool, size=min(nseed, len(top_pool)), replace=False))
            for s in seeds:
                if s not in selected_set and len(selected) < k:
                    selected.append(int(s))
                    selected_set.add(int(s))

        hv_current = hv_of_indices(selected)

        # lazy heap initialization: use combo_score as optimistic estimate
        heap = []
        version = 0
        est = {}
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

        # greedy selection with lazy exact re-evaluation (bounded by time via limited exact recomputations)
        max_recomputations = max(200, 6 * k)
        recompute_count = 0
        while len(selected) < k and heap:
            neg_g, ver, idx = heapq.heappop(heap)
            if idx in selected_set:
                continue
            # recompute if stale or if we haven't validated enough
            if ver < version or recompute_count < max_recomputations:
                with_idx = hv_of_indices(selected + [idx])
                gain = with_idx - hv_current
                if gain < 0:
                    gain = 0.0
                est[idx] = gain
                recompute_count += 1
                heapq.heappush(heap, (-gain, version, int(idx)))
                continue
            # accept idx if still top
            selected.append(int(idx))
            selected_set.add(int(idx))
            hv_current = hv_of_indices(selected)
            version += 1

        # fill if not enough selected
        if len(selected) < k:
            rem_candidates = [i for i in base_candidates if i not in selected_set]
            rem_sorted = sorted(rem_candidates, key=lambda x: (-combo_score.get(x, 0.0), -box_vol[x]))
            for i in rem_sorted:
                selected.append(int(i))
                selected_set.add(int(i))
                if len(selected) >= k:
                    break
            hv_current = hv_of_indices(selected)

        # bounded 1-swap local improvement
        max_swap_iters = 60
        swap_iter = 0
        improved = True
        tol = 1e-12
        candidate_explore = sorted([i for i in base_candidates if i not in selected_set],
                                   key=lambda x: -combo_score.get(x, 0.0))[:min(200, len(base_candidates))]
        while improved and swap_iter < max_swap_iters:
            improved = False
            swap_iter += 1
            for sel_pos, sel_idx in enumerate(list(selected)):
                if not candidate_explore:
                    break
                base_without = [x for x in selected if x != sel_idx]
                best_local_gain = 0.0
                best_u = None
                # try limited top candidates
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
                                               key=lambda x: -combo_score.get(x, 0.0))[:min(200, len(base_candidates))]
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


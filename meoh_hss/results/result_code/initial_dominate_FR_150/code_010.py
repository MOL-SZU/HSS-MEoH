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
        raise ImportError("pygmo is required for hypervolume computations. Install pygmo.") from e

    # Try to use a fast KDTree for NN queries; fallback to brute force if not available
    try:
        from scipy.spatial import cKDTree as KDTree
    except Exception:
        KDTree = None

    rng = np.random.default_rng(12345)
    eps = 1e-12
    tol = 1e-12

    # Normalize input
    if not isinstance(points, np.ndarray):
        pts = np.array(points, dtype=float)
    else:
        pts = points.astype(float)

    if k <= 0:
        return np.zeros((0, pts.shape[1] if pts.ndim > 1 else 0))

    if pts.size == 0:
        # Nothing to choose from: return tiled reference or empty rows
        if reference_point is None:
            return np.zeros((k, 0))
        ref = np.asarray(reference_point, dtype=float).reshape(1, -1)
        return np.repeat(ref, k, axis=0)

    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    N, D = pts.shape

    # reference point
    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)
    if reference_point.shape[0] != D:
        raise ValueError("reference_point must have dimension D = points.shape[1].")

    # Pareto nondominated filter (minimization)
    def pareto_nondominated_idx(X):
        n = X.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        dominated = np.zeros(n, dtype=bool)
        # simple O(n^2) but OK for shortlist building; deterministic ordering
        for i in range(n):
            if dominated[i]:
                continue
            # if any j != i dominates i
            le = np.all(X <= X[i], axis=1)
            lt = np.any(X < X[i], axis=1)
            dom = le & lt
            dom[i] = False
            if np.any(dom):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = pareto_nondominated_idx(pts)
    if nd_idx.size == 0:
        nd_idx = np.arange(N)
    nd_points = pts[nd_idx]
    M = nd_points.shape[0]

    # If few nondominated points, return or pad deterministically
    if M <= k:
        if M == 0:
            ref = reference_point.reshape(1, -1)
            return np.repeat(ref, k, axis=0)
        subset = nd_points.copy()
        if subset.shape[0] < k:
            last = subset[-1].reshape(1, -1)
            extra = np.repeat(last, k - subset.shape[0], axis=0)
            subset = np.vstack([subset, extra])
        return subset[:k].copy()

    # HV proxy: log product of distances to reference (larger -> better)
    diffs = np.maximum(reference_point - nd_points, eps)
    logprod = np.sum(np.log(diffs), axis=1)

    # Diversity proxy: mean distance to nearest neighbors using KDTree if available
    k_neigh = min(10, max(1, int(round(np.sqrt(M)))))
    div_scores = np.zeros(M, dtype=float)
    if KDTree is not None and M > 1:
        try:
            tree = KDTree(nd_points)
            # query k_neigh+1 because includes self at zero
            kq = min(M, k_neigh + 1)
            dists, inds = tree.query(nd_points, k=kq, n_jobs=1)
            # If kq==1, dists shape (M,) -> make 2D
            if dists.ndim == 1:
                dists = dists.reshape(-1, 1)
            # drop self (first column typically zero)
            if dists.shape[1] > 1:
                # use first k_neigh non-self distances
                use = min(k_neigh, dists.shape[1] - 1)
                div_scores = np.mean(dists[:, 1:1 + use], axis=1)
            else:
                div_scores.fill(0.0)
        except Exception:
            KDTree = None  # fallback
    if KDTree is None:
        # brute force (vectorized)
        diff_mat = nd_points[:, None, :] - nd_points[None, :, :]
        dmat = np.linalg.norm(diff_mat, axis=2)
        for i in range(M):
            drow = dmat[i].copy()
            drow[i] = np.inf
            k_act = min(k_neigh, M - 1)
            if k_act > 0:
                div_scores[i] = np.mean(np.partition(drow, k_act - 1)[:k_act])
            else:
                div_scores[i] = 0.0

    def _norm(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin <= eps:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    lp_n = _norm(logprod)
    dv_n = _norm(div_scores)

    # Surrogate: stronger emphasis on diversity but keeps interaction term
    surrogate = 0.35 * lp_n + 0.55 * dv_n + 0.10 * (lp_n * dv_n)

    # Shortlist: O(k) with modest floor, deterministic tie-break by surrogate then by lp
    shortlist_size = min(M, max(int(4 * k + 12), 20))
    order_s = np.lexsort((-lp_n, -surrogate))  # primary surrogate, tie-breaker by logprod
    shortlist_local_idx = order_s[:shortlist_size]
    shortlist_global = nd_idx[shortlist_local_idx]  # indices into original pts
    shortlist_set = set(int(i) for i in shortlist_global)

    # Ensure at least k candidates in shortlist
    if len(shortlist_global) < k:
        needed = k - len(shortlist_global)
        remaining = [int(i) for i in nd_idx if int(i) not in shortlist_set]
        add = remaining[:needed]
        if add:
            shortlist_global = np.concatenate([shortlist_global, np.array(add, dtype=int)])
            shortlist_set.update(add)

    # Canonical HV cache keyed by sorted tuple of ints
    hv_cache = {}

    def hv_of_indices(idxs):
        # idxs: iterable of ints or set; canonical key is sorted tuple
        if isinstance(idxs, (set, list, tuple, np.ndarray)):
            key = tuple(sorted(int(i) for i in idxs))
        else:
            key = (int(idxs),)
        if key in hv_cache:
            return hv_cache[key]
        if len(key) == 0:
            hv_cache[key] = 0.0
            return 0.0
        try:
            val = float(pg.hypervolume(pts[list(key)]).compute(reference_point.tolist()))
        except Exception:
            try:
                val = float(pg.hypervolume(pts[list(key)]).compute(reference_point))
            except Exception:
                val = 0.0
        hv_cache[key] = val
        return val

    # Precompute singleton HVs for shortlist
    singleton_hv = {}
    for g in shortlist_global:
        gi = int(g)
        singleton_hv[gi] = hv_of_indices((gi,))

    # CELF lazy greedy seeded by singletons
    selected_set = set()
    hv_selected = 0.0

    marginal_map = {}
    heap = []
    counter = 0
    for g in shortlist_global:
        gi = int(g)
        last_sz = 0
        marg = singleton_hv.get(gi, hv_of_indices((gi,)))
        marginal_map[gi] = (last_sz, marg)
        heapq.heappush(heap, (-marg, last_sz, counter, gi))
        counter += 1

    # CELF main loop with lazy updates and early stopping
    while len(selected_set) < k and heap:
        neg_marg, last_sz, _cnt, g = heapq.heappop(heap)
        if g in selected_set:
            continue
        est = -neg_marg
        cur_size = len(selected_set)
        if last_sz == cur_size:
            if est <= tol:
                break
            selected_set.add(int(g))
            hv_selected = hv_of_indices(selected_set)
            continue
        # recompute true marginal
        new_set = set(selected_set)
        new_set.add(int(g))
        true_hv = hv_of_indices(new_set)
        true_marg = true_hv - hv_selected
        marginal_map[int(g)] = (cur_size, true_marg)
        heapq.heappush(heap, (-true_marg, cur_size, counter, int(g)))
        counter += 1

    # If not enough selected, fill deterministically using surrogate among remaining nondominated
    if len(selected_set) < k:
        remaining_global = [int(i) for i in nd_idx if int(i) not in selected_set]
        if remaining_global:
            rem_pts = pts[remaining_global]
            rem_diffs = np.maximum(reference_point - rem_pts, eps)
            rem_logprod = np.sum(np.log(rem_diffs), axis=1)
            rem_div = np.zeros(len(remaining_global), dtype=float)
            # cheap approximate diversity using distances to shortlist points
            if len(shortlist_global) > 0:
                s_pts = pts[list(shortlist_global)]
                diff = rem_pts[:, None, :] - s_pts[None, :, :]
                d = np.linalg.norm(diff, axis=2)
                rem_div = np.mean(d, axis=1)
            rem_sur = _norm(rem_logprod) * 0.4 + _norm(rem_div) * 0.6
            order_rem = np.argsort(-rem_sur)
            for idx in order_rem:
                if len(selected_set) >= k:
                    break
                selected_set.add(int(remaining_global[idx]))
                hv_selected = hv_of_indices(selected_set)
        while len(selected_set) < k:
            if len(shortlist_global) > 0:
                selected_set.add(int(shortlist_global[0]))
            else:
                selected_set.add(0)

    # Bounded first-improvement swap refinement
    max_swap_iters = min(80, 12 * k)
    swap_iter = 0
    improved = True
    # Build swap pool: shortlist + small set of extras from nondominated
    extra_candidates = [int(i) for i in nd_idx if int(i) not in shortlist_set]
    extra_limit = min(len(extra_candidates), 30)
    swap_pool = sorted(list(shortlist_set.union(extra_candidates[:extra_limit])))
    current_hv = hv_of_indices(selected_set)

    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        # deterministic order through selected_set
        for a in sorted(selected_set):
            for b in swap_pool:
                if b in selected_set:
                    continue
                new_set = set(selected_set)
                new_set.remove(int(a))
                new_set.add(int(b))
                new_hv = hv_of_indices(new_set)
                if new_hv > current_hv + tol:
                    selected_set = new_set
                    current_hv = new_hv
                    hv_selected = new_hv
                    improved = True
                    break
            if improved:
                break

    final_selected = sorted(list(selected_set))

    # Trim or pad deterministically to size k
    if len(final_selected) > k:
        final_selected = final_selected[:k]
    elif len(final_selected) < k:
        remaining = [i for i in range(N) if int(i) not in final_selected]
        if remaining:
            rem_pts = pts[remaining]
            rem_diffs = np.maximum(reference_point - rem_pts, eps)
            rem_logprod = np.sum(np.log(rem_diffs), axis=1)
            order_rem = np.argsort(-rem_logprod)
            for idx in order_rem:
                if len(final_selected) >= k:
                    break
                final_selected.append(int(remaining[idx]))
        while len(final_selected) < k:
            if len(shortlist_global) > 0:
                final_selected.append(int(shortlist_global[0]))
            else:
                final_selected.append(0)

    subset = pts[np.array(final_selected[:k], dtype=int)]
    return subset.copy()


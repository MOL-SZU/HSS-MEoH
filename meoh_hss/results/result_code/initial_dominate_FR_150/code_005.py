import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations. Install pygmo.") from e

    rng = np.random.default_rng(42)
    eps = 1e-12
    tol = 1e-12

    # Normalize input
    if not isinstance(points, np.ndarray):
        pts = np.array(points, dtype=float)
    else:
        pts = points.astype(float)
    if pts.size == 0 or k <= 0:
        # handle degenerate cases and padding similar to original
        if pts.size == 0:
            D = reference_point.shape[0] if reference_point is not None else 0
            if k <= 0:
                return np.zeros((0, 0))
            if D > 0:
                return np.tile(np.asarray(reference_point, dtype=float).reshape(1, -1), (k, 1))
            return np.zeros((k, 0))
        return np.zeros((0, pts.shape[1]))
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
    def pareto_nondominated(pts_array):
        n = pts_array.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        order = np.argsort(np.sum(pts_array, axis=1))
        dominated = np.zeros(n, dtype=bool)
        for ii in range(n):
            i = order[ii]
            if dominated[i]:
                continue
            le = np.all(pts_array <= pts_array[i], axis=1)
            lt = np.any(pts_array < pts_array[i], axis=1)
            comp = le & lt
            comp[i] = False
            if np.any(comp):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = pareto_nondominated(pts)
    if nd_idx.size == 0:
        nd_idx = np.arange(N)
    nd_points = pts[nd_idx]
    M = nd_points.shape[0]

    # If few nondominated points, return or pad deterministically
    if M <= k:
        if M == 0:
            pad = np.tile(reference_point.reshape(1, -1), (k, 1))
            return pad
        subset = nd_points.copy()
        if subset.shape[0] < k:
            last = subset[-1].reshape(1, -1)
            extra = np.repeat(last, k - subset.shape[0], axis=0)
            subset = np.vstack([subset, extra])
        return subset[:k].copy()

    # Quick HV-proxy: log-product of distances to reference (higher is better)
    diffs = np.maximum(reference_point - nd_points, eps)
    logprod = np.sum(np.log(diffs), axis=1)

    # Diversity proxy: mean distance to k nearest neighbors (exclude self)
    # Use a conservative k_neigh but emphasize diversity more strongly in surrogate
    k_neigh = min(12, max(1, int(round(np.sqrt(M)))))
    # Efficient-ish NN: compute distance matrix in chunks if necessary
    div_scores = np.zeros(M, dtype=float)
    # If M is small compute full dists, else loop per point (safe)
    if M <= 2000:
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
    else:
        for i in range(M):
            dists = np.linalg.norm(nd_points - nd_points[i], axis=1)
            dists[i] = np.inf
            k_act = min(k_neigh, M - 1)
            if k_act > 0:
                div_scores[i] = np.mean(np.partition(dists, k_act - 1)[:k_act])
            else:
                div_scores[i] = 0.0

    def _norm(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin <= eps:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    lp_n = _norm(logprod)
    dv_n = _norm(div_scores)

    # New surrogate: stronger bias to diversity and slightly different smoothing weights
    # Emphasize diversity by mixing additively and multiplicatively with a higher diversity weight
    # surrogate = 0.35 * lp_n * (0.2 + 0.8 * dv_n) + 0.65 * (0.25 + 0.75 * dv_n)
    # Introduce an interaction term to reward points that are both high LP and highly diverse
    surrogate = 0.4 * lp_n + 0.45 * dv_n + 0.15 * (lp_n * dv_n)

    # Build tighter shortlist adaptively (more conservative than original)
    shortlist_size = min(M, max(int(4 * k + 15 * (D ** 0.3)), 40))
    shortlist_local_idx = np.argsort(-surrogate)[:shortlist_size]
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

    # Canonical HV cache
    hv_cache = {}

    def hv_of_indices(idxs):
        # Accept iterable or set of ints
        if isinstance(idxs, (set, list, tuple, np.ndarray)):
            key = tuple(sorted(int(i) for i in idxs))
        else:
            key = tuple(sorted(int(idxs),))
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

    # Precompute singleton HVs for shortlist (seed CELF)
    singleton_hv = {}
    for g in shortlist_global:
        gi = int(g)
        singleton_hv[gi] = hv_of_indices((gi,))

    # CELF lazy greedy with canonical cache and early stopping
    selected_set = set()
    hv_selected = 0.0

    marginal_map = {}  # idx -> (last_size, marginal_estimate)
    heap = []
    counter = 0
    # seed heap with shortlist candidates
    for g in shortlist_global:
        gi = int(g)
        last_sz = 0
        marg = singleton_hv.get(gi, hv_of_indices((gi,)))
        marginal_map[gi] = (last_sz, marg)
        heapq.heappush(heap, (-marg, last_sz, counter, gi))
        counter += 1

    # CELF main loop: greedy pick with lazy updates
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
            rem_sur = _norm(rem_logprod)
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

    # Bounded first-improvement swap refinement with stricter caps
    max_swap_iters = min(80, 15 * k)
    swap_iter = 0
    improved = True
    # Build swap pool: shortlist + some extras from nondominated not in shortlist (smaller extra)
    extra_candidates = [int(i) for i in nd_idx if int(i) not in shortlist_set]
    extra_limit = min(len(extra_candidates), 30)
    swap_pool = sorted(list(shortlist_set.union(extra_candidates[:extra_limit])))
    current_hv = hv_of_indices(selected_set)

    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        # Deterministic order
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


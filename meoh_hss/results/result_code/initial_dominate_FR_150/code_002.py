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
        # return empty (k, D) or padded reference if appropriate
        if pts.size == 0:
            if reference_point is None:
                return np.zeros((0, 0))
            D = np.asarray(reference_point).reshape(-1).shape[0]
            if k <= 0:
                return np.zeros((0, 0))
            return np.tile(np.asarray(reference_point, dtype=float).reshape(1, -1), (k, 1)) if D > 0 else np.zeros((k, 0))
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

    # Pareto nondominated filtering (minimization assumed)
    def pareto_nondominated_idx(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        # sort by sum for pruning (deterministic)
        order = np.argsort(np.sum(X, axis=1))
        dominated = np.zeros(n, dtype=bool)
        for ii in range(n):
            i = order[ii]
            if dominated[i]:
                continue
            # j dominates i if pj <= pi for all and < for some
            le = np.all(X <= X[i], axis=1)
            lt = np.any(X < X[i], axis=1)
            comp = le & lt
            comp[i] = False
            if np.any(comp):
                dominated[i] = True
        return np.where(~dominated)[0]

    nd_idx = pareto_nondominated_idx(pts)
    if nd_idx.size == 0:
        # if all dominated (rare), treat all as candidates
        nd_idx = np.arange(N)
    nd_points = pts[nd_idx]
    M = nd_points.shape[0]

    # If few nondominated, return/pad deterministically
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

    # Stage 1: surrogate scoring (stable log-product prox for single-point HV) + vectorized k-NN diversity
    diffs = np.maximum(reference_point - nd_points, eps)  # (M, D)
    logprod = np.sum(np.log(diffs), axis=1)  # larger is better (proxy for HV)

    # Diversity via mean distance to t nearest neighbors (exclude self), vectorized per-point kNN via partition
    t = min(10, max(1, int(round(np.sqrt(M)))))
    # compute squared distances matrix in chunks if needed to avoid huge memory; use pairwise for simplicity
    # We'll compute distances row by row to avoid MxM memory explosion for very large M
    div_scores = np.empty(M, dtype=float)
    for i in range(M):
        dists = np.linalg.norm(nd_points - nd_points[i], axis=1)
        dists[i] = np.inf
        if M - 1 <= t:
            # use all others
            valid = dists != np.inf
            div_scores[i] = np.mean(dists[valid]) if np.any(valid) else 0.0
        else:
            ksmall = np.partition(dists, t)[:t]
            div_scores[i] = np.mean(ksmall)

    def _norm(x):
        xmin = np.min(x)
        xmax = np.max(x)
        if xmax - xmin <= eps:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    log_n = _norm(logprod)
    div_n = _norm(div_scores)

    alpha = 0.6  # slightly favor HV proxy
    surrogate = alpha * log_n + (1.0 - alpha) * div_n

    # Shortlist deterministically by surrogate (cap adaptively)
    shortlist_size = min(M, max(int(8 * k), 50))
    shortlist_local_idx = np.argsort(-surrogate)[:shortlist_size]
    shortlist_set_local = set(int(i) for i in shortlist_local_idx)

    # Ensure at least k in shortlist (should hold)
    if len(shortlist_local_idx) < k:
        need = k - len(shortlist_local_idx)
        remaining = [int(i) for i in range(M) if int(i) not in shortlist_set_local]
        add = remaining[:need]
        if add:
            shortlist_local_idx = np.concatenate([shortlist_local_idx, np.array(add, dtype=int)])
            shortlist_set_local.update(add)

    # Hypervolume cache keyed by sorted tuple of local indices
    hv_cache = {}

    def hv_of_local_indices(local_idxs):
        # local_idxs: iterable of ints (indices into nd_points)
        key = tuple(sorted(int(i) for i in local_idxs))
        if key in hv_cache:
            return hv_cache[key]
        if len(key) == 0:
            hv_cache[key] = 0.0
            return 0.0
        arr = nd_points[list(key)]
        try:
            val = float(pg.hypervolume(arr).compute(reference_point.tolist()))
        except Exception:
            try:
                val = float(pg.hypervolume(arr).compute(reference_point))
            except Exception:
                val = 0.0
        hv_cache[key] = val
        return val

    # Precompute singleton HVs for shortlist locals
    singletons = {}
    for li in shortlist_local_idx:
        singletons[int(li)] = hv_of_local_indices([int(li)])

    # CELF lazy greedy on shortlist (indices are local into nd_points)
    marginal_cache = {}  # li -> (marg, last_updated_selected_size)
    heap = []
    counter = 0
    for li in shortlist_local_idx:
        marg = float(singletons[int(li)])
        marginal_cache[int(li)] = (marg, 0)
        tie = float(surrogate[int(li)]) + 1e-15 * marg  # deterministic tie-break
        heapq.heappush(heap, (-marg, 0, counter, int(li), tie))
        counter += 1

    selected_locals = []
    selected_set_local = set()
    hv_selected = 0.0
    selected_size = 0

    # Greedy selection with CELF and early stopping when top marginal <= 0
    while selected_size < k and heap:
        neg_marg, last_upd, _cnt, li, tie = heapq.heappop(heap)
        cached = marginal_cache.get(int(li), (None, -1))
        cached_marg, cached_iter = cached
        # If cache corresponds to current selection size, accept
        if cached_iter == selected_size:
            # if marginal nonpositive, stop early
            if cached_marg <= tol:
                break
            selected_locals.append(int(li))
            selected_set_local.add(int(li))
            selected_size += 1
            hv_selected = hv_of_local_indices(selected_locals)
            continue
        else:
            # recompute true marginal
            if selected_size == 0:
                new_marg = float(singletons[int(li)])
            else:
                hv_with = hv_of_local_indices(selected_locals + [int(li)])
                new_marg = hv_with - hv_selected
            marginal_cache[int(li)] = (new_marg, selected_size)
            tie2 = float(surrogate[int(li)]) + 1e-15 * new_marg
            heapq.heappush(heap, (-new_marg, selected_size, counter, int(li), tie2))
            counter += 1
            continue

    # Fill remaining slots deterministically by surrogate from shortlist first, then from all nd_points
    if selected_size < k:
        # preferences: leftover shortlist sorted by surrogate
        remaining_short = [i for i in shortlist_local_idx if i not in selected_set_local]
        rem_order = sorted(remaining_short, key=lambda x: -surrogate[int(x)])
        for li in rem_order:
            if len(selected_locals) >= k:
                break
            selected_locals.append(int(li))
            selected_set_local.add(int(li))
            hv_selected = hv_of_local_indices(selected_locals)
        # if still less, take from overall nondominated by surrogate
        if len(selected_locals) < k:
            remaining_all = [i for i in range(M) if i not in selected_set_local]
            rem_all_sorted = sorted(remaining_all, key=lambda x: -surrogate[int(x)])
            for li in rem_all_sorted:
                if len(selected_locals) >= k:
                    break
                selected_locals.append(int(li))
                selected_set_local.add(int(li))
                hv_selected = hv_of_local_indices(selected_locals)

    # Ensure exactly k elements (pad deterministically if necessary)
    if len(selected_locals) < k:
        last_choice = int(shortlist_local_idx[0]) if len(shortlist_local_idx) > 0 else 0
        while len(selected_locals) < k:
            if last_choice not in selected_set_local:
                selected_locals.append(last_choice)
                selected_set_local.add(last_choice)
            else:
                for i in range(M):
                    if i not in selected_set_local:
                        selected_locals.append(i)
                        selected_set_local.add(i)
                        break
                else:
                    break
        hv_selected = hv_of_local_indices(selected_locals)

    # Swap refinement: bounded first-improvement over a modest pool (shortlist + some extras)
    max_swap_iters = min(200, 40 * k)
    swap_iter = 0
    improved = True
    extras = [i for i in range(M) if i not in shortlist_set_local][:min(40, M)]
    swap_pool = sorted(set(list(shortlist_set_local) + extras))
    current_hv = hv_of_local_indices(selected_locals)
    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        # deterministic order over selected elements
        for si_idx, a in enumerate(list(selected_locals)):
            # consider candidates in swap_pool in deterministic order
            for b in swap_pool:
                if b in selected_set_local:
                    continue
                swap_iter += 1
                trial = selected_locals.copy()
                trial[si_idx] = int(b)
                hv_trial = hv_of_local_indices(trial)
                if hv_trial > current_hv + tol:
                    # accept first improvement
                    selected_locals = trial
                    selected_set_local = set(selected_locals)
                    current_hv = hv_trial
                    improved = True
                    break
            if improved:
                break

    # Finalize selection to exactly k (preserve selection order)
    final_locals = selected_locals[:k]
    if len(final_locals) < k:
        rem = [i for i in range(M) if i not in final_locals]
        rem_sorted = sorted(rem, key=lambda x: -surrogate[x])
        for r in rem_sorted:
            if len(final_locals) >= k:
                break
            final_locals.append(r)
    final_locals = final_locals[:k]

    subset = nd_points[np.array(final_locals, dtype=int)]
    return subset.copy()


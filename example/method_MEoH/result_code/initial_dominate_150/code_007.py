import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations. "
                          "Please install pygmo (e.g. pip install pygmo) or provide it in the environment.") from e

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N, D = points.shape

    # trivial cases
    if N == 0 or k <= 0:
        return np.empty((0, D))

    # reference point
    if reference_point is None:
        ref = np.max(points, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).ravel()
        if ref.size != D:
            raise ValueError("reference_point must have the same dimensionality as points")

    # If k >= N: return all points; if k > N, pad with random duplicates to reach k
    if k >= N:
        if k == N:
            return points.copy()
        else:
            rng = np.random.default_rng(2025)
            extra = k - N
            idx_extra = rng.choice(N, size=extra, replace=True)
            return np.vstack([points.copy(), points[idx_extra]])

    # Fast per-point upper-bound hypervolume (individual box to reference)
    diffs = ref.reshape(1, -1) - points  # shape (N, D)
    nonneg = np.maximum(diffs, 0.0)
    indiv_hv = np.prod(nonneg, axis=1)

    # Heap entries: (-estimated_gain, idx, version)
    heap = []
    for i in range(N):
        # initial estimate is the individual hv (upper bound)
        heap.append((-float(indiv_hv[i]), int(i), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_points = np.empty((0, D), dtype=float)
    current_hv = 0.0
    version = 0  # equals len(selected) (logical timestamp for cached estimates)
    tol = 1e-12

    # helper to compute true marginal gain of adding candidate j to current_points
    def true_marginal_gain(j):
        if current_points.shape[0] == 0:
            hv_with = pg.hypervolume(points[j].reshape(1, -1)).compute(ref)
        else:
            arr = np.vstack([current_points, points[j].reshape(1, -1)])
            hv_with = pg.hypervolume(arr).compute(ref)
        tg = hv_with - current_hv
        if tg < 0 and tg > -tol:
            tg = 0.0
        return float(tg), float(hv_with)

    # Periodic refresh parameters (different than original batch-verify)
    REFRESH_PERIOD = 5  # after this many successful accepts, refresh top of heap
    REFRESH_TOP = max(2, min(20, N // 10 + 1))  # how many top candidates to recompute during refresh
    accepts_since_refresh = 0

    MAX_ITERS = max(2000, k * 200)
    iters = 0

    # Main loop: single-verified lazy-greedy with periodic top-k refresh
    while len(selected) < k and heap and iters < MAX_ITERS:
        iters += 1
        neg_est_gain, idx, ver = heapq.heappop(heap)
        if idx in selected_set:
            continue
        est_gain = -neg_est_gain

        # If entry is up-to-date wrt current selection, accept it immediately
        if ver == version:
            # numeric guard
            if est_gain < 0 and est_gain > -tol:
                est_gain = 0.0
            selected.append(int(idx))
            selected_set.add(int(idx))
            # update current points and hv
            if current_points.shape[0] == 0:
                current_points = points[idx].reshape(1, -1).copy()
            else:
                current_points = np.vstack([current_points, points[idx].reshape(1, -1)])
            current_hv = float(pg.hypervolume(current_points).compute(ref))
            version += 1
            accepts_since_refresh += 1
            # perform periodic refresh of top candidates to reduce repeated single recomputes
            if accepts_since_refresh >= REFRESH_PERIOD:
                accepts_since_refresh = 0
                # Extract up to REFRESH_TOP entries that are not selected, recompute their true gains
                temp = []
                extracted = 0
                while heap and extracted < REFRESH_TOP:
                    neg_e, j, vj = heapq.heappop(heap)
                    if j in selected_set:
                        continue
                    temp.append((j, -neg_e))
                    extracted += 1
                # recompute true gains for extracted and push back with current version
                for (j, old_est) in temp:
                    tg, hvw = true_marginal_gain(j)
                    heapq.heappush(heap, (-float(tg), int(j), version))
                # loop continues
            continue

        # Otherwise (stale estimate), compute true marginal for this candidate only, push back with current version
        tg, hvw = true_marginal_gain(idx)
        # push back updated estimate (stamped with current version)
        heapq.heappush(heap, (-float(tg), int(idx), version))

    # If we stopped prematurely (heap exhausted or iteration cap), fill remaining with highest individual hv not selected
    if len(selected) < k:
        unselected = [i for i in range(N) if i not in selected_set]
        unselected_sorted = sorted(unselected, key=lambda i: indiv_hv[i], reverse=True)
        for idx in unselected_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))

    # Final safety: if still short, pad randomly
    if len(selected) < k:
        rng = np.random.default_rng(2025)
        unselected = [i for i in range(N) if i not in selected_set]
        need = k - len(selected)
        if len(unselected) == 0:
            idx_extra = rng.choice(N, size=need, replace=True)
            selected.extend([int(i) for i in idx_extra])
        else:
            picks = rng.choice(unselected, size=min(need, len(unselected)), replace=False)
            selected.extend([int(i) for i in picks])
            if len(selected) < k:
                extra = rng.choice(N, size=k - len(selected), replace=True)
                selected.extend([int(i) for i in extra])

    selected = selected[:k]
    return points[np.array(selected, dtype=int)].copy()


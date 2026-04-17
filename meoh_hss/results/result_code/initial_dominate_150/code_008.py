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

    # If k >= N: return all points; if k > N, pad with random duplicates to reach k (consistent behavior)
    if k >= N:
        if k == N:
            return points.copy()
        else:
            rng = np.random.default_rng(2025)
            extra = k - N
            idx_extra = rng.choice(N, size=extra, replace=True)
            return np.vstack([points.copy(), points[idx_extra]])

    # individual hypervolume (simple closed-form upper-bound / exact for singletons)
    diffs = ref.reshape(1, -1) - points  # shape (N, D)
    nonneg = np.maximum(diffs, 0.0)
    indiv_hv = np.prod(nonneg, axis=1)

    # Survivor set size M: keep a manageable pool to avoid full scans every iteration
    # Heuristic: at least 10, at most N, and scale with k
    M = int(min(max(10, 5 * k), N))

    # Initialize survivor set S as indices of top-M by individual hv
    order_by_indiv = np.argsort(indiv_hv)[::-1]
    survivors = list(order_by_indiv[:M])
    outside_pool = list(order_by_indiv[M:])  # remaining candidates ordered by indiv_hv (desc)

    # lazy heap over survivors: store (-estimated_gain, idx, version)
    # estimated_gain initially = indiv_hv (fast upper bound)
    heap = []
    for i in survivors:
        heap.append((-float(indiv_hv[int(i)]), int(i), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_points = np.empty((0, D), dtype=float)
    current_hv = 0.0
    version = 0  # increments when we accept a point; used to mark staleness

    # helper to compute true hv of union with candidate quickly using pygmo
    def hv_of_with_candidate(curr_pts, cand_idx):
        if curr_pts.shape[0] == 0:
            cand = points[cand_idx].reshape(1, -1)
            return pg.hypervolume(cand).compute(ref)
        else:
            arr = np.vstack([curr_pts, points[cand_idx].reshape(1, -1)])
            return pg.hypervolume(arr).compute(ref)

    # parameters
    tol = 1e-12
    MAX_ITERS = max(2000, k * 200)
    iters = 0

    # Main loop: lazy verification but restricted to dynamic survivor set; replenish survivors from outside_pool
    while len(selected) < k and heap and iters < MAX_ITERS:
        iters += 1
        neg_est, idx, ver = heapq.heappop(heap)
        idx = int(idx)
        if idx in selected_set:
            continue

        est_gain = -neg_est

        # If this estimate was computed w.r.t. current version, accept directly.
        if ver == version:
            # numerical guard
            if est_gain < 0 and est_gain > -tol:
                est_gain = 0.0
            selected.append(idx)
            selected_set.add(idx)
            # update current points and hv
            current_points = np.vstack([current_points, points[idx].reshape(1, -1)])
            current_hv = float(pg.hypervolume(current_points).compute(ref))
            version += 1

            # Replenish survivors to keep size ~M by pulling top from outside_pool (by indiv_hv)
            while len(heap) < M and outside_pool:
                j = outside_pool.pop(0)
                if j in selected_set:
                    continue
                heapq.heappush(heap, (-float(indiv_hv[int(j)]), int(j), version))
            continue

        # Otherwise, the entry is stale --- recompute its true marginal gain lazily
        hv_with = hv_of_with_candidate(current_points, idx)
        true_gain = float(hv_with - current_hv)
        if true_gain < 0 and true_gain > -tol:
            true_gain = 0.0

        # Push back updated estimate with current version
        heapq.heappush(heap, (-float(true_gain), int(idx), version))

        # If after pushing updated value this candidate is still (likely) the best, we will pop it again soon and accept
        # To avoid busy loops: if the recomputed gain is 0 and the heap's top has est 0 too, we may be stuck;
        # in that case, break to final filling by indiv_hv.
        if true_gain <= 0.0:
            # check top of heap
            if heap:
                top_neg, top_idx, top_ver = heap[0]
                top_est = -top_neg
                if top_est <= 0.0:
                    # no positive marginal gains remain among survivors -> try to bring in better outsiders
                    # bring a batch of outsiders into survivors (if any) and continue
                    brought = 0
                    while outside_pool and brought <  max(1, M // 4):
                        j = outside_pool.pop(0)
                        if j in selected_set:
                            continue
                        heapq.heappush(heap, (-float(indiv_hv[int(j)]), int(j), version))
                        brought += 1
                    # if no outsiders left to bring, we can break early
                    if brought == 0:
                        break
        # loop continues

    # If we stopped prematurely or heap exhausted, fill remaining with highest individual hv not selected
    if len(selected) < k:
        unselected = [i for i in range(N) if i not in selected_set]
        if unselected:
            unselected_sorted = sorted(unselected, key=lambda i: indiv_hv[i], reverse=True)
            for idx in unselected_sorted:
                if len(selected) >= k:
                    break
                selected.append(int(idx))
                selected_set.add(int(idx))

    # Final safety: if still short (very unlikely), pad randomly
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

    # ensure exactly k indices (trim or keep order)
    selected = selected[:k]

    return points[np.array(selected, dtype=int)].copy()


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

    # Fast closed-form individual hypervolume (upper bound / exact for singletons)
    diffs = ref.reshape(1, -1) - points  # shape (N, D)
    nonneg = np.maximum(diffs, 0.0)
    indiv_hv = np.prod(nonneg, axis=1)

    # Prepare lazy heap: store (-gain_estimate, idx, version)
    # version corresponds to number of selected points at the time the gain was computed
    heap = []
    for i in range(N):
        heap.append((-float(indiv_hv[i]), int(i), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_points = np.empty((0, D), dtype=float)
    current_hv = 0.0
    version = 0  # equals len(selected)

    # helper to compute true hv of union with candidate quickly using pygmo
    def hv_of_with_candidate(curr_pts, cand_idx):
        if curr_pts.shape[0] == 0:
            cand = points[cand_idx].reshape(1, -1)
            return pg.hypervolume(cand).compute(ref)
        else:
            arr = np.vstack([curr_pts, points[cand_idx].reshape(1, -1)])
            return pg.hypervolume(arr).compute(ref)

    # batch size for verifying top candidates (heuristic)
    BATCH_SIZE = max(2, min(10, N))  # at least 2, at most 10

    # main loop: batch-verified lazy-greedy
    MAX_ITERS = max(1000, k * 100)  # safety cap
    iters = 0
    tol = 1e-12

    while len(selected) < k and heap and iters < MAX_ITERS:
        iters += 1
        # Pop the top candidate
        neg_est_gain, idx, ver = heapq.heappop(heap)
        if idx in selected_set:
            continue  # skip already selected
        est_gain = -neg_est_gain

        # If the entry was computed w.r.t. current selection, accept directly (classic lazy-greedy)
        if ver == version:
            # numerical guard
            if est_gain < 0 and est_gain > -tol:
                est_gain = 0.0
            selected.append(int(idx))
            selected_set.add(int(idx))
            # update current points and hv
            current_points = np.vstack([current_points, points[idx].reshape(1, -1)])
            current_hv = pg.hypervolume(current_points).compute(ref)
            version += 1
            continue

        # Otherwise, build a batch of top outdated candidates to verify at once
        batch = [(idx, est_gain)]
        # gather up to BATCH_SIZE-1 more candidates
        while len(batch) < BATCH_SIZE and heap:
            neg_e, j, vj = heapq.heappop(heap)
            if j in selected_set:
                continue
            batch.append((j, -neg_e))
        # For each candidate in the batch, compute true marginal gain w.r.t. current selection
        true_gains = []
        for (j, _) in batch:
            hv_with = hv_of_with_candidate(current_points, j)
            tg = hv_with - current_hv
            # numerical correction
            if tg < 0 and tg > -tol:
                tg = 0.0
            true_gains.append((j, float(tg), hv_with))

        # select the best candidate from the batch
        # if all gains are equal and zero, we still pick the first highest indiv_hv fallback to avoid stalling
        best_idx, best_gain, best_hv_with = max(true_gains, key=lambda t: (t[1], indiv_hv[t[0]]))
        # Accept the best
        selected.append(int(best_idx))
        selected_set.add(int(best_idx))
        # update current points and hv with the stored hv (we computed hv_with for best)
        current_points = np.vstack([current_points, points[best_idx].reshape(1, -1)])
        current_hv = float(best_hv_with)
        # increment version (we had computed true gains w.r.t. previous version)
        prev_version = version
        version += 1

        # push back the rest of the batch with their computed gains tied to prev_version
        for (j, tg, hvw) in true_gains:
            if j == best_idx:
                continue
            # push back with version equal to the version when their gain was computed (prev_version)
            # Their computed tg is with respect to the state before we accepted best_idx, so it'll be stale after acceptance.
            heapq.heappush(heap, (-float(tg), int(j), prev_version))

    # If we stopped prematurely (heap exhausted or iteration cap), fill remaining with highest individual hv not selected
    if len(selected) < k:
        unselected = [i for i in range(N) if i not in selected_set]
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


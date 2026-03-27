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

    # trivial handling
    if k <= 0:
        return np.empty((0, D), dtype=float)
    if N == 0:
        return np.empty((0, D), dtype=float)

    # reference point
    if reference_point is None:
        ref = np.max(points, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).ravel()
        if ref.size != D:
            raise ValueError("reference_point must have the same dimensionality as points")

    # If k >= N: return all points (pad if requested larger)
    if k >= N:
        if k == N:
            return points.copy()
        else:
            rng = np.random.default_rng(2025)
            extra = k - N
            idx_extra = rng.choice(N, size=extra, replace=True)
            return np.vstack([points.copy(), points[idx_extra]])

    # fast singleton hypervolume (closed form)
    diffs = ref.reshape(1, -1) - points  # (N, D)
    nonneg = np.maximum(diffs, 0.0)
    indiv_hv = np.prod(nonneg, axis=1)

    # Prefilter top candidates by indiv_hv to reduce pool size (major speedup)
    # Heuristic pool size: grow with k but capped for speed
    POOL_MIN = 50
    POOL_PER_K = 6
    POOL_MAX = 800
    pool_size = int(min(N, max(POOL_MIN, min(POOL_MAX, POOL_PER_K * k))))
    if pool_size < 10:
        pool_size = min(N, 10)

    sorted_idx = np.argsort(-indiv_hv)
    pool_indices = [int(i) for i in sorted_idx[:pool_size]]
    in_pool = set(pool_indices)

    # lazy heap over pool: (-estimated_gain, idx, version)
    heap = []
    for idx in pool_indices:
        heap.append((-float(indiv_hv[idx]), int(idx), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_points = np.empty((0, D), dtype=float)
    current_hv = 0.0
    version = 0
    tol = 1e-12

    def hv_with_candidate(curr_pts, cand_idx):
        # returns hv(curr_pts U {cand}) as float
        if curr_pts.shape[0] == 0:
            arr = points[cand_idx].reshape(1, -1)
            return float(pg.hypervolume(arr).compute(ref))
        else:
            arr = np.vstack([curr_pts, points[cand_idx].reshape(1, -1)])
            return float(pg.hypervolume(arr).compute(ref))

    # adaptive batch size: small to limit evaluations, scaled with k
    def batch_size_func(remaining_pool):
        return max(2, min(8, int(2 + min(6, max(0, k // 5))))) if remaining_pool > 2 else remaining_pool

    # Main loop: batch-verified lazy-greedy inside small pool
    MAX_ITERS = max(1000, k * 200)
    iters = 0
    while len(selected) < k and heap and iters < MAX_ITERS:
        iters += 1
        neg_est, idx, ver = heapq.heappop(heap)
        if idx in selected_set:
            continue
        est_gain = -neg_est

        # If estimate was computed w.r.t current selection, accept directly
        if ver == version:
            if est_gain < 0 and est_gain > -tol:
                est_gain = 0.0
            selected.append(int(idx))
            selected_set.add(int(idx))
            # update current points and hv
            current_points = np.vstack([current_points, points[idx].reshape(1, -1)])
            current_hv = float(pg.hypervolume(current_points).compute(ref))
            version += 1
            # small early exit: if marginal gain is negligibly small, stop greedy
            if est_gain <= tol:
                break
            continue

        # otherwise, verify a small batch at once
        batch = [(idx, est_gain)]
        # gather more top candidates
        while len(batch) < batch_size_func(len(heap) + len(batch)) and heap:
            neg_e, j, vj = heapq.heappop(heap)
            if j in selected_set:
                continue
            batch.append((j, -neg_e))

        # compute true gains for the batch
        true_info = []
        for (j, _) in batch:
            hvw = hv_with_candidate(current_points, j)
            tg = hvw - current_hv
            if tg < 0 and tg > -tol:
                tg = 0.0
            true_info.append((j, float(tg), float(hvw)))

        # pick best by gain, tie-break by indiv_hv
        best_j, best_gain, best_hv_with = max(true_info, key=lambda t: (t[1], indiv_hv[t[0]]))
        # if the best gain is negligible, stop early and fill remaining by indiv_hv ordering
        if best_gain <= tol:
            break

        # accept best
        selected.append(int(best_j))
        selected_set.add(int(best_j))
        current_points = np.vstack([current_points, points[best_j].reshape(1, -1)])
        current_hv = float(best_hv_with)
        prev_version = version
        version += 1

        # push back other batch members with their computed gains tied to prev_version
        for (j, tg, hvw) in true_info:
            if j == best_j:
                continue
            heapq.heappush(heap, (-float(tg), int(j), prev_version))

    # If we broke early or ran out of heap, fill remaining from pool (by indiv_hv) then from global indiv_hv
    if len(selected) < k:
        # first consider remaining pool indices not selected
        pool_remaining = [i for i in pool_indices if i not in selected_set]
        # sort by indiv_hv descending
        pool_remaining_sorted = sorted(pool_remaining, key=lambda i: indiv_hv[i], reverse=True)
        for idx in pool_remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))

    if len(selected) < k:
        # fill from global indiv_hv ordering excluding already selected
        for idx in sorted_idx:
            if idx in selected_set:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            if len(selected) >= k:
                break

    # final safety pad (should not be necessary)
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


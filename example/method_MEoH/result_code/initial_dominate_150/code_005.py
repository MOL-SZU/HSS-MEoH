import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    # lazy import for pygmo
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo (pg) is required for hypervolume computations but is not available.") from e

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N, D = points.shape

    # trivial cases
    if k <= 0 or N == 0:
        return np.empty((0, D), dtype=float)
    if k >= N:
        return points.copy()

    # reference point
    if reference_point is None:
        ref = np.max(points, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).ravel()
        if ref.size != D:
            raise ValueError("reference_point must have the same dimensionality as points")

    # compute singleton hypervolumes (closed form for minimization to reference)
    diffs = ref.reshape(1, -1) - points  # (N, D)
    nonneg = np.maximum(diffs, 0.0)
    indiv_hv = np.prod(nonneg, axis=1)

    # Build a compact candidate pool to limit expensive HV evaluations:
    # Pool size scales with k but capped to a moderate constant for speed.
    POOL_MIN = 40
    POOL_PER_K = 5
    POOL_MAX = 800
    pool_size = int(min(N, max(POOL_MIN, min(POOL_MAX, POOL_PER_K * max(1, k)))))
    # For extremely small N, ensure at least k
    pool_size = max(pool_size, min(N, k))

    # We'll preselect a larger short-list by top singleton hv, then sample a diverse subset with farthest-first.
    shortlist_multiplier = 5
    shortlist_size = min(N, max(pool_size * shortlist_multiplier, 200))
    sorted_idx = np.argsort(-indiv_hv)
    shortlist = list(sorted_idx[:shortlist_size])

    # If shortlist is small, just take it
    if len(shortlist) <= pool_size:
        pool_indices = shortlist.copy()
    else:
        # Farthest-first selection within shortlist to ensure diversity
        pool_indices = []
        # Seed with the highest indiv hv
        first = shortlist[0]
        pool_indices.append(int(first))
        remaining = shortlist[1:]
        # compute squared distances incrementally to avoid large memory if possible
        rem_pts = points[np.array(remaining, dtype=int)]
        sel_pt = points[int(first)].reshape(1, -1)
        # initial min distances
        min_d2 = np.sum((rem_pts - sel_pt) ** 2, axis=1)
        while len(pool_indices) < pool_size and len(remaining) > 0:
            # pick the farthest
            idx_arg = int(np.argmax(min_d2))
            chosen = remaining[idx_arg]
            pool_indices.append(int(chosen))
            # remove chosen from remaining and min_d2
            remaining.pop(idx_arg)
            min_d2 = np.delete(min_d2, idx_arg)
            if len(remaining) == 0 or len(pool_indices) >= pool_size:
                break
            # update min_d2 with distances to newly added point
            new_pt = points[int(chosen)].reshape(1, -1)
            # compute distances to remaining in batch
            rem_array = points[np.array(remaining, dtype=int)]
            d2_new = np.sum((rem_array - new_pt) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2_new)

    # Prepare lazy heap for pool entries: (-estimate, idx, version)
    heap = []
    for idx in pool_indices:
        heap.append((-float(indiv_hv[idx]), int(idx), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_points = np.empty((0, D), dtype=float)
    current_hv = 0.0
    version = 0  # number of selected points when estimate was computed
    tol = 1e-12

    # helper: compute hv of current_points U candidate
    def hv_with_candidate(curr_pts: np.ndarray, cand_idx: int) -> float:
        if curr_pts.shape[0] == 0:
            arr = points[cand_idx].reshape(1, -1)
            return float(pg.hypervolume(arr).compute(ref))
        else:
            arr = np.vstack([curr_pts, points[cand_idx].reshape(1, -1)])
            return float(pg.hypervolume(arr).compute(ref))

    # batch-verified lazy greedy inside pool
    BATCH_SIZE = min(8, max(2, pool_size // 10))
    MAX_ITERS = max(1000, k * 200)
    iters = 0

    while len(selected) < k and heap and iters < MAX_ITERS:
        iters += 1
        neg_est, idx, ver = heapq.heappop(heap)
        if idx in selected_set:
            continue
        est = -neg_est

        # if estimate is up-to-date accept and compute hv precisely for update
        if ver == version:
            # numerical guard
            if est < 0 and est > -tol:
                est = 0.0
            selected.append(int(idx))
            selected_set.add(int(idx))
            current_points = np.vstack([current_points, points[idx].reshape(1, -1)])
            current_hv = float(pg.hypervolume(current_points).compute(ref))
            version += 1
            continue

        # stale entry: gather a small batch of top candidates for verification
        batch = [(idx, est)]
        while len(batch) < BATCH_SIZE and heap:
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
            true_info.append((int(j), float(tg), float(hvw)))

        # choose best candidate by true marginal gain, tie-break by indiv_hv
        best_j, best_gain, best_hv_with = max(true_info, key=lambda t: (t[1], indiv_hv[t[0]]))
        # early stop if no meaningful gains
        if best_gain <= tol:
            break

        # accept best
        selected.append(int(best_j))
        selected_set.add(int(best_j))
        current_points = np.vstack([current_points, points[best_j].reshape(1, -1)])
        current_hv = float(best_hv_with)
        prev_version = version
        version += 1

        # push back the rest of batch with their computed gains tied to prev_version
        for (j, tg, hvw) in true_info:
            if j == best_j:
                continue
            heapq.heappush(heap, (-float(tg), int(j), prev_version))

    # If we didn't reach k, fill from remaining pool (by indiv_hv) then from global indiv_hv
    if len(selected) < k:
        # remaining in pool
        pool_remaining = [i for i in pool_indices if i not in selected_set]
        pool_remaining_sorted = sorted(pool_remaining, key=lambda i: indiv_hv[i], reverse=True)
        for idx in pool_remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))

    if len(selected) < k:
        # global fill by indiv_hv
        for idx in sorted_idx:
            if idx in selected_set:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            if len(selected) >= k:
                break

    # Final safety pad (unlikely)
    if len(selected) < k:
        rng = np.random.default_rng(2026)
        unselected = [i for i in range(N) if i not in selected_set]
        need = k - len(selected)
        if len(unselected) == 0:
            picks = rng.choice(N, size=need, replace=True)
            selected.extend([int(x) for x in picks])
        else:
            picks = rng.choice(unselected, size=min(need, len(unselected)), replace=False)
            selected.extend([int(x) for x in picks])
            if len(selected) < k:
                extra = rng.choice(N, size=k - len(selected), replace=True)
                selected.extend([int(x) for x in extra])

    selected = selected[:k]
    # return in index order for determinism
    selected = sorted(int(i) for i in selected)
    return points[np.array(selected, dtype=int), :].copy()


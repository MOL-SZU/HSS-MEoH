import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    # lazy import for pygmo hypervolume
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

    # NEW parameterization: more aggressive compact pool and diversity weighting
    POOL_MIN = 20
    POOL_PER_K = 3
    POOL_MAX = 400
    pool_size = int(min(N, max(POOL_MIN, min(POOL_MAX, POOL_PER_K * max(1, k)))))
    pool_size = max(pool_size, min(N, k))

    # shortlist before farthest-first: smaller multiplier for speed
    shortlist_multiplier = 3
    shortlist_size = min(N, max(pool_size * shortlist_multiplier, 100))
    sorted_idx = np.argsort(-indiv_hv)
    shortlist = list(sorted_idx[:shortlist_size])

    # If shortlist is small, just take it
    if len(shortlist) <= pool_size:
        pool_indices = shortlist.copy()
    else:
        # Farthest-first selection within shortlist to ensure diversity (deterministic)
        pool_indices = []
        first = shortlist[0]
        pool_indices.append(int(first))
        remaining = shortlist[1:]
        rem_pts = points[np.array(remaining, dtype=int)]
        sel_pt = points[int(first)].reshape(1, -1)
        min_d2 = np.sum((rem_pts - sel_pt) ** 2, axis=1)
        while len(pool_indices) < pool_size and len(remaining) > 0:
            idx_arg = int(np.argmax(min_d2))
            chosen = remaining[idx_arg]
            pool_indices.append(int(chosen))
            # remove chosen
            remaining.pop(idx_arg)
            min_d2 = np.delete(min_d2, idx_arg)
            if len(remaining) == 0 or len(pool_indices) >= pool_size:
                break
            new_pt = points[int(chosen)].reshape(1, -1)
            rem_array = points[np.array(remaining, dtype=int)]
            d2_new = np.sum((rem_array - new_pt) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2_new)

    pool_indices = [int(i) for i in pool_indices]

    # Compute a static diversity score for pool candidates (min distance to other pool members)
    pool_pts = points[np.array(pool_indices, dtype=int)]
    if pool_pts.shape[0] <= 1:
        div_scores = np.zeros(len(pool_indices), dtype=float)
    else:
        # pairwise squared distances
        # efficient computation: (a-b)^2 = aa + bb - 2ab
        AA = np.sum(pool_pts * pool_pts, axis=1).reshape(-1, 1)
        BB = AA.T
        AB = pool_pts.dot(pool_pts.T)
        d2 = AA + BB - 2.0 * AB
        # numerical safety
        d2[d2 < 0] = 0.0
        # set diagonal to large value so min excludes self
        np.fill_diagonal(d2, np.inf)
        min_d2 = np.min(d2, axis=1)
        div_scores = np.sqrt(min_d2)  # use Euclidean min distance

    # normalize diversity and indiv_hv within pool to [0,1]
    eps = 1e-12
    indiv_pool = indiv_hv[np.array(pool_indices, dtype=int)]
    max_ind = np.max(indiv_pool) if indiv_pool.size > 0 else 0.0
    indiv_norm = indiv_pool / (max_ind + eps)
    max_div = np.max(div_scores) if div_scores.size > 0 else 0.0
    div_norm = div_scores / (max_div + eps)

    # blended score: weighted multiplicative boost of normalized hv by diversity
    beta = 1.2  # diversity importance multiplier (different from original which used no diversity)
    blended_score = indiv_norm * (1.0 + beta * div_norm)

    # Prepare lazy heap for pool entries: (-estimate, idx, version)
    heap = []
    for s, idx in zip(blended_score, pool_indices):
        heap.append((-float(s), int(idx), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_points = np.empty((0, D), dtype=float)
    current_hv = 0.0
    version = 0
    tol = 1e-12

    # helper: compute hv of current_points U candidate
    def hv_with_candidate(curr_pts: np.ndarray, cand_idx: int) -> float:
        if curr_pts.shape[0] == 0:
            arr = points[cand_idx].reshape(1, -1)
            return float(pg.hypervolume(arr).compute(ref))
        else:
            arr = np.vstack([curr_pts, points[cand_idx].reshape(1, -1)])
            return float(pg.hypervolume(arr).compute(ref))

    # batch-verified lazy greedy inside pool with slightly larger batches
    BATCH_SIZE = min(16, max(3, pool_size // 8))
    MAX_ITERS = max(2000, k * 300)
    iters = 0

    while len(selected) < k and heap and iters < MAX_ITERS:
        iters += 1
        neg_est, idx, ver = heapq.heappop(heap)
        if idx in selected_set:
            continue
        est = -neg_est

        # up-to-date estimate: accept directly
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

        # choose best candidate by true marginal gain, tie-break by indiv_hv and diversity
        # Get diversity and indiv_hv for tie-break
        def tie_key(t):
            j = t[0]
            indiv = float(indiv_hv[j])
            div = 0.0
            if j in pool_indices:
                try:
                    pos = pool_indices.index(j)
                    div = float(div_norm[pos])
                except ValueError:
                    div = 0.0
            return (t[1], indiv, div)

        best_j, best_gain, best_hv_with = max(true_info, key=tie_key)
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
        pool_remaining = [i for i in pool_indices if i not in selected_set]
        pool_remaining_sorted = sorted(pool_remaining, key=lambda i: indiv_hv[i], reverse=True)
        for idx in pool_remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))

    if len(selected) < k:
        for idx in sorted_idx:
            if idx in selected_set:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            if len(selected) >= k:
                break

    # Final safety pad (very unlikely)
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
    selected = sorted(int(i) for i in selected)
    return points[np.array(selected, dtype=int), :].copy()


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    import random
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = pts.shape

    if k <= 0:
        return np.empty((0, d))
    if k > n:
        k = n

    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    # ensure reference is at least slightly larger than any coordinate to avoid negative volumes
    eps = 1e-12
    min_coords = np.min(pts, axis=0)
    reference_point = np.maximum(reference_point, min_coords + eps)

    # fast hypervolume wrapper
    def hv_of(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return 0.0
        arr2 = np.atleast_2d(arr)
        hv = pg.hypervolume(arr2)
        return float(hv.compute(reference_point))

    # Stage 1: approximate coverage using Latin Hypercube Sampling (compact m)
    # choose sample size sensitive to k and d but small to speed up
    m = int(min(2000, max(400, 120 * min(max(1, k), max(1, d)))))
    m = max(400, m)

    rng = np.random.RandomState(42)
    lower = min_coords.copy()
    span = reference_point - lower
    span_adj = span.copy()
    span_adj[span_adj <= 0] = 1.0

    # LHS sampling
    samples_unit = np.empty((m, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(m)
        offsets = rng.rand(m)
        samples_unit[:, j] = (perm + offsets) / float(m)
    samples = lower + samples_unit * span_adj

    # compute coverage matrix in chunks to avoid memory blowups
    def compute_coverage_matrix(pts_arr, smpls):
        N, D = pts_arr.shape
        T = smpls.shape[0]
        cov = np.zeros((N, T), dtype=bool)
        max_mem = int(5e6)  # heuristic
        chunk_size = max(1, int(max_mem // max(1, T)))
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            block = pts_arr[start:end, None, :] <= smpls[None, :, :]
            cov[start:end] = np.all(block, axis=2)
        return cov

    coverage = compute_coverage_matrix(pts, samples)  # shape (n, m)
    per_point_counts = coverage.sum(axis=1)

    raw_vols = (reference_point[None, :] - pts)
    raw_vols[raw_vols < eps] = eps
    box_volumes = np.prod(raw_vols, axis=1)
    box_volumes = np.maximum(box_volumes, eps)

    beta = 0.5
    initial_scores = per_point_counts * (box_volumes ** beta)

    # CELF on sampled coverage to pick a compact candidate pool (size p)
    heap = [(-initial_scores[i], i, 0) for i in range(n)]
    heapq.heapify(heap)
    covered = np.zeros(m, dtype=bool)
    selected_sampled = []
    selected_set = set()
    current_size = 0
    # choose pool size as min(2k, n) but at least k and limited to reasonable number
    pool_size = min(n, max(k, min(n, 2 * k)))
    # but keep cap to avoid too large pool if k huge
    pool_size = min(pool_size, max(k, 200))

    while len(selected_sampled) < pool_size and heap:
        neg_score, idx, stamp = heapq.heappop(heap)
        if stamp == current_size:
            if idx in selected_set:
                continue
            selected_sampled.append(idx)
            selected_set.add(idx)
            covered |= coverage[idx]
            current_size += 1
            continue
        # recompute true marginal on samples
        newly = np.count_nonzero(coverage[idx] & (~covered))
        new_score = newly * (box_volumes[idx] ** beta)
        heapq.heappush(heap, (-new_score, idx, current_size))

    if len(selected_sampled) < pool_size:
        for _, i, _ in sorted(heap, key=lambda x: x[0]):
            if i not in selected_set:
                selected_sampled.append(i)
                selected_set.add(i)
                if len(selected_sampled) >= pool_size:
                    break
        if len(selected_sampled) < pool_size:
            for i in range(n):
                if i not in selected_set:
                    selected_sampled.append(i)
                    selected_set.add(i)
                    if len(selected_sampled) >= pool_size:
                        break

    # Stage 2: exact CELF on compact candidate pool using pygmo hypervolume (much fewer hv calls)
    pool_indices = list(dict.fromkeys(selected_sampled))  # preserve order, dedup
    p = len(pool_indices)
    # prepare initial heap with single-point hv as upper bounds
    heap2 = []
    for idx in pool_indices:
        g = hv_of([pts[idx]])
        heap2.append((-g, idx, -1))
    heapq.heapify(heap2)

    selected_final = []
    selected_mask = np.zeros(n, dtype=bool)
    curr_hv = 0.0
    iter_id = 0

    while len(selected_final) < k and heap2:
        neg_gain, idx, last_iter = heapq.heappop(heap2)
        cached_gain = -neg_gain
        if last_iter == iter_id:
            if selected_mask[idx]:
                continue
            selected_final.append(idx)
            selected_mask[idx] = True
            # update curr_hv exactly
            # compute hv of selected set
            sel_pts = pts[selected_final]
            curr_hv = hv_of(sel_pts)
            iter_id += 1
            continue
        # recompute true marginal gain
        if len(selected_final) == 0:
            true_gain = hv_of([pts[idx]])
        else:
            cand_set = np.vstack([pts[selected_final], pts[idx]])
            true_gain = hv_of(cand_set) - curr_hv
        heapq.heappush(heap2, (-true_gain, idx, iter_id))

    # fill if not enough
    if len(selected_final) < k:
        for i in pool_indices:
            if not selected_mask[i]:
                selected_final.append(i)
                selected_mask[i] = True
                if len(selected_final) == k:
                    break
    selected_final = selected_final[:k]
    selected_points = pts[np.array(selected_final, dtype=int)].copy()
    curr_hv = hv_of(selected_points) if selected_points.size else 0.0

    # lightweight randomized swap refinement inside candidate pool to improve exact hv
    max_swaps = 150
    swaps = 0
    unselected_in_pool = [i for i in pool_indices if not selected_mask[i]]
    while swaps < max_swaps and len(unselected_in_pool) > 0:
        swaps += 1
        improved = False
        # sample some unselected and some positions
        sample_un = random.sample(unselected_in_pool, min(40, len(unselected_in_pool)))
        sample_pos = random.sample(range(len(selected_points)), min(8, len(selected_points)))
        best_impr = 0.0
        best_swap = None
        for pos in sample_pos:
            for cand in sample_un:
                candidate = selected_points.copy()
                candidate[pos] = pts[cand]
                hv_cand = hv_of(candidate)
                gain = hv_cand - curr_hv
                if gain > best_impr + 1e-12:
                    best_impr = gain
                    best_swap = (pos, cand, hv_cand)
        if best_swap is not None:
            pos, cand_idx, hv_after = best_swap
            old_idx = selected_final[pos]
            selected_mask[old_idx] = False
            selected_final[pos] = cand_idx
            selected_mask[cand_idx] = True
            selected_points[pos] = pts[cand_idx].copy()
            curr_hv = hv_after
            unselected_in_pool = [i for i in pool_indices if not selected_mask[i]]
            improved = True
        if not improved:
            break  # early stop

    return np.asarray(selected_points, dtype=float)


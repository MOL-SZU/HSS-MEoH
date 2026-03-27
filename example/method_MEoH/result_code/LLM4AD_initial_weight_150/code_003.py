import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    import random
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")

    n, d = points.shape
    if k <= 0:
        return np.empty((0, d))
    k = min(k, n)

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def hv_of(array_like):
        if array_like is None or len(array_like) == 0:
            return 0.0
        arr = np.array(array_like)
        hv = pg.hypervolume(arr)
        return float(hv.compute(reference_point))

    # CELF (lazy greedy) priority queue:
    # Each entry: (-cached_gain, idx, last_updated_iter)
    # last_updated_iter indicates the iteration number when cached_gain was computed.
    heap = []
    # compute initial individual hypervolumes as upper bounds for marginal gains
    for i in range(n):
        g = hv_of([points[i]])
        # store negative gain because heapq is a min-heap
        heap.append((-g, i, -1))
    heapq.heapify(heap)

    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []
    selected_points = []
    curr_hv = 0.0
    iter_id = 0
    tol = 1e-12

    while len(selected_indices) < k and heap:
        # Pop best candidate
        neg_gain, idx, last_iter = heapq.heappop(heap)
        cached_gain = -neg_gain
        # If cached_gain was computed in this iteration, accept it directly
        if last_iter == iter_id:
            # Accept candidate
            selected_indices.append(idx)
            selected_mask[idx] = True
            selected_points.append(points[idx].copy())
            curr_hv += max(cached_gain, 0.0)
            iter_id += 1
            continue
        # Otherwise, recompute true marginal gain with current set
        if len(selected_points) == 0:
            true_gain = hv_of([points[idx]])
        else:
            cand_set = np.vstack([np.array(selected_points), points[idx]])
            true_gain = hv_of(cand_set) - curr_hv
        # Push back with updated iteration stamp
        heapq.heappush(heap, (-true_gain, idx, iter_id))
        # loop continues to pop top again

    # In case of ties or emptiness, fill arbitrarily
    if len(selected_indices) < k:
        for i in range(n):
            if not selected_mask[i]:
                selected_indices.append(i)
                selected_mask[i] = True
                selected_points.append(points[i].copy())
                if len(selected_indices) == k:
                    break
        curr_hv = hv_of(np.vstack(selected_points)) if len(selected_points) > 0 else 0.0

    selected_points = np.array(selected_points)

    # Lightweight randomized pairwise-swap refinement:
    # Try a bounded number of random swaps (sampled) to improve hypervolume.
    max_swap_iters = 200  # budget for sampled swaps
    if n - k > 0:
        swap_iters = 0
        improved = True
        while improved and swap_iters < max_swap_iters:
            improved = False
            swap_iters += 1
            # sample a small set of candidate unselected indices and selected positions
            sample_unselected_size = min(50, n - k)
            sample_selected_size = min(10, k)
            unselected_candidates = [i for i in range(n) if not selected_mask[i]]
            if len(unselected_candidates) == 0:
                break
            sampled_unselected = random.sample(unselected_candidates, min(sample_unselected_size, len(unselected_candidates)))
            sampled_positions = random.sample(range(k), sample_selected_size)
            # evaluate sampled swaps and accept best improving one
            best_impr = 0.0
            best_swap = None
            for pos in sampled_positions:
                for cand in sampled_unselected:
                    candidate_set = selected_points.copy()
                    candidate_set[pos] = points[cand]
                    hv_cand = hv_of(candidate_set)
                    improvement = hv_cand - curr_hv
                    if improvement > best_impr + tol:
                        best_impr = improvement
                        best_swap = (pos, cand, hv_cand)
            if best_swap is not None:
                pos, cand_idx, hv_after = best_swap
                old_idx = selected_indices[pos]
                selected_mask[old_idx] = False
                selected_indices[pos] = cand_idx
                selected_mask[cand_idx] = True
                selected_points[pos] = points[cand_idx].copy()
                curr_hv = hv_after
                improved = True

    return np.array(selected_points)


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    try:
        import pygmo as pg
    except Exception as _pg_err:
        raise ImportError("pygmo (pg) is required for hypervolume computations but is not available.") from _pg_err

    import numpy as np
    import heapq

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, m = points.shape

    if k <= 0:
        return np.empty((0, m), dtype=float)

    if k >= n:
        return points.copy()

    # Reference point default
    if reference_point is None:
        ref = np.max(points, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).ravel()
        if ref.size != m:
            raise ValueError("reference_point must have the same dimensionality as points")

    # Helper: compute hypervolume for a set of indices using pygmo
    def hv_of_indices(idx_list):
        if len(idx_list) == 0:
            return 0.0
        arr = points[np.array(idx_list, dtype=int), :]
        return float(pg.hypervolume(arr).compute(ref))

    # Precompute individual HV (useful for initial priorities and tie-breaking)
    diffs = ref.reshape(1, -1) - points
    indiv_hv = np.prod(np.maximum(diffs, 0.0), axis=1)

    # Lazy greedy with max-heap (implemented via min-heap storing negative gains).
    # Heap entries: ( -estimated_gain, -indiv_hv[idx], idx, seen_size )
    # seen_size stores the size of the selected set when the estimate was computed.
    heap = []
    for i in range(n):
        # initial estimate is individual hv (since hv(empty)=0)
        est = float(indiv_hv[i])
        heap.append((-est, -float(indiv_hv[i]), int(i), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    hv_current = 0.0
    eps = 1e-12
    # main selection loop
    while len(selected) < k and heap:
        neg_est, neg_indhv, idx, seen_size = heapq.heappop(heap)
        if idx in selected_set:
            continue  # already chosen by another pop
        # If the estimate was computed at the same selected size, accept it (lazy greedy correctness)
        if seen_size == len(selected):
            # select idx
            new_hv = hv_of_indices(selected + [idx])
            # due to numerical issues, ensure strict improvement or accept if no other choice
            hv_gain = new_hv - hv_current
            # Accept even if zero (to fill up to k)
            selected.append(int(idx))
            selected_set.add(int(idx))
            hv_current = new_hv
            continue
        else:
            # recompute true marginal gain
            true_hv = hv_of_indices(selected + [idx])
            true_gain = true_hv - hv_current
            # push updated entry with current seen_size
            heapq.heappush(heap, (-float(true_gain), neg_indhv, int(idx), len(selected)))
            # loop continues; next pop will either be this updated element or another
            continue

    # If for some reason we didn't reach k (shouldn't happen), pad with top individual hv among remaining
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        if remaining:
            # pick top by individual hv
            rem_sorted = sorted(remaining, key=lambda i: indiv_hv[i], reverse=True)
            for i in rem_sorted:
                if len(selected) >= k:
                    break
                selected.append(int(i))
                selected_set.add(int(i))

    # Small bounded swap-based refinement: try at most R swaps that increase hv
    # This is a light post-processing step different from heavy local search in other algorithms
    R = min(200, n * 2)
    attempts = 0
    improved = True
    while attempts < R and improved:
        improved = False
        attempts += 1
        # try each selected element and see if swapping with any unselected improves hv
        unselected = [i for i in range(n) if i not in selected_set]
        if not unselected:
            break
        for s_idx in list(selected):
            base = [x for x in selected if x != s_idx]
            best_gain = 0.0
            best_cand = None
            # try a limited sample of candidates to keep cost down
            # sample size:
            sample_size = min(50, len(unselected))
            # pick highest individual hv among unselected as candidates (more promising)
            cand_sorted = sorted(unselected, key=lambda i: indiv_hv[i], reverse=True)[:sample_size]
            for cand in cand_sorted:
                hv_after = hv_of_indices(base + [cand])
                gain = hv_after - hv_current
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_cand = cand
            if best_cand is not None:
                # perform swap
                pos = selected.index(s_idx)
                selected[pos] = int(best_cand)
                selected_set.remove(int(s_idx))
                selected_set.add(int(best_cand))
                hv_current = hv_of_indices(selected)
                improved = True
                break  # restart outer attempts loop when an improvement is made

    # Finalize: ensure exactly k distinct indices
    selected = list(dict.fromkeys(selected))
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected]
        need = k - len(selected)
        if remaining:
            add = sorted(remaining, key=lambda i: indiv_hv[i], reverse=True)[:need]
            selected.extend([int(x) for x in add])

    # Return selected points in the order of selection
    selected = selected[:k]
    result = points[np.array(selected, dtype=int), :].copy()
    return result


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import pygmo as pg
    import heapq
    import random

    # Basic validation and normalization
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D).")
    n, d = points.shape
    if not (isinstance(k, int) and 0 < k <= n):
        raise ValueError("k must be an integer with 0 < k <= number of points")

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,) matching points' dimension")

    # Helper to compute hypervolume for a set of indices
    def hv_of_indices(indices):
        if len(indices) == 0:
            return 0.0
        try:
            arr = points[np.array(indices)]
            return float(pg.hypervolume(arr).compute(reference_point))
        except Exception:
            return 0.0

    # Precompute individual hypervolumes (single point contributions)
    individual_hv = np.zeros(n, dtype=float)
    for i in range(n):
        try:
            individual_hv[i] = float(pg.hypervolume(points[[i]]).compute(reference_point))
        except Exception:
            individual_hv[i] = 0.0

    # Candidate preselection: focus effort on the most promising points by individual hv
    min_candidates = max(5 * k, 50)  # heuristic: at least 50 or 5*k, whichever larger
    num_candidates = min(n, max(k, min_candidates))
    # select indices of top individual_hv
    candidate_indices = np.argsort(individual_hv)[-num_candidates:][::-1]
    candidate_set = set(candidate_indices.tolist())

    # Initialize lazy-greedy (CELF) structures
    # last_gain: last computed marginal gain for each candidate (initialized to individual hv)
    # last_updated: selection size when the last_gain was computed
    last_gain = {int(idx): float(individual_hv[int(idx)]) for idx in candidate_indices}
    last_updated = {int(idx): 0 for idx in candidate_indices}

    # Max-heap built with negative gains
    heap = [(-last_gain[idx], random.random(), idx) for idx in candidate_indices]  # random tie-breaker
    heapq.heapify(heap)

    selected_indices = []
    selected_set = set()
    current_hv = 0.0
    selected_points_indices = []

    tol = 1e-12
    # Main lazy greedy loop
    while len(selected_indices) < k and heap:
        neg_gain, _, idx = heapq.heappop(heap)
        idx = int(idx)
        # If last computed at current selection size, accept it
        if last_updated[idx] == len(selected_indices):
            # If marginal is effectively zero (numerical), still accept to fill k
            selected_indices.append(idx)
            selected_set.add(idx)
            selected_points_indices.append(idx)
            # Update current_hv
            current_hv = hv_of_indices(selected_points_indices)
            # continue to next selection
            continue
        # Otherwise recompute marginal gain w.r.t current selection
        # form candidate selection
        candidate_sel = selected_points_indices + [idx]
        hv_with = hv_of_indices(candidate_sel)
        marginal = hv_with - current_hv
        # Ensure non-negative numerically
        if marginal < 0 and marginal > -1e-12:
            marginal = 0.0
        last_gain[idx] = float(marginal)
        last_updated[idx] = len(selected_indices)
        # push back with updated gain (use random tie-breaker to avoid deterministic degeneracy)
        heapq.heappush(heap, (-last_gain[idx], random.random(), idx))

    # If we still haven't selected k (heap exhausted), fill with best remaining by individual hv
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: individual_hv[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected_indices) >= k:
                break
            selected_indices.append(int(idx))
            selected_set.add(int(idx))

    # Final safety: if somehow more than k, truncate; if less, pad
    selected_indices = selected_indices[:k]
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: individual_hv[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected_indices) >= k:
                break
            selected_indices.append(int(idx))

    # Return points in the order selected
    subset = points[np.array(selected_indices, dtype=int)]
    return subset


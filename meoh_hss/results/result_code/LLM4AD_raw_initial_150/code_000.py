import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

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

    # attempt to import pygmo hypervolume; if missing, raise informative error
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations: %s" % e)

    # Helper to compute hypervolume for a set of indices
    def hv_of_indices(indices):
        if len(indices) == 0:
            return 0.0
        try:
            arr = points[np.array(indices)]
            return float(pg.hypervolume(arr).compute(reference_point))
        except Exception:
            return 0.0

    # Compute individual hypervolume contributions
    individual_hv = np.zeros(n, dtype=float)
    for i in range(n):
        try:
            individual_hv[i] = float(pg.hypervolume(points[[i]]).compute(reference_point))
        except Exception:
            individual_hv[i] = 0.0

    # Normalize individual hv to [0,1] for blending
    if np.ptp(individual_hv) > 0:
        scaled_hv = (individual_hv - individual_hv.min()) / (individual_hv.max() - individual_hv.min())
    else:
        scaled_hv = np.zeros_like(individual_hv)

    # Diversity proxy: distance to reference point (encourage spread away from ref)
    dist_ref = np.linalg.norm(points - reference_point, axis=1)
    if np.ptp(dist_ref) > 0:
        norm_dist = (dist_ref - dist_ref.min()) / (dist_ref.max() - dist_ref.min())
    else:
        norm_dist = np.zeros_like(dist_ref)

    # Composite initial score: convex blend of normalized hv and normalized distance
    # New parameters (different from original):
    alpha = 0.75  # weight on normalized individual hv (favor hv more strongly)
    beta = 1.0 - alpha  # weight on normalized distance
    # Add a small epsilon to break ties in favor of larger raw hv
    composite_score = alpha * scaled_hv + beta * norm_dist
    composite_score += 1e-9 * (individual_hv / (1.0 + individual_hv.max() if individual_hv.max() > 0 else 1.0))

    # Candidate preselection: focused pool (smaller than original to speed up)
    min_candidates = max(12 * k, 80)  # different setting: moderate candidate pool
    num_candidates = min(n, max(k, int(min_candidates)))
    candidate_indices = np.argsort(composite_score)[-num_candidates:][::-1]
    candidate_indices = [int(x) for x in candidate_indices]
    candidate_set = set(candidate_indices)

    # Initialize CELF (lazy greedy) structures
    # Use composite score as initial approximate marginal gain
    last_gain = {idx: float(composite_score[idx]) for idx in candidate_indices}
    last_updated = {idx: 0 for idx in candidate_indices}

    # Max-heap: sort by (-approx_gain, -individual_hv, idx) to break ties preferring larger true hv
    heap = [(-last_gain[idx], -individual_hv[idx], int(idx)) for idx in candidate_indices]
    heapq.heapify(heap)

    selected_indices = []
    selected_set = set()
    current_hv = 0.0
    selected_points_indices = []

    # Main lazy greedy loop
    while len(selected_indices) < k and heap:
        neg_gain, neg_ind_hv, idx = heapq.heappop(heap)
        idx = int(idx)
        # If last computed at current selection size, accept it
        if last_updated.get(idx, -1) == len(selected_indices):
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            selected_points_indices.append(idx)
            current_hv = hv_of_indices(selected_points_indices)
            continue
        # Otherwise recompute true marginal gain w.r.t current selection
        hv_with = hv_of_indices(selected_points_indices + [idx])
        marginal = hv_with - current_hv
        if marginal < 0 and marginal > -1e-12:
            marginal = 0.0
        last_gain[idx] = float(marginal)
        last_updated[idx] = len(selected_indices)
        # push back with updated exact gain (tie-break by individual hv)
        heapq.heappush(heap, (-last_gain[idx], -individual_hv[idx], idx))

    # If not enough selected (heap exhausted), fill with best remaining by composite score then individual hv
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: (composite_score[x], individual_hv[x]), reverse=True)
        for idx in remaining_sorted:
            if len(selected_indices) >= k:
                break
            selected_indices.append(int(idx))
            selected_set.add(int(idx))

    # Final safety: ensure exactly k
    selected_indices = selected_indices[:k]
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: (composite_score[x], individual_hv[x]), reverse=True)
        for idx in remaining_sorted:
            if len(selected_indices) >= k:
                break
            selected_indices.append(int(idx))

    # Ensure we return a numpy array of shape (k, D)
    selected_indices = selected_indices[:k]
    subset = points[np.array(selected_indices, dtype=int)]
    return subset


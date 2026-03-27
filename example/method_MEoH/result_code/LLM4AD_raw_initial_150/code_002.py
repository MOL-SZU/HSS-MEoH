import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations") from e

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

    # Helper: compute hypervolume for indices (safe)
    def hv_of_indices(indices):
        if len(indices) == 0:
            return 0.0
        try:
            arr = points[np.array(indices, dtype=int)]
            return float(pg.hypervolume(arr).compute(reference_point))
        except Exception:
            return 0.0

    # Compute individual hypervolume contributions (singletons)
    individual_hv = np.zeros(n, dtype=float)
    for i in range(n):
        try:
            individual_hv[i] = float(pg.hypervolume(points[[i]]).compute(reference_point))
        except Exception:
            individual_hv[i] = 0.0

    # Diversity proxy: distance to reference point (encourage spread away from ref)
    dist_ref = np.linalg.norm(points - reference_point, axis=1)

    # Normalize individual_hv and dist_ref to [0,1]
    def normalize_arr(x):
        x = np.asarray(x, dtype=float)
        rng = np.ptp(x)
        if rng > 0:
            return (x - x.min()) / rng
        else:
            return np.zeros_like(x)

    hv_norm = normalize_arr(individual_hv)
    dist_norm = normalize_arr(dist_ref)

    # New composite score parameters (different from original)
    # alpha: weight on diversity (distance), favoring more spread (set > 0.5)
    # beta: small amplification of high-HV points after combining
    alpha = 0.65  # stronger emphasis on diversity
    beta = 0.15   # mild amplification of hv-rich candidates

    # Composite initial score: convex combination of normalized hv and distance,
    # then mildly amplify by a function of hv_norm to prefer high hv among ties.
    composite_score = (1.0 - alpha) * hv_norm + alpha * dist_norm
    composite_score = composite_score * (1.0 + beta * hv_norm)

    # Candidate preselection: smaller focused pool than original, but with an upper cap
    min_candidates = max(8 * k, 100)
    max_cap = 500
    num_candidates = min(n, max(k, min(min_candidates, max_cap)))
    candidate_indices = np.argsort(composite_score)[-num_candidates:][::-1]
    candidate_indices = [int(x) for x in candidate_indices]
    candidate_set = set(candidate_indices)

    # Initialize CELF (lazy greedy) structures
    last_gain = {idx: float(composite_score[idx]) for idx in candidate_indices}
    last_updated = {idx: -1 for idx in candidate_indices}

    # Max-heap: sort by (-approx_gain, -individual_hv, idx) to break ties preferring larger true hv
    heap = [(-last_gain[idx], -individual_hv[idx], idx) for idx in candidate_indices]
    heapq.heapify(heap)

    selected_indices = []
    selected_set = set()
    current_hv = 0.0
    selected_list_for_hv = []

    # Main lazy greedy loop (CELF)
    while len(selected_indices) < k and heap:
        neg_gain, neg_ind_hv, idx = heapq.heappop(heap)
        idx = int(idx)
        # If this candidate was last updated at current selection size, accept it
        if last_updated[idx] == len(selected_indices):
            selected_indices.append(idx)
            selected_set.add(idx)
            selected_list_for_hv.append(idx)
            current_hv = hv_of_indices(selected_list_for_hv)
            continue
        # Otherwise recompute exact marginal gain w.r.t current selection
        hv_with = hv_of_indices(selected_list_for_hv + [idx])
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

    subset = points[np.array(selected_indices, dtype=int)]
    return subset


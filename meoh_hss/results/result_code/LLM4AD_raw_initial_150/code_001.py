import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for exact hypervolume computations") from e

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

    # Helper to compute hypervolume for a set of indices (uses pygmo)
    def hv_of_indices(indices):
        if len(indices) == 0:
            return 0.0
        try:
            arr = points[np.array(indices, dtype=int)]
            return float(pg.hypervolume(arr).compute(reference_point))
        except Exception:
            return 0.0

    # Compute individual hypervolume contributions (single-box volumes)
    individual_hv = np.zeros(n, dtype=float)
    for i in range(n):
        try:
            individual_hv[i] = float(pg.hypervolume(points[[i]]).compute(reference_point))
        except Exception:
            individual_hv[i] = 0.0

    # Diversity proxy: distance to reference point (encourage spread away from reference)
    dist_ref = np.linalg.norm(points - reference_point, axis=1)
    if np.ptp(dist_ref) > 0:
        norm_dist = (dist_ref - dist_ref.min()) / (dist_ref.max() - dist_ref.min())
    else:
        norm_dist = np.zeros_like(dist_ref)

    # Log-scale individual HV for robustness to outliers, then normalize to [0,1]
    ind_hv_log = np.log1p(individual_hv)
    if np.ptp(ind_hv_log) > 0:
        ind_hv_norm = (ind_hv_log - ind_hv_log.min()) / (ind_hv_log.max() - ind_hv_log.min())
    else:
        ind_hv_norm = np.zeros_like(ind_hv_log)

    # New composite score parameters (different from original)
    gamma = 2.5    # stronger multiplicative bias towards distant points
    eta = 0.15     # additive diversity weight
    beta = 1.5     # power for the additive distance term

    # Composite: log-scaled normalized hv amplified by distance (multiplicative) plus power-law additive diversity
    composite_score = ind_hv_norm * (1.0 + gamma * norm_dist) + eta * (norm_dist ** beta)

    # Candidate preselection: smaller, more focused candidate pool
    min_candidates = max(4 * k, 50)
    num_candidates = min(n, max(k, min_candidates))
    candidate_indices = np.argsort(composite_score)[-num_candidates:][::-1]
    candidate_indices = [int(x) for x in candidate_indices]
    candidate_set = set(candidate_indices)

    # Initialize CELF (lazy greedy) structures
    last_gain = {int(idx): float(composite_score[int(idx)]) for idx in candidate_indices}
    last_updated = {int(idx): -1 for idx in candidate_indices}  # -1 means never updated

    # Max-heap: sort by (-approx_gain, -individual_hv, idx) to break ties preferring larger true hv
    heap = [(-last_gain[int(idx)], -individual_hv[int(idx)], int(idx)) for idx in candidate_indices]
    heapq.heapify(heap)

    selected_indices = []
    selected_set = set()
    current_hv = 0.0
    selected_points_indices = []

    # Main lazy greedy loop (CELF)
    while len(selected_indices) < k and heap:
        neg_gain, neg_ind_hv, idx = heapq.heappop(heap)
        idx = int(idx)
        # If last computed at current selection size, accept it
        if last_updated.get(idx, -2) == len(selected_indices):
            if idx in selected_set:
                # already selected by some race condition (shouldn't normally happen), skip
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

    subset = points[np.array(selected_indices, dtype=int)]
    return subset


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N, D = points.shape

    if k <= 0:
        return np.zeros((0, D), dtype=points.dtype)
    if k >= N:
        return points.copy()

    # Reference handling
    if reference_point is None:
        ref = points.max(axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float)
        if ref.ndim == 0:
            ref = np.full((D,), float(ref))
        if ref.shape != (D,):
            raise ValueError("reference_point must have shape (D,)")
        # ensure reference >= max points
        max_p = points.max(axis=0)
        ref = np.maximum(ref, max_p + 1e-12)

    # Discard points that are trivially invalid (any coordinate > ref makes rect empty)
    valid_mask = np.all(points <= ref + 1e-12, axis=1)
    if not np.all(valid_mask):
        points = points[valid_mask]
        N = points.shape[0]
        if N == 0:
            return np.zeros((0, D), dtype=points.dtype)
        if k >= N:
            return points.copy()

    # Pareto (non-dominated) filtering for minimization: remove points dominated by others
    # A point i is dominated if exists j != i s.t. points[j] <= points[i] (all dims) and any strictly <
    keep = np.ones(N, dtype=bool)
    for i in range(N):
        if not keep[i]:
            continue
        # check if any other dominates i
        leq = np.all(points <= points[i], axis=1)
        lt = np.any(points < points[i], axis=1)
        dominated_by = leq & lt
        # ignore self
        dominated_by[i] = False
        if np.any(dominated_by):
            keep[i] = False
    if not np.all(keep):
        points = points[keep]
        N = points.shape[0]
        if N == 0:
            return np.zeros((0, D), dtype=points.dtype)
        if k >= N:
            return points.copy()

    # Monte-Carlo sampling box: from lower = min(points) to upper = ref
    lower = points.min(axis=0)
    upper = ref.copy()
    # avoid zero-range problems
    zero_range = (upper - lower) <= 0
    if np.any(zero_range):
        upper = upper.copy()
        upper[zero_range] = lower[zero_range] + 1e-6

    # Choose modest sample size to trade quality vs speed
    M = int(min(15000, max(2000, 300 * k)))
    rng = np.random.default_rng(123456)
    u = rng.random((M, D))
    samples = lower + u * (upper - lower)

    # Precompute dominated samples: sample s is in rect of point i if s >= point (componentwise)
    dominated = (samples[:, None, :] >= points[None, :, :]).all(axis=2)  # shape (M, N)

    # Individual counts used as initial estimates
    individual_counts = dominated.sum(axis=0)

    # Lazy greedy: max-heap of (-estimated_gain, idx)
    heap = [(-int(individual_counts[i]), int(i)) for i in range(N)]
    heapq.heapify(heap)

    available = np.ones(N, dtype=bool)
    covered = np.zeros(M, dtype=bool)
    selected_idx = []

    # helper to compute true marginal gain for candidate idx relative to current covered
    def marginal_gain(idx):
        # number of samples newly covered by idx
        return int(np.count_nonzero(dominated[:, idx] & ~covered))

    # Main lazy-greedy selection
    selects = min(k, N)
    while len(selected_idx) < selects and heap:
        est_neg, idx = heapq.heappop(heap)
        if not available[idx]:
            continue
        est = -int(est_neg)
        # recompute true marginal
        true_gain = marginal_gain(idx)
        if true_gain == est:
            # accept
            selected_idx.append(idx)
            available[idx] = False
            covered |= dominated[:, idx]
        else:
            # push updated estimate back and continue
            heapq.heappush(heap, (-true_gain, idx))

    # If not enough selected (shouldn't happen often), fill by largest individual_counts
    if len(selected_idx) < k:
        remaining = np.where(available)[0]
        if remaining.size > 0:
            order = remaining[np.argsort(-individual_counts[remaining])]
            need = k - len(selected_idx)
            for idx in order[:need]:
                selected_idx.append(int(idx))
                available[idx] = False

    # Bounded local swap improvement (sample-estimated HV using covered sample count)
    # Prepare dominated arrays for current selection
    selected_idx = selected_idx[:k]
    selected_set = set(selected_idx)
    current_covered = covered.copy()
    current_score = int(np.count_nonzero(current_covered))

    # Prepare list of promising outsiders (by individual_counts)
    outsiders = np.where(~np.isin(np.arange(N), selected_idx))[0]
    if outsiders.size > 0:
        outsider_order = outsiders[np.argsort(-individual_counts[outsiders])]
    else:
        outsider_order = np.array([], dtype=int)

    # Limit attempts: only try at most S swaps and consider top T outsiders
    S = min(10, max(1, len(selected_idx)))  # number of selected items to try
    T = min(50, outsider_order.size)        # number of outsiders to consider
    # We'll try first S selected slots (or random subset if longer)
    try_sel_indices = list(range(min(S, len(selected_idx))))

    for si in try_sel_indices:
        sel_idx = selected_idx[si]
        # compute base coverage excluding this selected point
        base = np.zeros(M, dtype=bool)
        for j, sidx in enumerate(selected_idx):
            if j == si:
                continue
            base |= dominated[:, sidx]
        # try candidate outsiders
        for cand in outsider_order[:T]:
            new_cov = base | dominated[:, cand]
            new_score = int(np.count_nonzero(new_cov))
            if new_score > current_score:
                # perform swap
                selected_set.remove(sel_idx)
                selected_set.add(int(cand))
                selected_idx[si] = int(cand)
                current_covered = new_cov
                current_score = new_score
                # update available/outsider lists
                outsider_order = outsider_order[outsider_order != cand]
                outsider_order = np.append(outsider_order, np.array([sel_idx], dtype=int))
                break  # move to next selected slot

    # Ensure exactly k selection (pad if necessary with largest individual_counts)
    if len(selected_idx) < k:
        remaining = np.setdiff1d(np.arange(N), selected_idx, assume_unique=True)
        if remaining.size > 0:
            order = remaining[np.argsort(-individual_counts[remaining])]
            need = k - len(selected_idx)
            selected_idx.extend(order[:need].tolist())

    selected_idx = selected_idx[:k]
    result = points[np.array(selected_idx, dtype=int)].copy()
    return result


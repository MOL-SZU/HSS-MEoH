import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    from typing import Optional

    # Helper: compute hypervolume using pygmo if available, otherwise Monte-Carlo approx
    def compute_hv(subset: np.ndarray, ref: np.ndarray) -> float:
        if subset.size == 0:
            return 0.0
        try:
            import pygmo as pg  # type: ignore
            hv = pg.hypervolume(subset)
            return float(hv.compute(ref))
        except Exception:
            return approx_hv_mc(subset, ref)

    def approx_hv_mc(subset: np.ndarray, ref: np.ndarray, base_samples: int = 5000) -> float:
        # Monte Carlo estimate of hypervolume: sample uniformly in [mins, ref], count dominated samples
        D = subset.shape[1]
        mins = np.min(subset, axis=0)
        # If any mins >= ref, the volume is zero
        if np.any(mins >= ref):
            return 0.0
        # number of samples scales with dimension but bounded
        samples = int(min(max(base_samples, 1000 * D), 20000))
        # draw samples
        rng = np.random.default_rng()
        u = rng.random((samples, D))
        samples_coords = mins + u * (ref - mins)
        # for minimization HV: a sample s is dominated if there exists p with p <= s (elementwise)
        # we can vectorize check: for each p, mark samples dominated by that p
        dominated = np.zeros(samples, dtype=bool)
        for p in subset:
            # p <= samples_coords along all dims
            dominated |= np.all(p <= samples_coords, axis=1)
            if dominated.all():
                break
        frac = dominated.mean()
        volume_box = np.prod(ref - mins)
        return float(frac * volume_box)

    # Input validation and setup
    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype=float)
    else:
        points = points.astype(float, copy=False)

    n, D = points.shape
    if k <= 0 or k > n:
        raise ValueError("k must be in 1..n")

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.array(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Clip points that are worse than reference in any dimension (they contribute zero volume beyond ref)
    clipped_points = np.minimum(points, reference_point)

    # Precompute single-point hypervolumes (cheap: product of (ref - p) in minimization)
    single_hv = np.maximum(0.0, reference_point - clipped_points)
    # product over axis; if any (ref - p) < 0 then hv is zero
    single_hv = np.prod(single_hv, axis=1)

    # Lazy greedy with max-heap: store tuples (-gain, idx, version)
    # version = size_selected when gain was computed
    heap = []
    for idx in range(n):
        gain = float(single_hv[idx])
        heap.append((-gain, idx, 0))
    heapq.heapify(heap)

    selected_indices = []
    selected_mask = np.zeros(n, dtype=bool)
    current_selected_points = np.empty((0, D))
    current_hv = 0.0
    selected_size = 0

    # Main loop: select k elements
    # To avoid pathological behavior if many points give zero gain, allow fallback to choose top single_hv order
    attempts = 0
    max_attempts = n * 10 + k * 100

    while selected_size < k and attempts < max_attempts and heap:
        attempts += 1
        neg_gain, idx, version = heapq.heappop(heap)
        stored_gain = -neg_gain
        # If stored version equals current selected_size, the gain is up-to-date -> accept
        if version == selected_size:
            # If the gain is effectively 0, we still accept the best remaining until k is filled
            selected_indices.append(idx)
            selected_mask[idx] = True
            # update current_selected_points and hv
            if current_selected_points.size == 0:
                current_selected_points = clipped_points[[idx], :]
            else:
                current_selected_points = np.vstack([current_selected_points, clipped_points[idx]])
            current_hv = compute_hv(current_selected_points, reference_point)
            selected_size += 1
            # continue to next selection
            continue
        # Otherwise, need to recompute marginal gain w.r.t current selection
        # compute hv(selected ∪ {idx}) - current_hv
        if selected_size == 0:
            new_hv = float(single_hv[idx])
        else:
            # build union set points for hv computation: use already-clipped selected points + candidate
            cand_set = np.vstack([current_selected_points, clipped_points[idx]])
            new_hv = compute_hv(cand_set, reference_point)
        marginal = max(0.0, new_hv - current_hv)
        # push back with updated version
        heapq.heappush(heap, (-float(marginal), idx, selected_size))

    # If for any reason we didn't select enough (e.g., heap exhausted), fill remaining by highest single_hv not chosen
    if selected_size < k:
        remaining = [i for i in np.argsort(-single_hv) if not selected_mask[i]]
        need = k - selected_size
        for i in remaining[:need]:
            selected_indices.append(i)
            selected_mask[i] = True
            if current_selected_points.size == 0:
                current_selected_points = clipped_points[[i], :]
            else:
                current_selected_points = np.vstack([current_selected_points, clipped_points[i]])
            selected_size += 1

    # Prepare return: original (unclipped) points corresponding to selected indices
    selected_indices = selected_indices[:k]
    if len(selected_indices) < k:
        # pad with random remaining indices if necessary
        unselected = [i for i in range(n) if i not in selected_indices]
        rng = np.random.default_rng()
        need = k - len(selected_indices)
        if len(unselected) > 0 and need > 0:
            extra = list(rng.choice(unselected, size=min(need, len(unselected)), replace=False))
            selected_indices.extend(extra)

    subset = points[selected_indices]
    # ensure shape (k, D)
    if subset.shape[0] < k:
        # pad by repeating last row if extremely necessary (shouldn't happen)
        if subset.shape[0] == 0:
            pad = np.tile(reference_point, (k, 1))
            subset = pad
        else:
            last = subset[-1:]
            pad = np.repeat(last, k - subset.shape[0], axis=0)
            subset = np.vstack([subset, pad])
    return subset.astype(float)


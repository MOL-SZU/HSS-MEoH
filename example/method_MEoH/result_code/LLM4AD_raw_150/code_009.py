import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np

    EPS = 1e-12
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N, D = points.shape
    if N == 0 or k <= 0:
        return np.zeros((0, D), dtype=float)

    if reference_point is None:
        reference_point = points.max(axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(D,)

    # Build axis-aligned boxes between each point and reference
    lows = np.minimum(points, reference_point)
    highs = np.maximum(points, reference_point)

    # Compute individual box volumes
    diffs = np.maximum(highs - lows, 0.0)
    volumes = np.prod(diffs, axis=1)

    # Handle degenerate zero-volume points
    nonzero_mask = volumes > EPS
    if not np.any(nonzero_mask):
        select_count = min(k, N)
        return points[:select_count].copy()

    # Keep original indices for returning proper points
    idx_map = np.arange(N)
    # Remove zero-volume candidates to speed up
    lows = lows[nonzero_mask]
    highs = highs[nonzero_mask]
    volumes = volumes[nonzero_mask]
    idx_map = idx_map[nonzero_mask]
    M = len(volumes)
    if k >= M:
        out_idxs = list(idx_map)
        if len(out_idxs) < k:
            zeros_to_add = [i for i in range(N) if i not in out_idxs][: (k - len(out_idxs))]
            out_idxs.extend(zeros_to_add)
        return points[np.array(out_idxs, dtype=int)][:k].copy()

    # Determine global sampling region (bounding box of all candidate boxes)
    sample_low = np.min(lows, axis=0)
    sample_high = np.max(highs, axis=0)
    region_diffs = np.maximum(sample_high - sample_low, 0.0)
    total_region_vol = np.prod(region_diffs)

    # If global region has zero volume in any dimension, fallback to volume-based selection
    if total_region_vol <= EPS:
        order = np.argsort(-volumes)
        sel_local = order[:k]
        selected_original_idxs = idx_map[sel_local]
        subset = points[selected_original_idxs.astype(int), :].copy()
        return subset

    # Choose number of Monte-Carlo samples adaptively
    # Balance between accuracy and runtime
    S = int(min(20000, max(3000, 200 * int(max(1, k)))))
    # Generate uniform samples in the global region
    rng = np.random.default_rng()
    # Handle zero-length dims by broadcasting constant coordinate
    if np.any(region_diffs <= EPS):
        # For degenerate dims, just set that coordinate to the fixed value
        samples = np.empty((S, D), dtype=float)
        rand_part = rng.random((S, D), dtype=float)
        samples = sample_low + rand_part * region_diffs
    else:
        samples = sample_low + rng.random((S, D)) * region_diffs

    # Precompute dominance masks: dominated_masks[i, s] = True if sample s lies inside box i
    dominated_masks = np.empty((M, S), dtype=bool)
    # Build masks in a loop to avoid huge temporary broadcasted arrays when M or S is big
    for i in range(M):
        ge = samples >= lows[i]  # (S, D)
        le = samples <= highs[i]  # (S, D)
        dominated_masks[i, :] = np.all(ge & le, axis=1)

    # Greedy selection using sample coverage as proxy for hypervolume contribution
    covered = np.zeros(S, dtype=bool)  # which samples are already covered by selected boxes
    selected_mask = np.zeros(M, dtype=bool)
    selected_indices_local = []

    for _ in range(k):
        # Marginal counts = number of uncovered samples each candidate would newly cover
        # Use vectorized boolean operation
        uncovered = ~covered  # (S,)
        # If no uncovered samples, remaining gains are zero
        if not np.any(uncovered):
            break
        # compute marginal counts
        # dominated_masks[:, uncovered] has shape (M, sum(uncovered))
        # Sum along axis 1 to get counts
        marginal_counts = dominated_masks[:, uncovered].sum(axis=1)
        # Exclude already selected candidates
        marginal_counts[selected_mask] = -1
        best = int(np.argmax(marginal_counts))
        best_count = marginal_counts[best]
        if best_count <= 0:
            # no positive marginal gain; stop early
            break
        # select best
        selected_mask[best] = True
        selected_indices_local.append(int(best))
        # update covered samples
        covered |= dominated_masks[best, :]

    # If selected fewer than k, fill with largest-volume remaining boxes
    if len(selected_indices_local) < k:
        remaining = [i for i in range(M) if i not in selected_indices_local]
        if remaining:
            rem_vols = volumes[remaining]
            need = k - len(selected_indices_local)
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_indices_local.append(int(remaining[int(idx)]))

    # Map back to original indices
    selected_original_idxs = idx_map[np.array(selected_indices_local, dtype=int)]
    if len(selected_original_idxs) < k:
        missing = k - len(selected_original_idxs)
        add = [i for i in range(N) if i not in selected_original_idxs][:missing]
        if add:
            selected_original_idxs = np.concatenate([selected_original_idxs, np.array(add, dtype=int)])

    subset = points[selected_original_idxs[:k].astype(int), :].copy()
    return subset


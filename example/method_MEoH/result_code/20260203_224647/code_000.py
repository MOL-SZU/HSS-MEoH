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
    nonzero_mask = volumes > EPS
    if not np.any(nonzero_mask):
        select_count = min(k, N)
        return points[:select_count].copy()

    # Keep mapping to original indices
    idx_map = np.arange(N)
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

    # Precompute pairwise intersection volumes (M x M)
    L1 = lows[:, None, :]  # (M,1,D)
    L2 = lows[None, :, :]  # (1,M,D)
    H1 = highs[:, None, :]
    H2 = highs[None, :, :]
    inter_low = np.maximum(L1, L2)
    inter_high = np.minimum(H1, H2)
    inter_diffs = np.maximum(inter_high - inter_low, 0.0)
    pairwise_inter = np.prod(inter_diffs, axis=2)  # shape (M, M)
    pairwise_inter = np.maximum(pairwise_inter, 0.0)

    # New scoring parameters (different from original)
    p = 3.0      # stronger nonlinear penalization for overlap fraction
    beta = 0.15  # linear overlap penalty weight
    gamma = 0.9  # mild dampening of raw volume (reduces dominance of huge volumes)

    # Greedy selection with vectorized updates: est_gain = volumes**gamma * (1 - overlap_ratio**p - beta*overlap_ratio)
    overlap_sum = np.zeros(M, dtype=float)  # cumulative pairwise overlap with selected set
    selected_mask = np.zeros(M, dtype=bool)
    selected_indices_local = []

    while len(selected_indices_local) < k:
        denom = volumes + EPS
        overlap_ratio = overlap_sum / denom
        overlap_ratio = np.minimum(overlap_ratio, 1.0)

        est_gain = (np.power(volumes, gamma) *
                    (1.0 - np.power(overlap_ratio, p) - beta * overlap_ratio))
        # numerical safety
        est_gain = np.where(est_gain > 0.0, est_gain, 0.0)

        est_gain[selected_mask] = -1.0

        i = int(np.argmax(est_gain))
        if est_gain[i] <= EPS:
            break

        selected_mask[i] = True
        selected_indices_local.append(i)

        if len(selected_indices_local) < k:
            overlap_sum += pairwise_inter[:, i]

    if len(selected_indices_local) < k:
        remaining = [i for i in range(M) if i not in selected_indices_local]
        if remaining:
            rem_vols = volumes[remaining]
            need = k - len(selected_indices_local)
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_indices_local.append(int(remaining[int(idx)]))

    selected_original_idxs = idx_map[np.array(selected_indices_local, dtype=int)]
    if len(selected_original_idxs) < k:
        missing = k - len(selected_original_idxs)
        add = [i for i in range(N) if i not in selected_original_idxs][:missing]
        if add:
            selected_original_idxs = np.concatenate([selected_original_idxs, np.array(add, dtype=int)])

    subset = points[selected_original_idxs[:k].astype(int), :].copy()
    return subset


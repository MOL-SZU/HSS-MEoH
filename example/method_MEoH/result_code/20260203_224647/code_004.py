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

    # For candidate i, ratio_ji = pairwise_inter[j,i] / volume_i
    overlap_ratio_col = pairwise_inter / (volumes[None, :] + EPS)
    overlap_ratio_col = np.clip(overlap_ratio_col, 0.0, 1.0)

    selected_mask = np.zeros(M, dtype=bool)
    selected_indices_local = []

    # Greedy: marginal_gain(i) = volume_i * prod_{j in selected} (1 - pairwise_inter[j,i] / volume_i)
    while len(selected_indices_local) < k:
        if selected_mask.any():
            prod_terms = np.prod(1.0 - overlap_ratio_col[selected_mask, :], axis=0)
        else:
            prod_terms = np.ones(M, dtype=float)

        marginal_gain = volumes * prod_terms
        marginal_gain = np.where(marginal_gain > 0.0, marginal_gain, 0.0)
        marginal_gain[selected_mask] = -1.0  # exclude already selected

        i = int(np.argmax(marginal_gain))
        if marginal_gain[i] <= EPS:
            break

        selected_mask[i] = True
        selected_indices_local.append(i)

    # If not enough selected (due to tiny gains), fill by largest remaining volumes
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


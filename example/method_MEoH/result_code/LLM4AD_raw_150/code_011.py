import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

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
        # all volumes are zero: return arbitrary k points (first ones)
        select_count = min(k, N)
        return points[:select_count].copy()

    # Keep original indices for returning proper points
    idx_map = np.arange(N)
    # Optionally remove zero-volume candidates to speed up
    lows = lows[nonzero_mask]
    highs = highs[nonzero_mask]
    volumes = volumes[nonzero_mask]
    idx_map = idx_map[nonzero_mask]
    M = len(volumes)
    if k >= M:
        # return all nonzero-volume points plus some zero-volume if needed
        out_idxs = list(idx_map)
        if len(out_idxs) < k:
            # add zeros from original set
            zeros_to_add = [i for i in range(N) if i not in out_idxs][: (k - len(out_idxs))]
            out_idxs.extend(zeros_to_add)
        return points[np.array(out_idxs, dtype=int)][:k].copy()

    # Precompute pairwise intersection volumes (M x M)
    # inter_low[i,j,d] = max(lows[i,d], lows[j,d])
    # inter_high[i,j,d] = min(highs[i,d], highs[j,d])
    # intersection volume = prod(max(0, inter_high - inter_low))
    # Compute efficiently with broadcasting
    L1 = lows[:, None, :]  # (M,1,D)
    L2 = lows[None, :, :]  # (1,M,D)
    H1 = highs[:, None, :]
    H2 = highs[None, :, :]
    inter_low = np.maximum(L1, L2)
    inter_high = np.minimum(H1, H2)
    inter_diffs = np.maximum(inter_high - inter_low, 0.0)
    pairwise_inter = np.prod(inter_diffs, axis=2)  # shape (M, M)

    # Ensure numerical stability, clip small negatives
    pairwise_inter = np.maximum(pairwise_inter, 0.0)

    # Greedy selection using pairwise-subtraction approximation:
    # est_gain[i] = volumes[i] - sum_{j in selected} pairwise_inter[i,j]
    est_gain = volumes.copy()
    selected_mask = np.zeros(M, dtype=bool)
    selected_indices_local = []

    # max-heap via negatives
    heap = [(-float(est_gain[i]), int(i)) for i in range(M)]
    heapq.heapify(heap)

    while heap and len(selected_indices_local) < k:
        neg, i = heapq.heappop(heap)
        if selected_mask[i]:
            continue
        # lazy check: if current estimate differs, reinsert updated
        cur_est = est_gain[i]
        if abs(-neg - cur_est) > 1e-9:
            heapq.heappush(heap, (-float(cur_est), i))
            continue
        if cur_est <= EPS:
            # no positive gain left; stop early
            break
        # select i
        selected_mask[i] = True
        selected_indices_local.append(int(i))
        # update estimates: subtract pairwise intersection with newly selected box
        # For all remaining j: est_gain[j] -= pairwise_inter[j, i]
        if len(selected_indices_local) < k:
            # vectorized update
            dec = pairwise_inter[:, i]
            # avoid subtracting for already selected (they are masked anyway)
            est_gain = est_gain - dec
            # clip to zero
            est_gain = np.maximum(est_gain, 0.0)
            # push updated entries for remaining candidates
            for j in range(M):
                if not selected_mask[j]:
                    heapq.heappush(heap, (-float(est_gain[j]), int(j)))

    # If we selected fewer than k (e.g., due to zero marginal gains), fill by largest remaining volumes
    if len(selected_indices_local) < k:
        remaining = [i for i in range(M) if i not in selected_indices_local]
        if remaining:
            rem_vols = volumes[remaining]
            need = k - len(selected_indices_local)
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_indices_local.append(int(remaining[int(idx)]))

    # Map back to original point indices
    selected_original_idxs = idx_map[np.array(selected_indices_local, dtype=int)]
    # If we still have fewer than k (rare), pad with arbitrary remaining original indices
    if len(selected_original_idxs) < k:
        missing = k - len(selected_original_idxs)
        add = [i for i in range(N) if i not in selected_original_idxs][:missing]
        if add:
            selected_original_idxs = np.concatenate([selected_original_idxs, np.array(add, dtype=int)])

    subset = points[selected_original_idxs[:k].astype(int), :].copy()
    return subset


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import random
    from typing import List
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume calculations") from e

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array (N, D)")
    n, d = points.shape

    if k <= 0:
        return np.zeros((0, d))

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point).reshape(-1)
    if reference_point.shape[0] != d:
        raise ValueError("reference_point must have same dimension as points")

    # Quick nondominated filter (minimization assumed)
    def nondominated_indices(arr: np.ndarray) -> List[int]:
        N = arr.shape[0]
        is_nd = np.ones(N, dtype=bool)
        order = np.argsort(np.sum(arr, axis=1))  # cheap heuristic to accelerate
        for idx in order:
            if not is_nd[idx]:
                continue
            # mark any point dominated by arr[idx] as non-dominated=False
            comp = np.all(arr[idx] <= arr, axis=1) & np.any(arr[idx] < arr, axis=1)
            is_nd[comp] = False
            is_nd[idx] = True
        return list(np.nonzero(is_nd)[0])

    # cheap box-volume proxy (clip negatives)
    diff = np.maximum(reference_point - points, 0.0)
    # small eps to avoid exact zeros causing degenerate normalization
    eps = 1e-12
    box_vol = np.prod(diff + eps, axis=1)

    # Build initial candidate pool: prefer nondominated sorted by box_vol, then fill up to m with dominated best
    nd_idx = nondominated_indices(points)
    if len(nd_idx) == 0:
        candidate_indices = list(range(n))
    else:
        nd_sorted = sorted(nd_idx, key=lambda i: -box_vol[i])
        # shortlist size m: small to reduce HV calls but adaptive with k
        m = int(min(max(4 * k, 30), n))
        if len(nd_sorted) >= m:
            candidate_indices = nd_sorted[:m]
        else:
            # include dominated by box_vol until m
            remaining = [i for i in range(n) if i not in nd_sorted]
            rem_sorted = sorted(remaining, key=lambda i: -box_vol[i])
            candidate_indices = nd_sorted + rem_sorted[:max(0, m - len(nd_sorted))]

    # Ensure candidate list unique and bounded
    candidate_indices = list(dict.fromkeys(candidate_indices))  # keep order, remove duplicates
    if len(candidate_indices) < k:
        # fall back to all points
        candidate_indices = list(range(n))

    # Helper to compute hypervolume
    def compute_hv(arr: np.ndarray) -> float:
        if arr is None or arr.size == 0:
            return 0.0
        try:
            hv = pg.hypervolume(np.asarray(arr))
            return float(hv.compute(reference_point))
        except Exception:
            return 0.0

    selected_idx: List[int] = []
    selected_points: List[np.ndarray] = []

    # Precompute candidate arrays and box_vol for quick access
    cand_set = list(candidate_indices)

    # Greedy selection with limited exact HV evaluations per iteration
    target_k = min(k, n)
    hv_selected = 0.0
    selected_array = np.empty((0, d))

    # Per-iteration shortlist size L (limit exact hv computations)
    def shortlist_size(cand_count: int) -> int:
        return max(8, min(int(3 * k), cand_count, 60))  # bounded and adaptive

    # Using a simple centroid-based diversity to diversify picks cheaply
    for t in range(target_k):
        if len(cand_set) == 0:
            break

        # If nothing selected, pick the candidate with largest box_vol
        if t == 0:
            best = max(cand_set, key=lambda i: box_vol[i])
            selected_idx.append(best)
            selected_points.append(points[best])
            selected_array = np.vstack([selected_array, points[best]])
            hv_selected = compute_hv(selected_array)
            cand_set.remove(best) if best in cand_set else None
            continue

        # compute cheap proxy scores for remaining candidates
        cand_arr = points[cand_set]
        cand_box = box_vol[cand_set]
        centroid = np.mean(np.asarray(selected_points), axis=0)
        # diversity distance
        dif = cand_arr - centroid[None, :]
        dist = np.linalg.norm(dif, axis=1)
        # normalize proxies
        nb = cand_box
        ndist = dist
        if nb.max() > nb.min():
            nb_norm = (nb - nb.min()) / (nb.max() - nb.min() + eps)
        else:
            nb_norm = np.ones_like(nb)
        if ndist.max() > ndist.min():
            nd_norm = (ndist - ndist.min()) / (ndist.max() - ndist.min() + eps)
        else:
            nd_norm = np.ones_like(ndist)
        # combine proxy: favor box_vol but keep diversity
        proxy = 0.7 * nb_norm + 0.3 * nd_norm

        # choose top L by proxy for exact HV evaluation
        L = shortlist_size(len(cand_set))
        top_pos = np.argsort(-proxy)[:L]
        top_candidates = [cand_set[i] for i in top_pos]

        # evaluate exact marginal HV on top_candidates
        best_contrib = -np.inf
        best_cand = None
        hv_before = hv_selected
        # build base selected once
        base_selected = selected_array
        for ci in top_candidates:
            # stack and compute hv
            tmp = np.vstack([base_selected, points[ci]])
            hv_after = compute_hv(tmp)
            contrib = hv_after - hv_before
            if contrib > best_contrib:
                best_contrib = contrib
                best_cand = ci

        # Fallback if none selected
        if best_cand is None:
            best_cand = top_candidates[0]

        # Add best
        selected_idx.append(best_cand)
        selected_points.append(points[best_cand])
        selected_array = np.vstack([selected_array, points[best_cand]])
        hv_selected = compute_hv(selected_array)
        if best_cand in cand_set:
            cand_set.remove(best_cand)

    # If we didn't reach k (n < k), pad by best remaining box_vol
    if len(selected_idx) < k:
        remaining = [i for i in range(n) if i not in selected_idx]
        remaining_sorted = sorted(remaining, key=lambda i: -box_vol[i])
        for idx in remaining_sorted:
            if len(selected_idx) >= k:
                break
            selected_idx.append(idx)
            selected_points.append(points[idx])
            selected_array = np.vstack([selected_array, points[idx]]) if selected_array.size else np.array([points[idx]])
            hv_selected = compute_hv(selected_array)

    # Small randomized swap-based local improvement (limited budget)
    max_swaps = min(50, 10 * k)
    if len(selected_idx) == target_k and n > target_k:
        non_selected = [i for i in range(n) if i not in selected_idx]
        swaps = 0
        while swaps < max_swaps:
            si = random.randrange(0, len(selected_idx))
            ri = random.choice(non_selected)
            # attempt swap
            trial = selected_array.copy()
            trial[si] = points[ri]
            trial_hv = compute_hv(trial)
            if trial_hv > hv_selected + 1e-12:
                # accept swap
                old_idx = selected_idx[si]
                selected_idx[si] = ri
                selected_array = trial
                selected_points[si] = points[ri]
                hv_selected = trial_hv
                # update non_selected set
                non_selected.remove(ri)
                non_selected.append(old_idx)
            swaps += 1

    # Final selection array ensure shape (k, d)
    final = np.array(selected_points)
    if final.shape[0] > k:
        final = final[:k]
    elif final.shape[0] < k:
        # pad with best remaining by box_vol
        already = set(selected_idx)
        remaining = [i for i in range(n) if i not in already]
        rem_sorted = sorted(remaining, key=lambda i: -box_vol[i])
        need = k - final.shape[0]
        pads = [points[idx] for idx in rem_sorted[:need]]
        if len(pads) > 0:
            final = np.vstack([final, np.array(pads)])

    return final.reshape(-1, d)


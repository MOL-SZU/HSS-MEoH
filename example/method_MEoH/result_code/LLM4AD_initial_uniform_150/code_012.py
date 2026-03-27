import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = pts.shape

    if k <= 0:
        return np.empty((0, d))
    if k >= n:
        return pts.copy()

    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def compute_hv(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        hv = pg.hypervolume(arr)
        return float(hv.compute(reference_point))

    # Precompute single-point hypervolumes (used for seeding and tie-breaking)
    single_hv = np.zeros(n, dtype=float)
    for i in range(n):
        single_hv[i] = compute_hv(pts[[i], :])

    # Farthest-first seeding to obtain k cluster centers (diverse seeds)
    centers_idx = []
    # start with the point of largest single_hv (likely important extreme)
    first_idx = int(np.argmax(single_hv))
    centers_idx.append(first_idx)
    if k > 1:
        # distances to nearest center
        dist_sq = np.full(n, np.inf, dtype=float)
        # update distances
        diff = pts - pts[first_idx]
        dist_sq = np.sum(diff * diff, axis=1)
        for _ in range(1, k):
            # pick farthest point from existing centers
            next_idx = int(np.argmax(dist_sq))
            centers_idx.append(next_idx)
            # update distances
            diff = pts - pts[next_idx]
            dist_sq = np.minimum(dist_sq, np.sum(diff * diff, axis=1))

    centers_idx = list(dict.fromkeys(centers_idx))  # unique preserving order (if duplicates)
    # if unique centers less than k (rare), extend by highest single_hv not already chosen
    if len(centers_idx) < k:
        remaining = [i for i in np.argsort(-single_hv) if i not in centers_idx]
        need = k - len(centers_idx)
        centers_idx.extend(remaining[:need])

    centers = pts[centers_idx, :]

    # Assign each point to nearest center (Euclidean)
    diff = pts[:, None, :] - centers[None, :, :]  # shape (n, k, d)
    dists = np.sum(diff * diff, axis=2)  # (n, k)
    assign = np.argmin(dists, axis=1)  # (n,)

    # For each cluster pick the point with maximum single_hv (not yet selected)
    selected_set = set()
    selected_list = []
    for c in range(centers.shape[0]):
        members = np.where(assign == c)[0]
        if members.size == 0:
            continue
        # choose member with highest single hv (tie-break by index)
        best = members[np.argmax(single_hv[members])]
        if best not in selected_set:
            selected_list.append(int(best))
            selected_set.add(int(best))
        if len(selected_list) >= k:
            break

    # If still short, fill by remaining highest single_hv
    if len(selected_list) < k:
        remaining = [i for i in np.argsort(-single_hv) if i not in selected_set]
        need = k - len(selected_list)
        for idx in remaining[:need]:
            selected_list.append(int(idx))
            selected_set.add(int(idx))

    # Prepare current selected array and hv
    selected_list = selected_list[:k]
    current_selected = pts[np.array(selected_list, dtype=int), :].copy()
    current_hv = compute_hv(current_selected)

    # Local-best swap improvement: try swapping an unselected point with a selected one
    # until no improvement (best-improvement strategy)
    unselected_indices = [i for i in range(n) if i not in selected_set]
    max_rounds = 10  # number of full rounds of improvement
    eps = 1e-12

    for _ in range(max_rounds):
        best_gain = 0.0
        best_swap = None  # (sel_pos_in_list, unselected_idx, new_hv)
        # iterate over unselected points (can be many). To speed up, consider ordering by single_hv desc
        # so promising candidates are tested first
        unselected_indices_sorted = sorted(unselected_indices, key=lambda x: -single_hv[x])
        for u_idx in unselected_indices_sorted:
            # quick upper-bound test: if single_hv[u_idx] + tiny <= min single_hv of selected, skip?
            # Not a reliable bound; skip the bound to keep correctness.
            for pos, s_idx in enumerate(selected_list):
                # form swapped set
                new_sel = current_selected.copy()
                new_sel[pos, :] = pts[u_idx]
                new_hv = compute_hv(new_sel)
                gain = new_hv - current_hv
                if gain > best_gain + eps:
                    best_gain = float(gain)
                    best_swap = (pos, int(s_idx), int(u_idx), float(new_hv))
        if best_swap is None or best_gain <= eps:
            break
        # apply best swap
        pos, old_s_idx, new_u_idx, new_hv = best_swap
        # update selected_list, selected_set, current_selected, current_hv, unselected_indices
        selected_list[pos] = int(new_u_idx)
        selected_set.remove(int(old_s_idx))
        selected_set.add(int(new_u_idx))
        unselected_indices = [i for i in range(n) if i not in selected_set]
        current_selected[pos, :] = pts[new_u_idx]
        current_hv = new_hv

    # Final safeguard: ensure exactly k unique indices
    final_indices = list(dict.fromkeys(selected_list))
    if len(final_indices) < k:
        remaining = [i for i in np.argsort(-single_hv) if i not in final_indices]
        need = k - len(final_indices)
        final_indices.extend(remaining[:need])
    elif len(final_indices) > k:
        final_indices = final_indices[:k]

    selected_array = pts[np.array(final_indices, dtype=int), :]
    return selected_array


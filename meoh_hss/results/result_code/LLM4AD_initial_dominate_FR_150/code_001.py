import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    from typing import List
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations") from e

    if points is None or len(points) == 0 or k <= 0:
        return np.empty((0, 0))

    points = np.asarray(points, dtype=float)
    n, d = points.shape

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)

    def hv_compute(objs: np.ndarray) -> float:
        if objs is None or len(objs) == 0:
            return 0.0
        try:
            hv = pg.hypervolume(np.asarray(objs))
            return float(hv.compute(reference_point))
        except Exception:
            return 0.0

    def nondominated_indices(arr: np.ndarray) -> List[int]:
        N = arr.shape[0]
        is_nd = np.ones(N, dtype=bool)
        for i in range(N):
            if not is_nd[i]:
                continue
            comp = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
            if np.any(comp):
                is_nd[i] = False
            else:
                dominated_by_i = np.all(arr[i] <= arr, axis=1) & np.any(arr[i] < arr, axis=1)
                is_nd[dominated_by_i] = False
                is_nd[i] = True
        return list(np.where(is_nd)[0])

    # New parameter settings: emphasize diversity more and use a smaller shortlist fraction
    alpha = 0.55            # weight for approximate volume proxy (reduced)
    beta = 0.45             # weight for diversity term (increased)
    shortlist_ratio = 0.12  # fraction of remaining candidates to shortlist (smaller)
    min_shortlist = 10      # minimum shortlist size increased slightly
    local_search_iters = max(30, 5 * k)  # smaller, deterministic local search budget

    nd_idx = nondominated_indices(points)
    nd_points = points[nd_idx]

    remaining_idx = list(nd_idx)
    if len(remaining_idx) < k:
        dom_idx = [i for i in range(n) if i not in remaining_idx]
        ref_diff = np.maximum(reference_point - points[dom_idx], 0.0)
        raw_dom = np.prod(ref_diff, axis=1)
        sorted_dom = [dom_idx[i] for i in np.argsort(-raw_dom)]
        remaining_idx.extend(sorted_dom)

    candidates = list(remaining_idx)
    selected_indices: List[int] = []
    selected_points: List[np.ndarray] = []

    ref_diff_all = np.maximum(reference_point - points, 0.0)
    raw_volume_all = np.prod(ref_diff_all, axis=1)

    rng = __import__("random").Random(12345)

    while len(selected_indices) < k and len(candidates) > 0:
        cand_arr = points[candidates]
        raw_cand = raw_volume_all[candidates]

        if len(selected_points) == 0:
            diversity = np.full(len(candidates), np.max(raw_cand) + 1.0)
        else:
            sel_arr = np.array(selected_points)
            dif = cand_arr[:, None, :] - sel_arr[None, :, :]
            dist = np.linalg.norm(dif, axis=2)
            diversity = np.min(dist, axis=1)

        def safe_norm(x):
            mn = np.min(x)
            mx = np.max(x)
            if mx <= mn:
                return np.zeros_like(x, dtype=float)
            return (x - mn) / (mx - mn)

        norm_raw = safe_norm(raw_cand)
        norm_div = safe_norm(diversity)

        score = alpha * norm_raw + beta * norm_div

        M = max(min_shortlist, int(np.ceil(shortlist_ratio * len(candidates))))
        M = min(M, len(candidates))
        top_idx_order = np.argsort(-score)[:M]
        shortlist_indices = [candidates[i] for i in top_idx_order]

        hv_before = hv_compute(np.array(selected_points)) if len(selected_points) > 0 else 0.0
        best_contrib = -np.inf
        best_idx = None

        # evaluate exact contributions on shortlist
        for idx in shortlist_indices:
            cand_point = points[idx]
            if len(selected_points) > 0:
                stacked = np.vstack([selected_points, cand_point])
            else:
                stacked = np.array([cand_point])
            hv_after = hv_compute(stacked)
            contrib = hv_after - hv_before
            if contrib > best_contrib:
                best_contrib = contrib
                best_idx = idx

        if best_idx is None:
            best_idx = shortlist_indices[0]

        selected_indices.append(best_idx)
        selected_points.append(points[best_idx])
        if best_idx in candidates:
            candidates.remove(best_idx)

    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_indices]
        if remaining:
            order = np.argsort(-raw_volume_all[remaining])
            for i in order:
                if len(selected_indices) >= k:
                    break
                selected_indices.append(remaining[i])
                selected_points.append(points[remaining[i]])

    if len(selected_points) == k:
        current_hv = hv_compute(np.array(selected_points))
        non_selected = [i for i in range(n) if i not in selected_indices]
        if len(non_selected) > 0:
            for _ in range(local_search_iters):
                s_idx = rng.randrange(0, k)
                r_sel_global = selected_indices[s_idx]
                r_non_idx = rng.choice(non_selected)
                new_selected = selected_points.copy()
                new_selected[s_idx] = points[r_non_idx]
                new_hv = hv_compute(np.array(new_selected))
                if new_hv > current_hv + 1e-12:
                    current_hv = new_hv
                    non_selected.remove(r_non_idx)
                    non_selected.append(r_sel_global)
                    selected_indices[s_idx] = r_non_idx
                    selected_points[s_idx] = points[r_non_idx]

    subset = np.array(selected_points) if len(selected_points) > 0 else np.empty((0, d))
    if subset.shape[0] > k:
        subset = subset[:k]
    elif subset.shape[0] < k:
        remaining = [i for i in range(n) if i not in selected_indices]
        order = np.argsort(-raw_volume_all[remaining]) if remaining else []
        to_add = min(k - subset.shape[0], len(order))
        if to_add > 0:
            add_pts = points[[remaining[i] for i in order[:to_add]]]
            subset = np.vstack([subset, add_pts])
        while subset.shape[0] < k:
            if subset.shape[0] == 0:
                subset = np.zeros((1, d))
            subset = np.vstack([subset, subset[-1]])

    return subset


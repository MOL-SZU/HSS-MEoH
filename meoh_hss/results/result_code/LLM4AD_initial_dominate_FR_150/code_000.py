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

    points = np.asarray(points)
    n, d = points.shape

    # default reference
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point).reshape(-1)

    # internal hv function
    def hv_compute(objs: np.ndarray) -> float:
        if objs is None or len(objs) == 0:
            return 0.0
        try:
            hv = pg.hypervolume(np.asarray(objs))
            return float(hv.compute(reference_point))
        except Exception:
            # fallback: return zero on numerical problems
            return 0.0

    # fast nondomination filter (minimization assumed)
    def nondominated_indices(arr: np.ndarray) -> List[int]:
        N = arr.shape[0]
        is_nd = np.ones(N, dtype=bool)
        for i in range(N):
            if not is_nd[i]:
                continue
            # compare i against all j
            # j dominates i if arr[j] <= arr[i] (all dims) and any <
            comp = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
            # if any j dominates i, mark i as dominated
            if np.any(comp):
                is_nd[i] = False
            else:
                # remove others dominated by i to speed up
                dominated_by_i = np.all(arr[i] <= arr, axis=1) & np.any(arr[i] < arr, axis=1)
                is_nd[dominated_by_i] = False
                is_nd[i] = True
        return list(np.where(is_nd)[0])

    # algorithm parameters (tunable)
    alpha = 0.85            # weight for approximate volume proxy
    beta = 0.15             # weight for diversity term
    shortlist_ratio = 0.20  # fraction of remaining candidates to shortlist (min enforced)
    min_shortlist = 8
    local_search_iters = max(50, 10 * k)  # number of random swap attempts

    # Step 1: filter nondominated points first (fast, reduces candidates)
    nd_idx = nondominated_indices(points)
    nd_points = points[nd_idx]

    # If nondominated less than k, keep the rest of points as backup
    remaining_idx = list(nd_idx)
    if len(remaining_idx) < k:
        # include dominated points sorted by raw proxy volume
        dom_idx = [i for i in range(n) if i not in remaining_idx]
        # compute raw proxy for dominated points
        ref_diff = np.maximum(reference_point - points[dom_idx], 0.0)
        raw_dom = np.prod(ref_diff, axis=1)
        # sort dominated by decreasing raw_dom
        sorted_dom = [dom_idx[i] for i in np.argsort(-raw_dom)]
        remaining_idx.extend(sorted_dom)

    # We'll operate on candidate indices list
    candidates = list(remaining_idx)

    selected_indices: List[int] = []
    selected_points: List[np.ndarray] = []

    # precompute raw volume proxy for all points: product(ref - x) clipped at 0
    ref_diff_all = np.maximum(reference_point - points, 0.0)
    raw_volume_all = np.prod(ref_diff_all, axis=1)

    # main greedy loop with shortlist using approximate score
    while len(selected_indices) < k and len(candidates) > 0:
        # prepare arrays for candidates
        cand_arr = points[candidates]
        raw_cand = raw_volume_all[candidates]

        # diversity term: distance to nearest selected (if none, give large reward)
        if len(selected_points) == 0:
            diversity = np.full(len(candidates), np.max(raw_cand) + 1.0)
        else:
            sel_arr = np.array(selected_points)
            # compute Euclidean distance to nearest selected
            # shape (num_cand, num_sel)
            dif = cand_arr[:, None, :] - sel_arr[None, :, :]
            dist = np.linalg.norm(dif, axis=2)
            diversity = np.min(dist, axis=1)

        # normalize raw_cand and diversity to [0,1] before combining
        def safe_norm(x):
            mn = np.min(x)
            mx = np.max(x)
            if mx <= mn:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        norm_raw = safe_norm(raw_cand)
        norm_div = safe_norm(diversity)

        score = alpha * norm_raw + beta * norm_div

        # shortlist top M candidates by score
        M = max(min_shortlist, int(np.ceil(shortlist_ratio * len(candidates))))
        top_idx_order = np.argsort(-score)[:M]  # indices into candidates
        shortlist_indices = [candidates[i] for i in top_idx_order]

        # evaluate exact hypervolume contribution only on shortlist
        hv_before = hv_compute(np.array(selected_points)) if len(selected_points) > 0 else 0.0
        best_contrib = -np.inf
        best_idx = None
        # Evaluate contributions
        for idx in shortlist_indices:
            cand_point = points[idx]
            hv_after = hv_compute(np.vstack([selected_points, cand_point]) if len(selected_points) > 0 else np.array([cand_point]))
            contrib = hv_after - hv_before
            if contrib > best_contrib:
                best_contrib = contrib
                best_idx = idx

        # Fallback: if no best found (should not happen), pick top score candidate
        if best_idx is None:
            best_idx = shortlist_indices[0]

        # add chosen
        selected_indices.append(best_idx)
        selected_points.append(points[best_idx])
        # remove from candidates
        if best_idx in candidates:
            candidates.remove(best_idx)

    # If still fewer than k (possible if candidates exhausted), fill from remaining original points by raw volume
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_indices]
        if remaining:
            order = np.argsort(-raw_volume_all[remaining])
            for i in order:
                if len(selected_indices) >= k:
                    break
                selected_indices.append(remaining[i])
                selected_points.append(points[remaining[i]])

    # Local randomized swap-based improvement (small budget)
    if len(selected_points) == k:
        current_hv = hv_compute(np.array(selected_points))
        import random
        non_selected = [i for i in range(n) if i not in selected_indices]
        if len(non_selected) > 0:
            for _ in range(local_search_iters):
                # choose random selected and random non-selected
                s_idx = random.randrange(0, k)
                r_sel_global = selected_indices[s_idx]
                r_non_idx = random.choice(non_selected)
                # propose swap
                new_selected = selected_points.copy()
                new_selected[s_idx] = points[r_non_idx]
                new_hv = hv_compute(np.array(new_selected))
                if new_hv > current_hv + 1e-12:
                    # accept swap
                    current_hv = new_hv
                    # update indices and lists
                    non_selected.remove(r_non_idx)
                    non_selected.append(r_sel_global)
                    selected_indices[s_idx] = r_non_idx
                    selected_points[s_idx] = points[r_non_idx]

    # return exactly k rows (if less, pad by repeating last)
    subset = np.array(selected_points)
    if subset.shape[0] > k:
        subset = subset[:k]
    elif subset.shape[0] < k:
        # pad with best remaining by raw volume
        remaining = [i for i in range(n) if i not in selected_indices]
        order = np.argsort(-raw_volume_all[remaining]) if remaining else []
        to_add = min(k - subset.shape[0], len(order))
        if to_add > 0:
            add_pts = points[[remaining[i] for i in order[:to_add]]]
            subset = np.vstack([subset, add_pts])
        # if still short, repeat last row
        while subset.shape[0] < k:
            if subset.shape[0] == 0:
                subset = np.zeros((1, d))
            subset = np.vstack([subset, subset[-1]])

    return subset


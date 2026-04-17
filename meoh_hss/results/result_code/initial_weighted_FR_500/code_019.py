import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import pygmo as pg

    rng = np.random.default_rng(42)

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape
    if k <= 0:
        return np.zeros((0, d))

    # reference point default
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)
    if reference_point.shape[0] != d:
        raise ValueError("reference_point must have same dimensionality as points")

    # Pareto front filtering (minimization convention: smaller is better)
    def pareto_mask(arr: np.ndarray) -> np.ndarray:
        N = arr.shape[0]
        mask = np.ones(N, dtype=bool)
        for i in range(N):
            if not mask[i]:
                continue
            # any point that dominates i
            # j dominates i if arr[j] <= arr[i] for all dims and arr[j] < arr[i] for some dim
            comp_le = np.all(arr <= arr[i], axis=1)
            comp_lt = np.any(arr < arr[i], axis=1)
            dominated_by_some = np.any(comp_le & comp_lt)
            if dominated_by_some:
                mask[i] = False
        return mask

    if n > 1:
        try:
            pd_mask = pareto_mask(points)
        except Exception:
            # fallback: if something goes wrong, treat all as eligible
            pd_mask = np.ones(n, dtype=bool)
    else:
        pd_mask = np.ones(n, dtype=bool)

    nondom_idx = np.where(pd_mask)[0]
    nondom_pts = points[nondom_idx]

    # If too few nondominated points, we'll include dominated ones later
    # Compute cheap heuristic score: axis-aligned hyper-rectangle volume dominated by single point
    # (reference_point assumed to be dominated by worse values)
    rect_vol_all = np.prod(np.maximum(reference_point - points, 0.0), axis=1)

    # Build initial candidate pool: prioritize nondominated by rect_vol, then others by rect_vol
    order_dom = np.argsort(-rect_vol_all[nondom_idx], kind="stable")
    sorted_nondom_idx = nondom_idx[order_dom]

    remaining_idx = np.setdiff1d(np.arange(n), sorted_nondom_idx, assume_unique=True)
    order_rem = np.argsort(-rect_vol_all[remaining_idx], kind="stable")
    sorted_remaining_idx = remaining_idx[order_rem]

    ordered_all_idx = np.concatenate([sorted_nondom_idx, sorted_remaining_idx])

    # shortlist size: dependent on k but capped
    shortlist_size = int(min(max(3 * k, 50), max(50, len(ordered_all_idx))))
    shortlist_idx = ordered_all_idx[:shortlist_size].tolist()

    # If we have fewer candidates than k, extend to all
    if len(shortlist_idx) < k:
        shortlist_idx = ordered_all_idx.tolist()

    # Helper to compute hypervolume of a set of points (minimization)
    def hv_of_set(arr: np.ndarray) -> float:
        if arr is None or len(arr) == 0:
            return 0.0
        hv = pg.hypervolume(arr)
        return hv.compute(reference_point)

    # Greedy selection from shortlist with exact hypervolume contribution (HVC).
    selected_idx = []
    selected_points = []
    hv_selected = 0.0

    # Precompute mapping from index to its point for speed
    pts = points

    # Main greedy insertion: at each step compute hv(selected + cand) - hv_selected for candidates
    for _ in range(min(k, len(shortlist_idx))):
        best_contrib = -np.inf
        best_idx = None
        # Evaluate each candidate not yet selected
        for idx in shortlist_idx:
            if idx in selected_idx:
                continue
            # compute hv(selected + candidate)
            if len(selected_points) == 0:
                hv_after = hv_of_set(pts[[idx]])
            else:
                arr = np.vstack([np.array(selected_points), pts[idx]])
                hv_after = hv_of_set(arr)
            contrib = hv_after - hv_selected
            if contrib > best_contrib:
                best_contrib = contrib
                best_idx = idx
        if best_idx is None:
            break
        # accept best and update
        selected_idx.append(best_idx)
        selected_points.append(pts[best_idx])
        hv_selected = hv_of_set(np.vstack(selected_points))

    # If still fewer than k (e.g., shortlist smaller), expand selection from remaining
    if len(selected_idx) < k:
        for idx in ordered_all_idx:
            if idx in selected_idx:
                continue
            selected_idx.append(int(idx))
            selected_points.append(pts[int(idx)])
            if len(selected_idx) >= k:
                break
        hv_selected = hv_of_set(np.vstack(selected_points))

    # Local pairwise swap improvement within a bounded budget
    max_swap_iters = min(200, max(20, 10 * k))
    improved = True
    iter_count = 0
    # Precompute unselected candidates (from ordered_all_idx to allow considering dominated ones)
    candidate_pool = [int(i) for i in ordered_all_idx if int(i) not in selected_idx]

    while improved and iter_count < max_swap_iters:
        improved = False
        iter_count += 1
        # Randomize order to avoid deterministic traps but with fixed RNG seed
        sel_order = list(range(len(selected_idx)))
        rng.shuffle(sel_order)
        cand_order = candidate_pool.copy()
        rng.shuffle(cand_order)
        for si in sel_order:
            if improved:
                break
            s_idx = selected_idx[si]
            for c_idx in cand_order:
                if c_idx in selected_idx:
                    continue
                # Build swapped set: replace selected_idx[si] with c_idx
                temp_selected = selected_idx.copy()
                temp_selected[si] = c_idx
                arr = pts[np.array(temp_selected)]
                hv_temp = hv_of_set(arr)
                if hv_temp > hv_selected + 1e-12:
                    # commit swap
                    selected_idx[si] = int(c_idx)
                    selected_points[si] = pts[int(c_idx)]
                    hv_selected = hv_temp
                    # update candidate pool: add s_idx back if not present, remove c_idx
                    if s_idx not in candidate_pool:
                        candidate_pool.append(int(s_idx))
                    if c_idx in candidate_pool:
                        candidate_pool.remove(int(c_idx))
                    improved = True
                    break
        # small cap to avoid infinite loop
        if not improved:
            break

    # Ensure exactly k points: if more, trim by lowest marginal contribution; if less, pad by best remaining
    if len(selected_idx) > k:
        # compute marginal contributions and remove smallest until k
        while len(selected_idx) > k:
            best_remove_gain = np.inf
            remove_pos = None
            base_arr = pts[np.array(selected_idx)]
            base_hv = hv_of_set(base_arr)
            for i in range(len(selected_idx)):
                temp_idx = selected_idx[:i] + selected_idx[i+1:]
                hv_temp = hv_of_set(pts[np.array(temp_idx)])
                # removal loss = base_hv - hv_temp; want to remove minimal loss
                loss = base_hv - hv_temp
                if loss < best_remove_gain:
                    best_remove_gain = loss
                    remove_pos = i
            if remove_pos is None:
                break
            selected_idx.pop(remove_pos)
            selected_points.pop(remove_pos)
            hv_selected = hv_of_set(np.vstack(selected_points)) if selected_points else 0.0

    if len(selected_idx) < k:
        remaining = [int(i) for i in ordered_all_idx if int(i) not in selected_idx]
        for idx in remaining:
            if len(selected_idx) >= k:
                break
            selected_idx.append(int(idx))
            selected_points.append(pts[int(idx)])
        # If still not enough (should not happen), pad randomly
        if len(selected_idx) < k:
            leftover = [i for i in range(n) if i not in selected_idx]
            if leftover:
                add_cnt = k - len(selected_idx)
                pick = rng.choice(leftover, size=min(add_cnt, len(leftover)), replace=False)
                for p in pick:
                    selected_idx.append(int(p))
                    selected_points.append(pts[int(p)])

    # Final return exactly k rows
    final = np.array(selected_points, dtype=float)
    if final.shape[0] > k:
        final = final[:k]
    elif final.shape[0] < k:
        # pad with best by rect_vol
        missing = k - final.shape[0]
        remaining = [i for i in range(n) if i not in selected_idx]
        if remaining:
            order_pad = np.argsort(-rect_vol_all[remaining], kind="stable")
            pick = [remaining[i] for i in order_pad[:missing]]
            final = np.vstack([final, pts[pick]])
        else:
            # repeat last row as last resort
            if final.shape[0] == 0:
                final = np.tile(np.minimum(points.mean(axis=0), reference_point), (k, 1))
            else:
                repeat = np.tile(final[-1], (missing, 1))
                final = np.vstack([final, repeat])

    return np.asarray(final, dtype=float)


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import pygmo as pg
    import heapq

    rng = np.random.default_rng(123456)  # deterministic

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = pts.shape
    if k <= 0:
        return np.zeros((0, d))

    # reference point default (minimization convention)
    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)
    if reference_point.shape[0] != d:
        raise ValueError("reference_point must have same dimensionality as points")

    # Deterministic Pareto filtering (minimization)
    def pareto_mask(arr: np.ndarray) -> np.ndarray:
        N = arr.shape[0]
        if N == 0:
            return np.zeros(0, dtype=bool)
        mask = np.ones(N, dtype=bool)
        for i in range(N):
            if not mask[i]:
                continue
            # a point j dominates i if all arr[j] <= arr[i] and any arr[j] < arr[i]
            comp_le = np.all(arr <= arr[i], axis=1)
            comp_lt = np.any(arr < arr[i], axis=1)
            dominated = comp_le & comp_lt
            # if any other point dominates i, mark false
            if np.any(dominated):
                mask[i] = False
        return mask

    if n > 1:
        try:
            pd_mask = pareto_mask(pts)
        except Exception:
            pd_mask = np.ones(n, dtype=bool)
    else:
        pd_mask = np.ones(n, dtype=bool)

    nondom_idx = np.where(pd_mask)[0]
    nondom_pts = pts[nondom_idx]

    # cheap axis-aligned box volume as baseline (rectangular proxy)
    ref_minus_all = np.maximum(reference_point - pts, 0.0)
    rect_vol_all = np.prod(ref_minus_all, axis=1)

    # Stronger vectorized scalar scoring: more weight vectors and mild non-linear transform
    M = int(min(max(4, d * 3), 12))  # more weight vectors than original
    # deterministic Dirichlet weights
    weights = rng.dirichlet(np.ones(d, dtype=float), size=M)
    # scalarized values: w.dot(ref - p)
    scalarized = ref_minus_all.dot(weights.T)  # shape (n, M)
    # use log1p to temper extremes, then mean across weights
    mean_scalar = np.mean(np.log1p(scalarized), axis=1)
    # normalize
    ms_max = max(1e-12, float(np.max(mean_scalar)))
    mean_scalar_norm = mean_scalar / ms_max

    # Compose score with mild exponent mixing to favor both volume and scalarization,
    # different constants than original to create a distinct parameterization.
    eps = 1e-16
    score = (rect_vol_all + eps) ** 0.7 * (0.05 + 0.95 * (mean_scalar_norm + eps)) ** 0.3

    # Order: prioritize nondominated by score, then rest
    order_nd = np.argsort(-score[nondom_idx], kind="stable")
    sorted_nondom_idx = nondom_idx[order_nd]

    remaining_idx = np.setdiff1d(np.arange(n), sorted_nondom_idx, assume_unique=True)
    order_rem = np.argsort(-score[remaining_idx], kind="stable")
    sorted_remaining_idx = remaining_idx[order_rem]

    ordered_all_idx = np.concatenate([sorted_nondom_idx, sorted_remaining_idx])

    # Shortlist: proportional to k, slightly larger caps than original for robustness
    shortlist_size = int(min(max(4 * k, 60), max(60, len(ordered_all_idx))))
    shortlist_idx = ordered_all_idx[:shortlist_size].tolist()
    if len(shortlist_idx) < k:
        shortlist_idx = ordered_all_idx.tolist()

    # Helper HV function (minimization)
    def hv_of_set(arr: np.ndarray) -> float:
        if arr is None or len(arr) == 0:
            return 0.0
        hv = pg.hypervolume(np.asarray(arr, dtype=float))
        return float(hv.compute(reference_point))

    # Precompute singleton HVs (cached) for shortlist
    hv_single = {}
    for idx in shortlist_idx:
        hv_single[int(idx)] = hv_of_set(pts[[int(idx)]])

    # CELF-style lazy greedy using max-heap of (neg_gain, idx, timestamp)
    heap = []
    for idx in shortlist_idx:
        gain = hv_single[int(idx)]  # marginal when selected set is empty
        heapq.heappush(heap, (-gain, int(idx), 0))

    selected_idx = []
    selected_set = set()
    selected_points = []
    hv_selected = 0.0

    target = min(k, len(shortlist_idx))
    # main CELF loop
    while len(selected_idx) < target and heap:
        neg_gain, idx, ts = heapq.heappop(heap)
        if idx in selected_set:
            continue
        if ts != len(selected_idx):
            # recompute true marginal on current selected set
            if len(selected_points) == 0:
                hv_after = hv_single[int(idx)]
            else:
                arr = np.vstack([np.array(selected_points), pts[int(idx)]])
                hv_after = hv_of_set(arr)
            true_gain = hv_after - hv_selected
            heapq.heappush(heap, (-true_gain, idx, len(selected_idx)))
            continue
        # accept candidate
        selected_idx.append(int(idx))
        selected_set.add(int(idx))
        selected_points.append(pts[int(idx)])
        hv_selected = hv_of_set(np.vstack(selected_points))

    # If still fewer than k, greedily fill from ordered list deterministically
    if len(selected_idx) < k:
        for idx in ordered_all_idx:
            if int(idx) in selected_set:
                continue
            selected_idx.append(int(idx))
            selected_set.add(int(idx))
            selected_points.append(pts[int(idx)])
            if len(selected_idx) >= k:
                break
        hv_selected = hv_of_set(np.vstack(selected_points)) if selected_points else 0.0

    # Bounded deterministic 1-for-1 swaps: limited iterations to keep runtime predictable
    max_swap_iters = min(300, max(30, 15 * k))
    improved = True
    iter_count = 0
    candidate_pool = [int(i) for i in ordered_all_idx if int(i) not in selected_set]

    while improved and iter_count < max_swap_iters:
        improved = False
        iter_count += 1
        sel_order = list(range(len(selected_idx)))
        rng.shuffle(sel_order)
        cand_order = candidate_pool.copy()
        rng.shuffle(cand_order)
        for si in sel_order:
            if improved:
                break
            s_idx = selected_idx[si]
            for c_idx in cand_order:
                if c_idx in selected_set:
                    continue
                temp_selected = selected_idx.copy()
                temp_selected[si] = int(c_idx)
                arr = pts[np.array(temp_selected)]
                hv_temp = hv_of_set(arr)
                if hv_temp > hv_selected + 1e-12:
                    # commit the swap
                    removed = selected_idx[si]
                    selected_idx[si] = int(c_idx)
                    selected_set.remove(int(removed))
                    selected_set.add(int(c_idx))
                    selected_points[si] = pts[int(c_idx)]
                    hv_selected = hv_temp
                    # maintain candidate_pool deterministically
                    if int(removed) not in candidate_pool:
                        candidate_pool.append(int(removed))
                    if int(c_idx) in candidate_pool:
                        candidate_pool.remove(int(c_idx))
                    improved = True
                    break
        if not improved:
            break

    # Trim if too many
    if len(selected_idx) > k:
        while len(selected_idx) > k:
            base_arr = pts[np.array(selected_idx)]
            base_hv = hv_of_set(base_arr)
            best_remove_pos = None
            best_loss = np.inf
            for i in range(len(selected_idx)):
                temp_idx = selected_idx[:i] + selected_idx[i+1:]
                hv_temp = hv_of_set(pts[np.array(temp_idx)])
                loss = base_hv - hv_temp
                if loss < best_loss:
                    best_loss = loss
                    best_remove_pos = i
            if best_remove_pos is None:
                break
            removed = selected_idx.pop(best_remove_pos)
            selected_points.pop(best_remove_pos)
            if removed in selected_set:
                selected_set.remove(removed)
            hv_selected = hv_of_set(np.vstack(selected_points)) if selected_points else 0.0

    # Pad if too few
    if len(selected_idx) < k:
        remaining = [int(i) for i in ordered_all_idx if int(i) not in selected_set]
        for idx in remaining:
            if len(selected_idx) >= k:
                break
            selected_idx.append(int(idx))
            selected_set.add(int(idx))
            selected_points.append(pts[int(idx)])
        if len(selected_idx) < k:
            leftover = [i for i in range(n) if i not in selected_idx]
            if leftover:
                add_cnt = k - len(selected_idx)
                pick = rng.choice(leftover, size=min(add_cnt, len(leftover)), replace=False)
                for p in pick:
                    selected_idx.append(int(p))
                    selected_points.append(pts[int(p)])
                    selected_set.add(int(p))

    # Final array: ensure exactly k rows, pad deterministically if needed
    final = np.array(selected_points, dtype=float)
    if final.shape[0] > k:
        final = final[:k]
    elif final.shape[0] < k:
        missing = k - final.shape[0]
        remaining = [i for i in range(n) if i not in selected_idx]
        if remaining:
            order_pad = np.argsort(-rect_vol_all[remaining], kind="stable")
            pick = [remaining[i] for i in order_pad[:missing]]
            if final.size == 0:
                final = pts[pick]
            else:
                final = np.vstack([final, pts[pick]])
        else:
            if final.shape[0] == 0:
                final = np.tile(np.minimum(pts.mean(axis=0), reference_point), (k, 1))
            else:
                repeat = np.tile(final[-1], (missing, 1))
                final = np.vstack([final, repeat])

    return np.asarray(final, dtype=float)


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    HSS (Hypervolume-based Subset Selection): 贪心选择超体积贡献最大的 k 个点

    参数:
        points: np.ndarray
            所有候选点，形状为 (N, D)，其中 N 是点数，D 是目标维度
        k: int
            需要选择的点数
        reference_point: np.ndarray, optional
            参考点，形状为 (D,)。如果为 None，则使用 points 的最大值 * 1.1

    返回:
        subset: np.ndarray
            选出的子集，形状为 (k, D)
    """
    import numpy as np
    import pygmo as pg
    import heapq
    import time
    import math

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D).")
    n, d = points.shape
    if not (isinstance(k, int) and 0 < k <= n):
        raise ValueError("k must be an integer with 0 < k <= number of points")

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,) matching points' dimension")

    def hv_of_indices(indices):
        if len(indices) == 0:
            return 0.0
        try:
            arr = points[np.array(indices, dtype=int)]
            return float(pg.hypervolume(arr).compute(reference_point))
        except Exception:
            return 0.0

    # Precompute individual hypervolumes robustly
    individual_hv = np.zeros(n, dtype=float)
    for i in range(n):
        try:
            individual_hv[i] = float(pg.hypervolume(points[[i]]).compute(reference_point))
        except Exception:
            individual_hv[i] = 0.0

    # trivial
    if k == n:
        return points.copy()

    start_time = time.time()
    time_limit = 5.0  # seconds for overall procedure

    # CELF-like lazy greedy with tiny random jitter to break ties and encourage diversity
    # Heap entries: (-cached_gain, idx, last_updated_iter)
    heap = []
    rng = np.random.default_rng(seed=12345)
    # Add tiny jitter based on magnitude to break ties (keeps determinism)
    jitter = rng.random(n) * 1e-12
    for i in range(n):
        # initial gain is individual hv plus tiny jitter
        heap.append((-(individual_hv[i] + jitter[i]), int(i), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_hv = 0.0
    iter_id = 1  # increment each time we accept a new point
    max_pops = max(2000, n * k * 3)
    pops = 0

    while len(selected) < k and (time.time() - start_time) < time_limit:
        if not heap:
            break
        pops += 1
        if pops > max_pops:
            break
        neg_gain, idx, last_upd = heapq.heappop(heap)
        cached_gain = -neg_gain
        if idx in selected_set:
            continue
        if last_upd == iter_id:
            # accept
            selected.append(int(idx))
            selected_set.add(int(idx))
            current_hv = hv_of_indices(selected)
            iter_id += 1
            continue
        else:
            # recompute true marginal gain
            new_hv = hv_of_indices(list(selected) + [int(idx)])
            true_gain = new_hv - current_hv
            # push back updated with current iter_id mark
            heapq.heappush(heap, (-(true_gain), int(idx), iter_id))
            continue

    # fill up if not enough due to timeout or emptiness
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: individual_hv[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))
        current_hv = hv_of_indices(selected)

    # Compute per-selected-point marginal contribution (contribution = hv(S) - hv(S \ {p}))
    def contributions_for_selected(sel_indices):
        contribs = []
        base_hv = hv_of_indices(sel_indices)
        for pos, idx in enumerate(sel_indices):
            reduced = list(sel_indices[:pos]) + list(sel_indices[pos+1:])
            hv_reduced = hv_of_indices(reduced)
            contrib = base_hv - hv_reduced
            contribs.append(float(contrib))
        return np.array(contribs, dtype=float)

    # Local search: attempt to replace the worst contributors with top outsiders
    max_swaps = max(10, 5 * k)
    swaps_done = 0
    tol = 1e-12

    outsiders = [i for i in range(n) if i not in selected_set]
    outsiders_sorted = sorted(outsiders, key=lambda x: individual_hv[x], reverse=True)
    # candidate pool size controlled by k
    pool_size = min(len(outsiders_sorted), max(50, 10 * k))
    candidate_pool = outsiders_sorted[:pool_size]

    # Precompute contributions
    contribs = contributions_for_selected(selected)

    improvement = True
    while improvement and swaps_done < max_swaps and (time.time() - start_time) < time_limit:
        improvement = False
        # identify worst contributors (indices in selected list)
        worst_order = np.argsort(contribs)  # ascending: worst first
        # we'll try a small number of worst positions
        try_worst = min(len(selected), max(1, int(math.ceil(0.3 * len(selected)))))
        for wi in range(try_worst):
            sel_pos = int(worst_order[wi])
            sel_idx = selected[sel_pos]
            best_local_gain = 0.0
            best_cand = None
            # try candidates in order of individual hv
            for cand in candidate_pool:
                if cand in selected_set:
                    continue
                # quick filter: if individual_hv[cand] + 1e-12 < contribs[sel_pos], skip (unlikely to improve)
                # but since hypervolume interactions exist, still attempt top candidates
                new_sel = list(selected)
                new_sel[sel_pos] = int(cand)
                hv_new = hv_of_indices(new_sel)
                gain = hv_new - current_hv
                if gain > best_local_gain + tol:
                    best_local_gain = gain
                    best_cand = int(cand)
                    # break early if significant
                    if best_local_gain >= 1e-6:
                        break
            if best_cand is not None and best_local_gain > tol:
                # apply swap
                old_idx = selected[sel_pos]
                selected[sel_pos] = int(best_cand)
                selected_set.remove(int(old_idx))
                selected_set.add(int(best_cand))
                current_hv += best_local_gain
                swaps_done += 1
                improvement = True
                # update candidate pool to remove used candidate and possibly add next outsider
                if best_cand in candidate_pool:
                    candidate_pool = [c for c in candidate_pool if c != best_cand]
                    # try to append next best outsider if available
                    next_out_idx = pool_size
                    if next_out_idx < len(outsiders_sorted):
                        candidate_pool.append(outsiders_sorted[next_out_idx])
                        pool_size += 1
                # recompute contributions and break to restart from worst contributors
                contribs = contributions_for_selected(selected)
                break

    # Backward elimination: if any selected point has near-zero contribution, remove and fill by best outsider
    contribs = contributions_for_selected(selected)
    removed_any = True
    fill_attempts = 0
    while removed_any and (time.time() - start_time) < time_limit and fill_attempts < n:
        removed_any = False
        for pos, c in enumerate(list(contribs)):
            if c <= 1e-14:
                # remove this point
                old_idx = selected[pos]
                selected.pop(pos)
                selected_set.remove(int(old_idx))
                # pick best outsider to fill
                outsiders = [i for i in range(n) if i not in selected_set]
                if not outsiders:
                    break
                best_out = max(outsiders, key=lambda x: individual_hv[x])
                selected.insert(pos, int(best_out))
                selected_set.add(int(best_out))
                current_hv = hv_of_indices(selected)
                contribs = contributions_for_selected(selected)
                removed_any = True
                fill_attempts += 1
                break
        if not removed_any:
            break

    # Final safeguard: ensure exactly k
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: individual_hv[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))
    elif len(selected) > k:
        # drop the smallest contributors to match k
        contribs = contributions_for_selected(selected)
        drop_order = np.argsort(contribs)  # drop worst contributors first
        to_drop = len(selected) - k
        drop_positions = set(int(p) for p in drop_order[:to_drop])
        new_selected = [s for pos, s in enumerate(selected) if pos not in drop_positions]
        selected = new_selected
        selected_set = set(selected)
        current_hv = hv_of_indices(selected)

    selected = selected[:k]
    subset = points[np.array(selected, dtype=int)]
    return subset


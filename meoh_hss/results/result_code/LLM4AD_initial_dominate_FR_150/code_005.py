import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    from typing import List, Tuple
    import random

    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations") from e

    # Basic checks
    if points is None:
        return np.empty((0, 0))
    points = np.asarray(points, dtype=float)
    if points.size == 0 or k <= 0:
        return np.empty((0, 0))
    n, d = points.shape

    # reference point default
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)

    # hypervolume wrapper (safe)
    def hv_compute(objs: np.ndarray) -> float:
        if objs is None or len(objs) == 0:
            return 0.0
        try:
            hv = pg.hypervolume(np.asarray(objs))
            return float(hv.compute(reference_point))
        except Exception:
            return 0.0

    # fast nondominated filter (minimization assumed)
    def nondominated_indices(arr: np.ndarray) -> List[int]:
        N = arr.shape[0]
        is_nd = np.ones(N, dtype=bool)
        for i in range(N):
            if not is_nd[i]:
                continue
            # compare i with all j>i for speed/stability
            comp = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
            if np.any(comp):
                is_nd[i] = False
            else:
                dominated_by_i = np.all(arr[i] <= arr, axis=1) & np.any(arr[i] < arr, axis=1)
                is_nd[dominated_by_i] = False
                is_nd[i] = True
        return list(np.where(is_nd)[0])

    # Precompute cheap proxy: "box volume" from reference (product of differences)
    eps = 1e-12
    ref_diff_all = np.maximum(reference_point - points, 0.0)
    raw_volume_all = np.prod(ref_diff_all + eps, axis=1)

    # Step 1: Pareto filter
    nd_idx = nondominated_indices(points)
    remaining_idx = list(nd_idx)

    # If nondominated less than k, add dominated by box-volume
    if len(remaining_idx) < k:
        dom_idx = [i for i in range(n) if i not in remaining_idx]
        if dom_idx:
            ref_diff_dom = np.maximum(reference_point - points[dom_idx], 0.0)
            raw_dom = np.prod(ref_diff_dom + eps, axis=1)
            order_dom = np.argsort(-raw_dom)
            for idx in order_dom:
                remaining_idx.append(dom_idx[int(idx)])
                if len(remaining_idx) >= k:
                    break

    # Candidate pool is the nondominated (possibly extended)
    candidates = list(dict.fromkeys(remaining_idx))  # keep order, unique

    # If very small candidate pool, just fallback to simple greedy
    if len(candidates) <= 0:
        return np.empty((0, d))

    # Shortlist formation using cheap proxy (volume + diversity)
    # Cap shortlist size to keep HV calls low: at most max(8*k, 150) or len(candidates)
    cap = max(8 * max(1, k), 150)
    shortlist_size = min(len(candidates), cap)

    # Diversity proxy: pairwise distances to a small set of extremes (corners)
    # Use centroid of candidate set as cheap diversity baseline
    cand_arr_all = points[candidates]
    centroid = np.mean(cand_arr_all, axis=0)
    # diversity measure: distance to centroid (fast)
    diversity_proxy = np.linalg.norm(cand_arr_all - centroid[None, :], axis=1)

    # normalize proxies
    def safe_norm_vec(x: np.ndarray) -> np.ndarray:
        mn = np.min(x)
        mx = np.max(x)
        if mx <= mn:
            return np.zeros_like(x, dtype=float)
        return (x - mn) / (mx - mn)

    norm_raw = safe_norm_vec(raw_volume_all[candidates])
    norm_div = safe_norm_vec(diversity_proxy)

    alpha = 0.65  # heavier weight to raw-volume initially for shortlist
    beta = 0.35

    proxy_score = alpha * norm_raw + beta * norm_div
    order_proxy = np.argsort(-proxy_score)[:shortlist_size]
    shortlist = [candidates[int(i)] for i in order_proxy]

    # If shortlist smaller than k (rare), enlarge to candidates
    if len(shortlist) < k:
        shortlist = candidates[:min(len(candidates), max(k, shortlist_size))]

    # CELF-style lazy greedy over shortlist to reduce HV evaluations
    # Each heap element: (-marginal_gain, candidate_idx, last_seen_selected_count)
    # Initialize by computing exact marginal for each shortlist candidate wrt empty set
    heap: List[Tuple[float, int, int]] = []
    selected_indices: List[int] = []
    selected_points: List[np.ndarray] = []

    # current hv of selected set
    hv_current = 0.0

    # Precompute initial marginal gains (adding single point)
    for c in shortlist:
        hv_single = hv_compute(np.array([points[c]]))
        gain = hv_single - hv_current
        # store negative gain for max-heap using heapq
        heap.append((-gain, int(c), 0))
    heapq.heapify(heap)

    # CELF loop
    selected_count = 0
    # limit total HV evaluations: upper bound = shortlist_size * k (but CELF reduces)
    max_evals = shortlist_size * min(k, 50)  # cap to avoid runaway for huge k; heuristics
    evals = 0

    while selected_count < k and heap and evals <= max_evals:
        neg_gain, cand_idx, seen_at = heapq.heappop(heap)
        # if this candidate was last evaluated at current selected_count, accept
        if seen_at == selected_count:
            # accept candidate
            selected_indices.append(int(cand_idx))
            selected_points.append(points[int(cand_idx)])
            # update hv_current incrementally
            hv_current = hv_compute(np.vstack(selected_points)) if len(selected_points) > 0 else 0.0
            selected_count += 1
            # remove this candidate from shortlist tracking (heap already popped)
            # and continue
            continue
        else:
            # recompute true marginal gain wrt current selected set
            # compute hv of selected + cand
            if len(selected_points) == 0:
                hv_after = hv_compute(np.array([points[int(cand_idx)]]))
            else:
                hv_after = hv_compute(np.vstack([selected_points, points[int(cand_idx)]]))
            gain = hv_after - hv_current
            evals += 1
            # push back with updated timestamp
            heapq.heappush(heap, (-gain, int(cand_idx), selected_count))

    # If we didn't reach k because of eval cap or heap exhaustion, fill greedily by proxy/unselected
    if len(selected_indices) < k:
        # fill from remaining (shortlist first then candidates then all points) by raw_volume
        picked = set(selected_indices)
        remaining_pool = [i for i in shortlist if i not in picked]
        if len(remaining_pool) < (k - len(selected_indices)):
            remaining_pool.extend([i for i in candidates if i not in picked and i not in remaining_pool])
        if len(remaining_pool) < (k - len(selected_indices)):
            remaining_pool.extend([i for i in range(n) if i not in picked and i not in remaining_pool])
        # sort by raw_volume proxy descending
        remaining_pool_sorted = sorted(remaining_pool, key=lambda idx: -raw_volume_all[idx])
        for idx in remaining_pool_sorted:
            if len(selected_indices) >= k:
                break
            selected_indices.append(int(idx))
            selected_points.append(points[int(idx)])
            hv_current = hv_compute(np.vstack(selected_points)) if len(selected_points) > 0 else 0.0

    # Ensure exactly k points: pad if necessary by repeating last best
    subset = np.array(selected_points, dtype=float) if len(selected_points) > 0 else np.empty((0, d))
    if subset.shape[0] > k:
        subset = subset[:k]
    elif subset.shape[0] < k:
        need = k - subset.shape[0]
        # choose best remaining by raw_volume
        remaining = [i for i in range(n) if int(i) not in selected_indices]
        if remaining:
            rem_order = np.argsort(-raw_volume_all[remaining])
            add_idx = [remaining[int(i)] for i in rem_order[:need]]
            if subset.size == 0:
                subset = points[add_idx].copy()
            else:
                subset = np.vstack([subset, points[add_idx]])
        # if still short, pad by repeating last point
        while subset.shape[0] < k:
            if subset.shape[0] == 0:
                subset = np.zeros((1, d))
            subset = np.vstack([subset, subset[-1]])

    # Small bounded first-improvement swap local search on small pool (shortlist) to refine
    try:
        rng = random.Random(42)
        non_selected = [i for i in shortlist if i not in selected_indices]
        max_swaps = min(100, 10 * max(1, k))
        swaps = 0
        improved = True
        while swaps < max_swaps and improved and len(non_selected) > 0:
            improved = False
            swaps += 1
            # try replacing any selected with any non-selected (first improvement)
            for si in range(len(selected_indices)):
                if improved or swaps >= max_swaps:
                    break
                for nj in list(non_selected):
                    # form candidate replacement
                    new_sel_pts = selected_points.copy()
                    new_sel_pts[si] = points[int(nj)]
                    new_hv = hv_compute(np.vstack(new_sel_pts))
                    if new_hv > hv_current + 1e-12:
                        # accept swap
                        old_idx = selected_indices[si]
                        selected_indices[si] = int(nj)
                        selected_points[si] = points[int(nj)]
                        hv_current = new_hv
                        non_selected.remove(nj)
                        non_selected.append(int(old_idx))
                        improved = True
                        break
            # end inner loops
    except Exception:
        pass

    # Final subset ensure shape and exactly k rows
    subset = np.array(selected_points, dtype=float) if len(selected_points) > 0 else np.empty((0, d))
    if subset.shape[0] > k:
        subset = subset[:k]
    elif subset.shape[0] < k:
        need = k - subset.shape[0]
        remaining = [i for i in range(n) if int(i) not in selected_indices]
        if remaining:
            rem_order = np.argsort(-raw_volume_all[remaining])
            add_idx = [remaining[int(i)] for i in rem_order[:need]]
            if subset.size == 0:
                subset = points[add_idx].copy()
            else:
                subset = np.vstack([subset, points[add_idx]])
        while subset.shape[0] < k:
            if subset.shape[0] == 0:
                subset = np.zeros((1, d))
            subset = np.vstack([subset, subset[-1]])

    return subset[:k]


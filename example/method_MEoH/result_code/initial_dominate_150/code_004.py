import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    from typing import List, Tuple

    # Basic checks and conversions
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D).")
    n, d = points.shape
    if k <= 0 or k > n:
        raise ValueError("k must be > 0 and <= number of points")

    # compute reference point if not provided
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.array(reference_point)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,) matching points' dimension")

    # helper: compute hypervolume of given list/array of points using pygmo
    def HV_cal(objectives_list: List[np.ndarray]) -> float:
        try:
            import pygmo as pg
        except Exception as e:
            raise ImportError("pygmo is required for hypervolume computations. "
                              "Please install pygmo (pip install pygmo) or provide it in your environment.") from e

        if len(objectives_list) == 0:
            return 0.0
        arr = np.array(objectives_list)
        try:
            hv = pg.hypervolume(arr)
            return float(hv.compute(reference_point))
        except Exception:
            # If hypervolume calculation fails for some reason, return 0.0 as a safe fallback
            return 0.0

    # ------------------------
    # Phase 1: Lazy Greedy
    # ------------------------
    selected_indices: List[int] = []
    selected_points: List[np.ndarray] = []
    current_hv = 0.0

    # priority queue: entries are (-contribution, idx, last_update_size)
    # last_update_size records len(selected_points) when the contribution was computed
    heap: List[Tuple[float, int, int]] = []

    # initialize contributions w.r.t. empty selected set (hv of each point alone)
    for idx in range(n):
        pt = points[idx]
        contrib = HV_cal([pt])  # hv([pt]) - 0
        # numeric safety
        if contrib < 0 and contrib > -1e-12:
            contrib = 0.0
        heap.append((-contrib, idx, 0))
    heapq.heapify(heap)

    selected_flag = [False] * n

    while len(selected_indices) < k and heap:
        neg_contrib, idx, last_update = heapq.heappop(heap)
        if selected_flag[idx]:
            continue

        # if the contribution is fresh w.r.t. current selected set size, accept it
        if last_update == len(selected_points):
            contrib = -neg_contrib
            if contrib < 0 and contrib > -1e-12:
                contrib = 0.0
            selected_indices.append(idx)
            selected_points.append(points[idx])
            selected_flag[idx] = True
            current_hv = HV_cal(selected_points)
            continue
        else:
            # stale contribution: recompute relative to current selected set
            hv_before = current_hv if len(selected_points) > 0 else 0.0
            hv_after = HV_cal(selected_points + [points[idx]])
            new_contrib = hv_after - hv_before
            if new_contrib < 0 and new_contrib > -1e-12:
                new_contrib = 0.0
            # push back with updated timestamp (len(selected_points))
            heapq.heappush(heap, (-new_contrib, idx, len(selected_points)))
            # loop continues to pop next candidate
            continue

    # safety fill if something went wrong
    if len(selected_indices) < k:
        for idx in range(n):
            if not selected_flag[idx]:
                selected_indices.append(idx)
                selected_points.append(points[idx])
                selected_flag[idx] = True
                if len(selected_indices) == k:
                    break
        current_hv = HV_cal(selected_points)

    # ------------------------
    # Phase 2: Bounded Local Search (best improving swaps)
    # ------------------------
    # To keep runtime comparable, we restrict the number of hypervolume evaluations:
    # allow at most max_hv_evals_ls additional HV evaluations during local search.
    # Default cap: 10 * k (tunable). This ensures local search is bounded and lightweight.
    max_hv_evals_ls = max(10 * k, k)  # at least k, typically 10*k
    hv_evals = 0
    improved = True
    tol = 1e-12  # improvement tolerance

    selected_set = set(selected_indices)
    remaining_set = set(range(n)) - selected_set

    # We perform passes trying to find a single best improving swap per pass (best improvement),
    # stopping when no improvement is found or hv evaluation budget is exhausted.
    while improved and hv_evals < max_hv_evals_ls:
        improved = False
        best_improvement = 0.0
        best_swap = None  # tuple (s_idx_in_selected_list_index, r_idx_global)

        # Precompute current selected array for quick replacement
        curr_sel_arr = np.array(selected_points)
        curr_hv = current_hv

        # For each selected point s (index within selected_indices), try swapping with some remaining r.
        # We iterate over selected first (size k) and remaining second (size n-k) but stop as soon as
        # budget is exhausted.
        for i_s, s_global_idx in enumerate(list(selected_indices)):
            if hv_evals >= max_hv_evals_ls:
                break
            for r_global_idx in list(remaining_set):
                if hv_evals >= max_hv_evals_ls:
                    break
                # construct swapped set: replace element at position i_s with candidate r_global_idx
                swapped = curr_sel_arr.copy()
                swapped[i_s] = points[r_global_idx]
                new_hv = HV_cal(swapped.tolist())
                hv_evals += 1
                gain = new_hv - curr_hv
                if gain > best_improvement + tol:
                    best_improvement = gain
                    best_swap = (i_s, s_global_idx, r_global_idx, new_hv)
                # small micro-optimization: if we found a very large improvement we can prefer it,
                # but we still continue until budget or all pairs checked to find best among considered.
            # end for remaining
        # end for selected

        if best_swap is not None and best_improvement > tol:
            # perform the best swap found
            i_s, s_global_idx, r_global_idx, new_hv_val = best_swap
            # update selected indices and points
            selected_indices[i_s] = r_global_idx
            selected_points[i_s] = points[r_global_idx]
            selected_set.remove(s_global_idx)
            selected_set.add(r_global_idx)
            remaining_set.remove(r_global_idx)
            remaining_set.add(s_global_idx)
            current_hv = float(new_hv_val)
            improved = True
            # continue another pass (unless budget exhausted)
        else:
            # no improving swap found or budget exhausted
            break

    # Final selected points as numpy array
    final_selected = np.array(selected_points)
    # ensure returned shape is (k, d)
    if final_selected.shape[0] != k:
        # If for some reason size mismatch, pad/truncate deterministically
        sel = []
        sel_flags = [False] * n
        for idx in selected_indices:
            if not sel_flags[idx]:
                sel.append(points[idx])
                sel_flags[idx] = True
            if len(sel) == k:
                break
        # if still short, append from remaining
        if len(sel) < k:
            for idx in range(n):
                if not sel_flags[idx]:
                    sel.append(points[idx])
                    sel_flags[idx] = True
                if len(sel) == k:
                    break
        final_selected = np.array(sel)

    return final_selected


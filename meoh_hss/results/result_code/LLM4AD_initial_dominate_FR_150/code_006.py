import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    from typing import List, Tuple
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations") from e

    # Basic validation
    if points is None:
        return np.empty((0, 0))

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape
    if k <= 0:
        return np.empty((0, d))
    if k > n:
        # We will pad or repeat later; allow k>n but warn via behavior
        pass

    # reference point default
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point).reshape(-1)
    if reference_point.shape[0] != d:
        raise ValueError("reference_point must have same dimension as points")

    # hypervolume compute wrapper
    def hv_compute(objs: np.ndarray) -> float:
        if objs is None or len(objs) == 0:
            return 0.0
        try:
            arr = np.asarray(objs)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            hv = pg.hypervolume(arr)
            return float(hv.compute(reference_point))
        except Exception:
            # numerical issues -> fallback zero
            return 0.0

    # Fast nondominated filter (minimization assumed)
    def nondominated_indices(arr: np.ndarray) -> List[int]:
        N = arr.shape[0]
        if N == 0:
            return []
        is_nd = np.ones(N, dtype=bool)
        for i in range(N):
            if not is_nd[i]:
                continue
            # j dominates i if arr[j] <= arr[i] (all dims) and any <
            le = np.all(arr <= arr[i], axis=1)
            lt = np.any(arr < arr[i], axis=1)
            dominated_by_some = np.where(le & lt)[0]
            if dominated_by_some.size > 0:
                is_nd[i] = False
                continue
            # remove those dominated by i
            dominated = np.where(np.all(arr[i] <= arr, axis=1) & np.any(arr[i] < arr, axis=1))[0]
            is_nd[dominated] = False
            is_nd[i] = True
        return list(np.where(is_nd)[0])

    # Precompute box (axis-aligned) volume proxy: product(max(0, ref - x))
    ref_diff_all = np.maximum(reference_point - points, 0.0)
    box_vol_all = np.prod(ref_diff_all, axis=1)

    # Step 1: filter nondominated points (keeps variety and reduces candidates)
    nd_idx = nondominated_indices(points)
    candidates = list(nd_idx)

    # If nondominated fewer than k, include dominated points sorted by box volume
    if len(candidates) < k:
        dominated = [i for i in range(n) if i not in candidates]
        # sort dominated by decreasing box volume
        if len(dominated) > 0:
            order = np.argsort(-box_vol_all[dominated])
            candidates.extend([dominated[i] for i in order])

    # If still empty, fallback to all points
    if len(candidates) == 0:
        candidates = list(range(n))

    # Lazy greedy (CELF-style) using box volume as initial upper bound.
    # Heap entries: (-key, idx, last_eval_size, last_gain)
    # last_eval_size: size of selected set when last_gain was computed; -1 means not yet exact
    heap: List[Tuple[float, int, int, float]] = []
    for idx in candidates:
        # Use box volume as initial surrogate key (larger is better)
        key = float(box_vol_all[idx])
        heap.append((-key, idx, -1, 0.0))
    heapq.heapify(heap)

    selected_indices: List[int] = []
    selected_points: List[np.ndarray] = []
    current_hv = 0.0

    # To avoid re-evaluating identical points, maintain a set of remaining candidate indices
    candidate_set = set(candidates)

    # Main selection loop
    while len(selected_indices) < k and heap:
        neg_key, idx, last_size, last_gain = heapq.heappop(heap)
        if idx not in candidate_set:
            continue  # already selected or removed
        # If last evaluated against current selection size, accept using last_gain (exact)
        if last_size == len(selected_indices) and last_size != -1:
            # accept idx
            selected_indices.append(idx)
            selected_points.append(points[idx])
            candidate_set.remove(idx)
            current_hv = current_hv + last_gain  # last_gain = hv(new) - hv(current_at_eval_time)
            # After accepting, we continue to next selection
            continue

        # Otherwise compute exact marginal gain w.r.t current selected set
        if len(selected_points) == 0:
            hv_before = 0.0
            hv_after = hv_compute(points[idx])
        else:
            hv_before = current_hv
            arr = np.vstack([np.asarray(selected_points), points[idx]])
            hv_after = hv_compute(arr)
        gain = hv_after - hv_before
        # Push back with updated exact gain and timestamp
        heapq.heappush(heap, (-gain, idx, len(selected_indices), gain))
        # Loop continues; the popped element will eventually be at top and accepted when up-to-date

    # If we ran out of heap but still need points (shouldn't happen often), fill by best remaining box volume
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_indices]
        if remaining:
            order = np.argsort(-box_vol_all[remaining])
            for i in order:
                if len(selected_indices) >= k:
                    break
                selected_indices.append(remaining[i])
                selected_points.append(points[remaining[i]])
    # Final padding if still short (repeat last chosen or zeros)
    subset = np.array(selected_points)
    if subset.shape[0] > k:
        subset = subset[:k]
    elif subset.shape[0] < k:
        to_add = k - subset.shape[0]
        if subset.shape[0] == 0:
            # no selection possible: return top k by box volume (or zeros if no points)
            if n == 0:
                return np.zeros((k, d))
            order = np.argsort(-box_vol_all)
            pick = [order[i % n] for i in range(k)]
            subset = points[pick]
        else:
            # add best remaining by box volume
            remaining = [i for i in range(n) if i not in selected_indices]
            if remaining:
                order = np.argsort(-box_vol_all[remaining])
                take = min(len(order), to_add)
                add_pts = points[[remaining[i] for i in order[:take]]]
                subset = np.vstack([subset, add_pts])
            # if still short, repeat last row
            while subset.shape[0] < k:
                subset = np.vstack([subset, subset[-1]])
    return subset


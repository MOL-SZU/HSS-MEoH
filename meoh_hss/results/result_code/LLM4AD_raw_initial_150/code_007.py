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
    import heapq

    try:
        import pygmo as pg
    except Exception:
        pg = None

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape

    # trivial cases
    if k <= 0:
        return np.zeros((0, d), dtype=float)
    if k >= n:
        return points.copy()

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,) matching points' dimension")

    # hypervolume computation helper (exact via pygmo if available, otherwise fallback approx)
    def hv_of_indices(idx_list):
        if len(idx_list) == 0:
            return 0.0
        arr = points[np.array(idx_list, dtype=int)]
        if pg is not None:
            try:
                # pygmo hypervolume expects a 2D array of points (objectives), compute() takes reference
                return float(pg.hypervolume(arr).compute(reference_point))
            except Exception:
                pass
        # fallback approximate union-of-boxes (fast but not exact)
        diffs = np.maximum(reference_point - arr, 0.0)
        try:
            if arr.shape[0] == 1:
                return float(np.prod(diffs[0]))
            single = np.prod(diffs, axis=1)
            approx = float(single.sum())
            m = arr.shape[0]
            if m > 1:
                max_pairs = 200000
                est_pairs = m * (m - 1) // 2
                if est_pairs > max_pairs:
                    rng = np.random.default_rng(0)
                    for _ in range(max_pairs):
                        i = int(rng.integers(0, m))
                        j = int(rng.integers(0, m - 1))
                        if j >= i:
                            j += 1
                        mins = np.minimum(diffs[i], diffs[j])
                        approx -= 0.5 * float(np.prod(mins))
                else:
                    for i in range(m):
                        for j in range(i + 1, m):
                            mins = np.minimum(diffs[i], diffs[j])
                            approx -= 0.5 * float(np.prod(mins))
            return max(0.0, approx)
        except Exception:
            return 0.0

    # Precompute single-point hypervolumes (upper bounds / initial marginal estimates)
    single_hvs = np.zeros(n, dtype=float)
    for i in range(n):
        single_hvs[i] = hv_of_indices([i])

    # Build max-heap of candidates with initial marginal = single_hv
    # Heap entries: (-marginal, idx, last_selected_count) where last_selected_count indicates
    # the selection size when marginal was computed
    heap = []
    for i in range(n):
        # last_selected_count = 0 (no points selected yet)
        heap.append((-single_hvs[i], int(i), 0))
    heapq.heapify(heap)

    selected_indices = []
    selected_set = set()
    current_hv = 0.0
    tol = 1e-12
    selected_count = 0

    # Lazy greedy (CELF-like): pop top, if its stored marginal was computed for current selected_count,
    # accept it; otherwise recompute actual marginal and push back with updated timestamp.
    while selected_count < k and heap:
        neg_marginal, idx, last_count = heapq.heappop(heap)
        if idx in selected_set:
            continue  # already selected via some path (shouldn't normally happen)
        stored_marginal = -neg_marginal
        if last_count == selected_count:
            # marginal valid -> select if positive
            if stored_marginal <= tol:
                break  # no useful improvement remains
            selected_indices.append(int(idx))
            selected_set.add(int(idx))
            selected_count += 1
            current_hv = hv_of_indices(selected_indices)  # exact update
            # continue selecting next
        else:
            # need to recompute true marginal w.r.t current selection
            new_hv = hv_of_indices(selected_indices + [int(idx)])
            new_marginal = max(0.0, new_hv - current_hv)
            # push back with updated timestamp
            heapq.heappush(heap, (-new_marginal, int(idx), selected_count))

    # If we stopped early (no positive marginals) or heap exhausted, fill remaining by top single_hvs
    if len(selected_indices) < k:
        # sort indices by single_hvs descending and pick unselected
        order = np.argsort(-single_hvs)
        for idx in order:
            if idx in selected_set:
                continue
            selected_indices.append(int(idx))
            selected_set.add(int(idx))
            if len(selected_indices) >= k:
                break
        # update current hv (best effort)
        current_hv = hv_of_indices(selected_indices)

    subset = points[np.array(selected_indices[:k], dtype=int), :].astype(float, copy=True)
    return subset


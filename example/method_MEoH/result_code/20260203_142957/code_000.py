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
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape
    if k <= 0:
        return np.zeros((0, D), dtype=points.dtype)
    if k >= N:
        return points.copy()

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Ensure boxes are anchored from reference to points: clip points below reference to reference
    pts_clipped = np.maximum(points, reference_point)
    pts_shift = pts_clipped - reference_point  # non-negative coordinates

    # Helper: filter nondominated points (maximization sense)
    def filter_nondominated(arr):
        # arr: (m, D)
        if arr.size == 0:
            return arr
        m = arr.shape[0]
        is_nd = np.ones(m, dtype=bool)
        for i in range(m):
            if not is_nd[i]:
                continue
            ai = arr[i]
            # j dominates i if all arr[j] >= ai and any > 
            # We'll compare with all j != i
            ge = np.all(arr >= ai, axis=1)
            gt = np.any(arr > ai, axis=1)
            dominates = ge & gt
            dominates[i] = False
            # if any other dominates i, mark i as dominated
            if np.any(dominates):
                is_nd[i] = False
            else:
                # eliminate any points dominated by i to reduce future checks
                le = np.all(arr <= ai, axis=1)
                lt = np.any(arr < ai, axis=1)
                dominated_by_i = le & lt
                is_nd[dominated_by_i] = False
                is_nd[i] = True
        return arr[is_nd]

    # Recursive WFG-like hypervolume for union of boxes anchored at origin (reference shifted)
    def hypervolume(arr, dim):
        # arr: (m, D) with non-negative coordinates, compute hypervolume in first dim dims
        if arr.size == 0:
            return 0.0
        arr = filter_nondominated(arr)
        if arr.size == 0:
            return 0.0
        if dim == 1:
            return float(np.max(arr[:, 0]))
        # sort by last coordinate (dim-1 index) ascending
        idx = np.argsort(arr[:, dim - 1])
        sorted_arr = arr[idx]
        total = 0.0
        prev = 0.0
        m = sorted_arr.shape[0]
        i = 0
        while i < m:
            z = sorted_arr[i, dim - 1]
            if z > prev:
                # consider tail with last coord >= z => indices i..m-1
                tail = sorted_arr[i:, :dim - 1]
                hv_rec = hypervolume(tail, dim - 1)
                total += hv_rec * (z - prev)
                prev = z
            # skip points with same last coordinate to next distinct
            j = i + 1
            while j < m and sorted_arr[j, dim - 1] == z:
                j += 1
            i = j
        return float(total)

    # Greedy selection
    selected_idx = []
    remaining = set(range(N))
    current_hv = 0.0
    # For acceleration: precompute single-point volumes
    single_volumes = np.prod(pts_shift, axis=1)
    # Main greedy loop
    for _ in range(min(k, N)):
        best_idx = None
        best_gain = -np.inf
        # Precompute hv of current selected set to reuse
        if selected_idx:
            hv_selected = hypervolume(pts_shift[np.array(selected_idx)], D)
            current_hv = hv_selected
        else:
            current_hv = 0.0
        for idx in list(remaining):
            # Quick upper bound: single volume of candidate plus current_hv (not accurate but can help prune)
            # For simplicity, use exact eval (but try a cheap check first)
            # cheap check: if single_volumes[idx] <= best_gain: skip (since incremental cannot exceed single volume)
            if best_gain >= 0 and single_volumes[idx] <= best_gain:
                continue
            candidate_indices = selected_idx + [idx]
            hv_with = hypervolume(pts_shift[np.array(candidate_indices)], D)
            gain = hv_with - current_hv
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining.remove(best_idx)
        current_hv += max(0.0, best_gain)

    selected_idx = selected_idx[:k]
    return points[np.array(selected_idx)]


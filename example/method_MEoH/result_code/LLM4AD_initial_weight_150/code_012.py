import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    from typing import List, Tuple
    import numpy as np
    import heapq

    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")

    n, d = points.shape
    if k <= 0:
        return np.empty((0, d))
    k = min(k, n)

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    # Helper: compute hypervolume of a list/array of objective points
    def hv_of(array_like: List[np.ndarray]) -> float:
        if not array_like:
            return 0.0
        arr = np.array(array_like)
        hv = pg.hypervolume(arr)
        return float(hv.compute(reference_point))

    # Helper: marginal contribution of a point given current selected_points
    def marginal(point: np.ndarray, selected_list: List[np.ndarray], hv_selected: float) -> float:
        if not selected_list:
            return hv_of([point])
        hv_after = hv_of(selected_list + [point])
        return hv_after - hv_selected

    # Initialize heap with individual contributions (selected set empty)
    heap: List[Tuple[float, int, int]] = []  # entries: (-contrib, idx, timestamp)
    for idx in range(n):
        single_hv = hv_of([points[idx]])
        # timestamp indicates the size of selected set when this contrib was computed (0 now)
        heap.append((-single_hv, idx, 0))
    heapq.heapify(heap)

    selected_indices: List[int] = []
    selected_points: List[np.ndarray] = []

    # Lazy greedy: on each selection, pop top; if its timestamp != current selected size,
    # recompute its true marginal and push back with updated timestamp; otherwise accept.
    while len(selected_indices) < k and heap:
        hv_selected = hv_of(selected_points)  # hypervolume of current selection
        while heap:
            neg_contrib, idx, ts = heapq.heappop(heap)
            # If timestamp matches current selected size, this contribution is up-to-date
            if ts == len(selected_indices):
                # Accept this idx
                selected_indices.append(idx)
                selected_points.append(points[idx])
                break
            else:
                # Recompute marginal contribution w.r.t. current selected set and push back
                true_contrib = marginal(points[idx], selected_points, hv_selected)
                heapq.heappush(heap, (-true_contrib, idx, len(selected_indices)))
        else:
            # heap exhausted unexpectedly
            break

    subset = np.array(selected_points)
    return subset


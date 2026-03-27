import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = pts.shape
    if k <= 0:
        return np.empty((0, d))
    if k >= n:
        return pts.copy()

    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def compute_hv(arr):
        if arr.size == 0:
            return 0.0
        hv = pg.hypervolume(arr)
        return float(hv.compute(reference_point))

    # Precompute single-point HVs for bootstrapping the heap
    single_hv = np.zeros(n, dtype=float)
    for i in range(n):
        single_hv[i] = compute_hv(pts[[i], :])

    # Max-heap using negative gains
    heap = [(-float(single_hv[i]), int(i)) for i in range(n)]
    heapq.heapify(heap)

    selected_list = []
    selected_set = set()
    current_selected = np.empty((0, d))
    current_hv = 0.0

    # Parameter L: number of top candidates to fully evaluate each iteration
    # Small constant keeps running time comparable while improving selection quality.
    L = min(max(5, int(max(5, n // (k + 1)))), n)  # adapt to problem size but not too large

    while len(selected_list) < k and heap:
        # Extract up to L distinct unselected candidates from heap
        candidates = []
        popped = []
        while len(candidates) < L and heap:
            neg_gain, idx = heapq.heappop(heap)
            if idx in selected_set:
                continue
            candidates.append(int(idx))
            popped.append((neg_gain, idx))
        if len(candidates) == 0:
            break

        # Recompute true marginal gains for these candidates
        best_idx = None
        best_gain = -np.inf
        best_hv_with = None
        candidate_results = []
        for idx in candidates:
            # compute hv of current_selected U {pts[idx]}
            if current_selected.size == 0:
                hv_with = single_hv[idx]
            else:
                stacked = np.vstack([current_selected, pts[[idx], :]])
                hv_with = compute_hv(stacked)
            marginal = hv_with - current_hv
            candidate_results.append((idx, marginal, hv_with))
            if marginal > best_gain or (marginal == best_gain and (best_idx is None or idx < best_idx)):
                best_gain = marginal
                best_idx = idx
                best_hv_with = hv_with

        # Push back non-selected candidates with updated cached gains
        for idx, marginal, hv_with in candidate_results:
            if idx == best_idx:
                continue
            # store updated cached gain (negative for max-heap)
            heapq.heappush(heap, (-float(marginal), int(idx)))

        # Accept best candidate (even if marginal is tiny/non-positive, to reach k)
        if best_idx is None:
            break
        selected_list.append(int(best_idx))
        selected_set.add(int(best_idx))
        # update current set and hv
        if current_selected.size == 0:
            current_selected = pts[[best_idx], :].copy()
        else:
            current_selected = np.vstack([current_selected, pts[[best_idx], :]])
        current_hv = float(best_hv_with)

    # If still short of k points (e.g., heap exhausted), fill by highest single_hv among remaining
    if len(selected_list) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        if remaining:
            remaining_sorted = sorted(remaining, key=lambda x: (-single_hv[x], x))
            need = k - len(selected_list)
            for idx in remaining_sorted[:need]:
                selected_list.append(int(idx))
                selected_set.add(int(idx))

    # Ensure exactly k output rows (if somehow still short, repeat last selection)
    if len(selected_list) < k:
        while len(selected_list) < k:
            selected_list.append(selected_list[-1] if selected_list else 0)

    selected_array = pts[np.array(selected_list[:k], dtype=int), :]
    return selected_array


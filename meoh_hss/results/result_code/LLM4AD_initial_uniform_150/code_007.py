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

    # Precompute single-point HVs
    single_hv = np.zeros(n, dtype=float)
    for i in range(n):
        single_hv[i] = compute_hv(pts[[i], :])
    max_single = float(single_hv.max()) if single_hv.size > 0 else 1.0
    single_norm = single_hv / (max_single + 1e-12)

    # initial cached scores: use single_hv as proxy
    heap = [(-float(single_hv[i]), int(i)) for i in range(n)]
    heapq.heapify(heap)

    selected_list = []
    selected_set = set()
    current_selected = np.empty((0, d))
    current_hv = 0.0

    # L: number of top candidates to fully evaluate each iteration
    L = int(min(max(10, int(np.sqrt(n) + 5)), n))

    iter_count = 0
    # parameters for new scoring:
    alpha0 = 0.6   # initial multiplicative emphasis on single-point quality
    tau = max(1.0, k / 2.0)  # decay factor for alpha
    beta_max = 0.4  # interaction term strength when many selected

    while len(selected_list) < k and heap:
        iter_count += 1
        sel_count = len(selected_list)
        # adaptive parameters:
        # alpha decays exponentially so single-point boost is stronger early and reduces smoothly
        alpha = alpha0 * np.exp(-float(sel_count) / float(max(1.0, tau)))
        # beta grows with selections (interaction term encourages choosing points that synergize later)
        beta = beta_max * float(sel_count) / float(max(1, k))

        # gather up to L distinct unselected candidates
        candidates = []
        popped = []
        while len(candidates) < L and heap:
            neg_score, idx = heapq.heappop(heap)
            idx = int(idx)
            if idx in selected_set:
                continue
            candidates.append(idx)
            popped.append((neg_score, idx))
        if len(candidates) == 0:
            break

        # recompute true marginal gains for these candidates and augmented score
        best_idx = None
        best_aug_score = -np.inf
        best_hv_with = None
        candidate_results = []
        for idx in candidates:
            if current_selected.size == 0:
                hv_with = single_hv[idx]
            else:
                stacked = np.vstack([current_selected, pts[[idx], :]])
                hv_with = compute_hv(stacked)
            marginal = float(hv_with - current_hv)
            # new augmented score: combination of additive and multiplicative interaction
            # aug = marginal + alpha * single + beta * marginal * single_norm
            aug_score = marginal + alpha * float(single_hv[idx]) + beta * marginal * float(single_norm[idx])
            candidate_results.append((idx, marginal, hv_with, aug_score))
            if aug_score > best_aug_score or (aug_score == best_aug_score and (best_idx is None or idx < best_idx)):
                best_aug_score = aug_score
                best_idx = idx
                best_hv_with = hv_with

        # push back non-selected candidates with updated cached augmented scores
        for idx, marginal, hv_with, aug_score in candidate_results:
            if idx == best_idx:
                continue
            # ensure score is finite
            heapq.heappush(heap, (-float(aug_score), int(idx)))

        # accept best candidate (even if marginal is small)
        if best_idx is None:
            break
        selected_list.append(int(best_idx))
        selected_set.add(int(best_idx))
        if current_selected.size == 0:
            current_selected = pts[[best_idx], :].copy()
        else:
            current_selected = np.vstack([current_selected, pts[[best_idx], :]])
        current_hv = float(best_hv_with)

    # fill remaining by highest single_hv among remaining
    if len(selected_list) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        if remaining:
            remaining_sorted = sorted(remaining, key=lambda x: (-single_hv[x], x))
            need = k - len(selected_list)
            for idx in remaining_sorted[:need]:
                selected_list.append(int(idx))
                selected_set.add(int(idx))

    # ensure exactly k (repeat last if necessary)
    if len(selected_list) < k:
        while len(selected_list) < k:
            selected_list.append(selected_list[-1] if selected_list else 0)

    selected_array = pts[np.array(selected_list[:k], dtype=int), :]
    return selected_array


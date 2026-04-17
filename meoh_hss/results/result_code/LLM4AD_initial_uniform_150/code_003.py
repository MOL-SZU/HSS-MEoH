import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for exact hypervolume computation") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k <= 0:
        return np.empty((0, D), dtype=pts.dtype)
    if N == 0:
        return np.empty((0, D), dtype=pts.dtype)
    if k >= N:
        return pts.copy()

    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Quick axis-aligned box volumes as a proxy
    delta = np.clip(reference_point - pts, a_min=0.0, a_max=None)
    with np.errstate(invalid='ignore'):
        box_vol = np.prod(delta, axis=1)
    box_vol = np.clip(box_vol, a_min=0.0, a_max=None)

    # Fast nondominated extraction among a candidate subset
    def nondominated_indices(idx_list):
        arr = pts[np.array(idx_list, dtype=int), :]
        order = np.argsort(np.sum(arr, axis=1), kind='mergesort')
        ordered = [idx_list[i] for i in order]
        kept = []
        for idx in ordered:
            p = pts[idx]
            dominated = False
            for j in kept:
                q = pts[j]
                if np.all(q <= p) and np.any(q < p):
                    dominated = True
                    break
            if dominated:
                continue
            # remove those dominated by p
            to_remove = []
            for j in kept:
                q = pts[j]
                if np.all(p <= q) and np.any(p < q):
                    to_remove.append(j)
            for tr in to_remove:
                kept.remove(tr)
            kept.append(idx)
        return kept

    # build initial candidate pool by top box volume, then pareto prune and cap
    candidate_pool_size = int(min(N, max(8 * k, 500)))
    candidate_pool_size = max(candidate_pool_size, k)
    order_by_box = np.argsort(-box_vol, kind='mergesort')
    initial_candidates = order_by_box[:candidate_pool_size].tolist()

    nd = nondominated_indices(initial_candidates)
    if len(nd) < k:
        candidates = initial_candidates
    else:
        cap = int(min(len(nd), max(6 * k, 1000)))
        candidates = nd[:cap]

    # diversity sampling via farthest-point to cap candidate set size
    def farthest_point_sampling(indices, m, seed=12345):
        if len(indices) <= m:
            return list(indices)
        P = pts[np.array(indices, dtype=int), :]
        mins = P.min(axis=0)
        maxs = P.max(axis=0)
        rngange = maxs - mins
        rngange[rngange == 0] = 1.0
        normP = (P - mins) / rngange
        chosen = []
        # start with the largest box_vol among indices
        idx0 = max(indices, key=lambda i: box_vol[i])
        chosen.append(idx0)
        chosen_mask = np.zeros(len(indices), dtype=bool)
        chosen_mask[indices.index(idx0)] = True
        dists = np.linalg.norm(normP - normP[indices.index(idx0)], axis=1)
        for _ in range(1, m):
            far_i = int(np.argmax(dists))
            chosen.append(indices[far_i])
            chosen_mask[far_i] = True
            newd = np.linalg.norm(normP - normP[far_i], axis=1)
            dists = np.minimum(dists, newd)
            # tiny break if already selected enough
        return chosen

    rng = np.random.RandomState(2026)
    cand_cap = int(min(len(candidates), max(4 * k, 600)))
    diverse_candidates = farthest_point_sampling(candidates, cand_cap, seed=2026)

    # exact hypervolume computation with caching
    hv_cache = {}

    def hv_of_indices(global_idxs):
        if not global_idxs:
            return 0.0
        key = tuple(sorted(int(i) for i in global_idxs))
        if key in hv_cache:
            return hv_cache[key]
        data = pts[np.array(key, dtype=int), :]
        hv = pg.hypervolume(data)
        val = float(hv.compute(reference_point))
        hv_cache[key] = val
        return val

    # precompute singletons for candidates
    single_hv = {}
    for idx in diverse_candidates:
        single_hv[idx] = hv_of_indices([idx])
    max_single = max(single_hv.values()) if single_hv else 1.0
    single_norm = {i: single_hv[i] / (max_single + 1e-12) for i in single_hv}

    # prepare heap: use estimated score = single_hv initially
    heap = [(-float(single_hv[i]), int(i)) for i in diverse_candidates]
    heapq.heapify(heap)

    selected_list = []
    selected_set = set()
    current_selected = []
    current_hv = 0.0

    # L: how many top candidates to fully evaluate each iteration
    L = int(min(max(12, int(np.sqrt(len(diverse_candidates)) + 5)), len(diverse_candidates)))
    alpha0 = 0.55
    tau = max(1.0, k / 2.0)
    beta_max = 0.45

    # precompute bounding box for diversity distances
    bbox_min = np.min(pts[np.array(diverse_candidates, dtype=int), :], axis=0)
    bbox_max = np.max(pts[np.array(diverse_candidates, dtype=int), :], axis=0)
    max_dist = np.linalg.norm(bbox_max - bbox_min)
    if max_dist <= 0:
        max_dist = 1.0

    while len(selected_list) < k and heap:
        sel_count = len(selected_list)
        alpha = alpha0 * np.exp(-float(sel_count) / float(max(1.0, tau)))
        beta = beta_max * float(sel_count) / float(max(1, k))

        # pop up to L distinct unselected candidates
        candidates_batch = []
        popped = []
        while len(candidates_batch) < L and heap:
            neg_est, idx = heapq.heappop(heap)
            idx = int(idx)
            if idx in selected_set:
                continue
            candidates_batch.append(idx)
            popped.append((neg_est, idx))
        if len(candidates_batch) == 0:
            break

        # evaluate true marginals for batch candidates and augmented score
        best_idx = None
        best_aug = -np.inf
        best_hv_with = None
        candidate_results = []
        for idx in candidates_batch:
            if not current_selected:
                hv_with = single_hv.get(idx, hv_of_indices([idx]))
            else:
                hv_with = hv_of_indices(current_selected + [idx])
            marginal = float(hv_with - current_hv)
            if marginal < 0:
                marginal = 0.0
            # diversity factor: min distance to current selected (if any)
            if not current_selected:
                div_factor = 1.0
            else:
                sel_pts = pts[np.array(current_selected, dtype=int), :]
                dists = np.linalg.norm(sel_pts - pts[int(idx)], axis=1)
                min_dist = float(np.min(dists))
                div_factor = 1.0 + (min_dist / max_dist)
            aug_score = marginal + alpha * float(single_hv.get(idx, 0.0)) + beta * marginal * float(single_norm.get(idx, 0.0)) * div_factor
            candidate_results.append((idx, marginal, hv_with, aug_score))
            if aug_score > best_aug or (aug_score == best_aug and (best_idx is None or idx < best_idx)):
                best_aug = aug_score
                best_idx = idx
                best_hv_with = hv_with

        # push back non-selected with updated estimates (use augmented as estimate)
        for idx, marginal, hv_with, aug_score in candidate_results:
            if idx == best_idx:
                continue
            heapq.heappush(heap, (-float(aug_score), int(idx)))

        # accept best (if none, break)
        if best_idx is None:
            break
        selected_list.append(int(best_idx))
        selected_set.add(int(best_idx))
        current_selected.append(int(best_idx))
        current_hv = float(best_hv_with)

        # if heap got too small, refill from other candidates not in heap but in diverse_candidates
        # ensure all remaining candidates are represented in heap
        present = set([t[1] for t in heap]) | selected_set
        for idx in diverse_candidates:
            if idx in present:
                continue
            est = single_hv.get(idx, hv_of_indices([idx]))
            heapq.heappush(heap, (-float(est), int(idx)))

    # fill remaining by highest single_hv among candidates not selected
    if len(selected_list) < k:
        remaining = [i for i in diverse_candidates if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: (-single_hv.get(x, 0.0), x))
        need = k - len(selected_list)
        for idx in remaining_sorted[:need]:
            selected_list.append(int(idx))
            selected_set.add(int(idx))

    # if still not enough (rare), fill from global box order
    if len(selected_list) < k:
        rem = [i for i in order_by_box if i not in selected_set]
        need = k - len(selected_list)
        for idx in rem[:need]:
            selected_list.append(int(idx))
            selected_set.add(int(idx))

    # bounded 1-swap local improvement
    max_swap_iters = 30
    swap_iter = 0
    improved = True
    tol = 1e-12
    candidate_explore = sorted([i for i in diverse_candidates if i not in selected_set],
                               key=lambda x: -single_hv.get(x, 0.0))[:min(300, len(diverse_candidates))]
    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        unselected = [i for i in diverse_candidates if i not in selected_set]
        if not unselected:
            break
        unselected_sorted = sorted(unselected, key=lambda x: -single_hv.get(x, 0.0))[:min(300, len(unselected))]
        for pos, s in enumerate(list(selected_list)):
            base_without = [x for x in selected_list if x != s]
            best_local_gain = 0.0
            best_u = None
            for u in unselected_sorted:
                if u in selected_set:
                    continue
                swapped_hv = hv_of_indices(base_without + [u])
                gain = swapped_hv - current_hv
                if gain > best_local_gain + tol:
                    best_local_gain = gain
                    best_u = u
            if best_u is not None:
                # perform swap
                for j, val in enumerate(selected_list):
                    if val == s:
                        selected_list[j] = int(best_u)
                        break
                selected_set.remove(s)
                selected_set.add(int(best_u))
                current_hv += best_local_gain
                improved = True
                break

    selected_list = selected_list[:k]
    subset = pts[np.array(selected_list, dtype=int), :].copy()
    return subset


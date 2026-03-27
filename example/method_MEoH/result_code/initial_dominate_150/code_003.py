import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    # lazy import for pygmo and friendly error if missing
    try:
        import pygmo as pg
    except Exception as _e:
        raise ImportError("pygmo (pg) is required for hypervolume computations but is not available.") from _e

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, m = points.shape

    if k <= 0:
        return np.empty((0, m), dtype=float)

    # If k >= n return all points (no selection needed)
    if k >= n:
        return points.copy()

    # Reference point default
    if reference_point is None:
        ref = np.max(points, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).ravel()
        if ref.size != m:
            raise ValueError("reference_point must have the same dimensionality as points")

    # Precompute single-point hypervolumes (fast closed form for minimization)
    diffs = ref.reshape(1, -1) - points
    indiv_hv = np.prod(np.maximum(diffs, 0.0), axis=1)

    # Normalize individual hv for combining into score
    max_indiv = float(np.max(indiv_hv)) if n > 0 else 1.0
    if max_indiv <= 0:
        indiv_norm = np.zeros_like(indiv_hv)
    else:
        indiv_norm = indiv_hv / max_indiv

    # Helper: compute hypervolume for a set of indices using pygmo
    def hv_of_indices(idx_list):
        if len(idx_list) == 0:
            return 0.0
        arr = points[np.array(idx_list, dtype=int), :]
        return pg.hypervolume(arr).compute(ref)

    # Scoring parameters (these define a different "score function" than pure marginal gain)
    alpha = 0.25       # weight for normalized individual hypervolume bonus
    lambda_div = 0.5   # penalty weight for closeness to already selected points
    # Note: higher lambda_div => stronger push for diversity (punishes near points more)

    # CELF (lazy greedy) selection with modified score:
    # Each heap element is (-estimated_score, idx, version)
    # version == current_size means the stored score was computed relative to the current selected set
    heap = []
    for i in range(n):
        # optimistic initial estimate: use individual hypervolume as upper bound and include normalized bonus
        est = float(indiv_hv[i] + alpha * indiv_norm[i])
        heap.append((-est, int(i), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    hv_S = 0.0
    curr_size = 0

    # main greedy loop: select k points
    while len(selected) < k and heap:
        neg_est, idx, ver = heapq.heappop(heap)
        if idx in selected_set:
            continue
        est = -neg_est
        if ver == curr_size:
            # entry is up-to-date -> accept it
            hv_with = hv_of_indices(selected + [idx])
            gain = hv_with - hv_S
            # update selection
            selected.append(int(idx))
            selected_set.add(int(idx))
            hv_S = hv_with
            curr_size += 1
        else:
            # outdated estimate: recompute the true marginal gain and the diversity penalty wrt current selected set
            hv_with = hv_of_indices(selected + [idx])
            gain = hv_with - hv_S
            # diversity penalty: mean distance to selected points (if any)
            if len(selected) == 0:
                penalty = 0.0
            else:
                # compute Euclidean distances
                dists = np.linalg.norm(points[idx] - points[np.array(selected, dtype=int)], axis=1)
                mean_dist = float(np.mean(dists)) if dists.size > 0 else 0.0
                # penalty decreases with distance; small mean_dist -> large penalty
                # use 1/(1+mean_dist) to keep penalty in (0,1], then scale by lambda_div
                penalty = 1.0 / (1.0 + mean_dist)
            # combined score: marginal gain minus diversity penalty plus normalized individual hv bonus
            score = float(gain) - lambda_div * float(penalty) + alpha * float(indiv_norm[idx])
            # push back updated with current version
            heapq.heappush(heap, (-score, int(idx), curr_size))

    # If we didn't get k due to exhausted heap (shouldn't happen), fill by top indiv_hv among unselected
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        if remaining:
            needed = k - len(selected)
            top_extra_idx = np.argsort(-indiv_hv[remaining])[:needed]
            for t in top_extra_idx:
                idx = remaining[int(t)]
                if idx not in selected_set:
                    selected.append(int(idx))
                    selected_set.add(int(idx))

    # Single-pass swap refinement: try replacing each selected point with any unselected point if it increases HV
    # (This refinement is based on true HV improvements only)
    hv_sel = hv_of_indices(selected)
    improved = True
    iter_limit = 1000
    it = 0
    tol = 1e-12
    while improved and it < iter_limit:
        improved = False
        it += 1
        for pos, s_idx in enumerate(list(selected)):
            base = [x for x in selected if x != s_idx]
            best_repl = None
            best_hv_after = hv_sel
            # try all unselected candidates
            for u_idx in range(n):
                if u_idx in selected_set:
                    continue
                hv_after = hv_of_indices(base + [u_idx])
                if hv_after > best_hv_after + tol:
                    best_hv_after = hv_after
                    best_repl = int(u_idx)
            if best_repl is not None:
                # perform swap
                selected[pos] = best_repl
                selected_set.remove(s_idx)
                selected_set.add(best_repl)
                hv_sel = best_hv_after
                improved = True
                # restart outer loop to allow cascading improvements
                break

    # Final safeguard: ensure exactly k unique indices
    seen = set()
    unique_selected = []
    for idx in selected:
        if idx not in seen:
            unique_selected.append(idx)
            seen.add(idx)
    selected = unique_selected

    rng = np.random.default_rng(0)  # deterministic padding selection
    if len(selected) < k:
        unselected = [i for i in range(n) if i not in selected]
        need = k - len(selected)
        if len(unselected) <= need:
            selected.extend(unselected)
        else:
            choices = list(np.array(unselected)[rng.choice(len(unselected), size=need, replace=False)])
            selected.extend([int(x) for x in choices])
    elif len(selected) > k:
        # drop smallest individual contributors among selected (fallback)
        selected = sorted(selected, key=lambda i: indiv_hv[i], reverse=True)[:k]

    # Return the selected points in deterministic order (sorted by index)
    selected = sorted(int(i) for i in selected[:k])
    return points[np.array(selected, dtype=int), :].copy()


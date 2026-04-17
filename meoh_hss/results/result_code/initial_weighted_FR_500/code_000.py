import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    # Deterministic seed
    seed = 1
    np.random.seed(seed)

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, m = points.shape

    if not (1 <= k <= n):
        raise ValueError("k must be between 1 and number of points")

    # Reference point handling
    if reference_point is None:
        ref = np.max(points, axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float).reshape(-1)
        if ref.size != m:
            raise ValueError("reference_point must have same dimension as points")

    # 1) Fast nondominated filter (returns indices of non-dominated points)
    def nondominated_indices(X: np.ndarray):
        N = X.shape[0]
        is_dom = np.zeros(N, dtype=bool)
        for i in range(N):
            if is_dom[i]:
                continue
            # any j dominates i?
            # j dominates i if all X[j] <= X[i] and any X[j] < X[i]
            cmp = np.all(X <= X[i], axis=1) & np.any(X < X[i], axis=1)
            # exclude self
            cmp[i] = False
            if np.any(cmp):
                is_dom[i] = True
        return np.where(~is_dom)[0]

    nd_idx = nondominated_indices(points)
    # If nondominated set smaller than k, we must consider all points later.
    if nd_idx.size >= k:
        pool_idx = nd_idx
    else:
        pool_idx = np.arange(n)

    pool = points[pool_idx]

    # 2) Cheap per-point approximate score: individual box volume (upper bound of single-point hv)
    eps = 1e-12
    diffs = np.maximum(ref - pool, eps)
    approx_vol = np.prod(diffs, axis=1)

    # Shortlist parameters (small multiple of k but capped by pool size)
    shortlist_size = int(min(pool.shape[0], max(5 * k, k + 50)))
    # deterministic sort: argsort of negative approx_vol, tie-break by pool_idx order
    order = np.lexsort((pool_idx, -approx_vol))
    top_idx_in_pool = order[:shortlist_size]
    candidates_idx = pool_idx[top_idx_in_pool].tolist()

    # 3) Greedy selection on candidates using exact hypervolume gains
    selected_idx = []
    selected_pts = np.empty((0, m), dtype=float)
    current_hv = 0.0

    # Pre-construct a map from index -> point for faster access
    # (points is already in memory; use direct indexing)
    # Greedy: select up to min(k, len(candidates))
    for _ in range(min(k, len(candidates_idx))):
        best_gain = -np.inf
        best_j = None
        best_hv = None
        # For each candidate remaining compute hv(selected U {cand}) and choose max gain
        for cand in candidates_idx:
            if cand in selected_idx:
                continue
            cand_pt = points[cand].reshape(1, -1)
            if selected_pts.shape[0] == 0:
                hv_cand = pg.hypervolume(cand_pt).compute(ref)
            else:
                hv_cand = pg.hypervolume(np.vstack([selected_pts, cand_pt])).compute(ref)
            gain = hv_cand - current_hv
            # deterministic tie-breaker: smaller index preferred
            if (gain > best_gain) or (np.isclose(gain, best_gain) and (best_j is None or cand < best_j)):
                best_gain = gain
                best_j = cand
                best_hv = hv_cand
        if best_j is None:
            break
        # accept best_j
        selected_idx.append(best_j)
        selected_pts = np.vstack([selected_pts, points[best_j].reshape(1, -1)])
        current_hv = float(best_hv)

    # 4) If fewer than k selected (shortlist too small), fill deterministically from remaining by approx_vol over all unselected
    if len(selected_idx) < k:
        unselected = [i for i in range(n) if i not in selected_idx]
        # compute approx vols for unselected relative to ref
        diffs_un = np.maximum(ref - points[unselected], eps)
        approx_un = np.prod(diffs_un, axis=1)
        order_un = np.argsort(-approx_un, kind='stable')
        need = k - len(selected_idx)
        for idx_pos in order_un[:need]:
            selected_idx.append(unselected[int(idx_pos)])
        selected_pts = points[selected_idx]

        # update current_hv to exact hv of filled set
        current_hv = float(pg.hypervolume(selected_pts).compute(ref))

    # 5) Local swap improvement (cached current_hv), limited iterations to avoid long runtimes
    # Consider swaps between selected set and the rest of points (global), but evaluate only until no improvement
    max_swaps = max(50, k * 5)  # limit number of successful swaps
    swaps_done = 0
    improved = True
    # create set for quick lookup
    selected_set = set(selected_idx)
    non_selected_global = [i for i in range(n) if i not in selected_set]

    while improved and swaps_done < max_swaps:
        improved = False
        # try all selected positions and attempt swapping with best candidate in non_selected_global
        # deterministic ordering on selected and non-selected to ensure reproducibility
        for si_pos, si in enumerate(list(selected_idx)):
            best_local_gain = 0.0
            best_local_r = None
            best_local_hv = None
            for r in non_selected_global:
                # try replacing selected_idx[si_pos] with r
                new_selected = selected_pts.copy()
                new_selected[si_pos, :] = points[r]
                new_hv = pg.hypervolume(new_selected).compute(ref)
                gain = new_hv - current_hv
                if (gain > best_local_gain) or (np.isclose(gain, best_local_gain) and (best_local_r is None or r < best_local_r)):
                    best_local_gain = float(gain)
                    best_local_r = r
                    best_local_hv = float(new_hv)
            if best_local_gain > 1e-12 and best_local_r is not None:
                # perform swap
                old_idx = selected_idx[si_pos]
                selected_idx[si_pos] = best_local_r
                selected_set.remove(old_idx)
                selected_set.add(best_local_r)
                # update non_selected_global: replace best_local_r with old_idx deterministically
                non_selected_global.remove(best_local_r)
                non_selected_global.append(old_idx)
                selected_pts[si_pos, :] = points[best_local_r]
                current_hv = best_local_hv
                swaps_done += 1
                improved = True
                break  # restart scan over selected points after a successful swap
        # recompute non_selected_global in case of duplicates/order
        non_selected_global = [i for i in range(n) if i not in set(selected_idx)]

    # Final result: ensure exactly k unique points (trim or pad deterministically)
    # If any duplicate indices accidentally present, deduplicate preserving order then pad
    final_idx = []
    seen = set()
    for idx in selected_idx:
        if idx not in seen:
            final_idx.append(idx)
            seen.add(idx)
        if len(final_idx) == k:
            break

    if len(final_idx) < k:
        # pad from remaining indices by deterministic approx_vol order
        remaining = [i for i in range(n) if i not in seen]
        if remaining:
            diffs_rem = np.maximum(ref - points[remaining], eps)
            approx_rem = np.prod(diffs_rem, axis=1)
            order_rem = np.argsort(-approx_rem, kind='stable')
            need = k - len(final_idx)
            for pos in order_rem[:need]:
                final_idx.append(remaining[int(pos)])
                if len(final_idx) == k:
                    break

    result = points[final_idx][:k]
    return result.astype(float)


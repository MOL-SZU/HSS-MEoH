import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    try:
        import pygmo as pg
    except Exception:
        pg = None

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape

    if k <= 0:
        return np.zeros((0, D), dtype=float)
    if N == 0:
        return np.zeros((0, D), dtype=float)

    # reference point default and sanitization
    if reference_point is None:
        reference = points.max(axis=0) * 1.1
    else:
        reference = np.asarray(reference_point, dtype=float)
        if reference.shape != (D,):
            raise ValueError("reference_point must have shape (D,)")

    # Remove points that are strictly worse than reference in any dim (cannot contribute)
    valid_mask = np.all(points <= reference, axis=1)
    if not np.any(valid_mask):
        # no point contributes; fall back to k nearest to reference (smallest distance)
        idx = np.argsort(np.linalg.norm(points - reference, axis=1))[:min(k, N)]
        return points[idx].copy()

    pts = points[valid_mask]
    orig_idx_map = np.nonzero(valid_mask)[0]
    P = pts.shape[0]
    if k >= P:
        # choose all valid points and possibly fill with some dominated across original if needed
        sel_idx = orig_idx_map.tolist()
        if len(sel_idx) > k:
            sel_idx = sel_idx[:k]
        return points[sel_idx].copy()

    # Precompute box volumes (rectangular volume between point and reference)
    box_vol = np.prod(np.maximum(0.0, reference - pts), axis=1)

    # Normalize coordinates to [0,1] between per-dim minima and reference for distance computations
    mins = pts.min(axis=0)
    denom = (reference - mins).copy()
    # avoid divide by zero
    denom[denom <= 0] = 1.0
    scaled = (pts - mins) / denom

    # Farthest-first traversal (k-center) seeded by largest box volume
    selected_local = []
    used = np.zeros(P, dtype=bool)
    # seed with the point with maximum box_vol
    seed = int(np.argmax(box_vol))
    selected_local.append(seed)
    used[seed] = True

    # distances to nearest center
    diff = scaled - scaled[seed:seed+1]
    min_dists = np.linalg.norm(diff, axis=1)

    for _ in range(1, k):
        # pick point with maximum distance to current centers
        # restrict to unused points
        min_dists[used] = -1.0
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] < 0:
            # no remaining, break
            break
        selected_local.append(next_idx)
        used[next_idx] = True
        # update min_dists
        d2 = np.linalg.norm(scaled - scaled[next_idx:next_idx+1], axis=1)
        min_dists = np.minimum(min_dists, d2)

    # If we have fewer than k (rare), fill by box_vol
    if len(selected_local) < k:
        remaining = np.where(~used)[0]
        need = k - len(selected_local)
        order = np.argsort(-box_vol[remaining])[:need]
        for o in order:
            selected_local.append(int(remaining[o]))
            used[remaining[o]] = True

    selected_local = selected_local[:k]
    selected_global = orig_idx_map[selected_local].tolist()

    # Helper: approximate hypervolume via shared MC samples (samples are reused)
    # Build sample domain [mins_all, reference]
    mins_all = points.min(axis=0)
    low = np.minimum(mins_all, mins)
    high = reference.copy()
    span = high - low
    span_safe = span.copy()
    span_safe[span_safe <= 0] = 1e-6

    # Prepare small shared sample set for swap evaluation
    rng = np.random.default_rng(12345)
    # sample size moderate and bounded
    S_eval = int(min(4000, max(1500, 300 * min(k, 10))))
    samples = rng.random((S_eval, D)) * span_safe + low
    domain_vol = float(np.prod(span_safe))

    def approx_hv_mc(subset_pts):
        """
        approximate hypervolume of subset_pts via shared samples in [low, high]
        """
        if subset_pts.size == 0:
            return 0.0
        # dominated if any point <= sample elementwise
        # subset_pts shape (m, D); samples (S_eval, D)
        dom = np.any(np.all(subset_pts[:, None, :] <= samples[None, :, :], axis=2), axis=0)
        frac = float(dom.sum()) / float(S_eval)
        return frac * domain_vol

    def exact_hv_if_possible(subset_pts):
        if pg is None:
            return None
        try:
            hv = float(pg.hypervolume(subset_pts).compute(reference))
            return hv
        except Exception:
            return None

    # compute current hypervolume (try exact)
    sel_pts = points[selected_global]
    current_hv = exact_hv_if_possible(sel_pts)
    if current_hv is None:
        current_hv = approx_hv_mc(sel_pts)

    # Prepare neighbor lists: for each selected local index, find nearest neighbors in scaled space
    # We'll restrict candidates per selected point to nearest L neighbors (not selected)
    L_neighbors = 10  # local candidate pool
    # Precompute distances matrix rows lazily: distances from each selected center to all points
    scaled_all = (points[orig_idx_map] - mins) / denom  # scaled for pts only mapping to orig_idx_map
    # For easier neighbor search, use scaled over the full original points (but safe to use pts)
    scaled_full = (points - mins_all) / np.maximum(1e-12, (reference - mins_all))

    max_passes = 3
    improved_overall = True
    passes = 0
    tol = 1e-12

    while improved_overall and passes < max_passes:
        passes += 1
        improved_overall = False
        # iterate over current selection positions (try swapping each with local nearby candidates)
        for pos_idx, sel_g in enumerate(list(selected_global)):
            # Build candidate pool: nearest L_neighbors not currently selected
            center_point = points[sel_g]
            # distances to all other points in scaled_full space
            dists = np.linalg.norm(scaled_full - scaled_full[sel_g:sel_g+1], axis=1)
            # exclude currently selected
            selected_mask_full = np.zeros(N, dtype=bool)
            selected_mask_full[selected_global] = True
            dists[selected_mask_full] = np.inf
            cand_idx = np.argsort(dists)[:min(L_neighbors * 3, N)]
            if cand_idx.size == 0:
                continue
            # Evaluate each candidate (limited budget)
            best_local_improvement = 0.0
            best_candidate = None
            best_hv = current_hv
            # We will use exact hv if available else approximate MC shared samples
            for cand in cand_idx:
                # candidate must be valid (<=reference)
                if not np.all(points[cand] <= reference):
                    continue
                # quick box-volume pruning: if box_vol of candidate less than smallest in selection and likely not improving, still allow a few tries
                # Build trial set: replace sel_g with cand
                trial_sel = selected_global.copy()
                trial_sel[pos_idx] = int(cand)
                trial_pts = points[trial_sel]
                hv_trial = exact_hv_if_possible(trial_pts)
                if hv_trial is None:
                    hv_trial = approx_hv_mc(trial_pts)
                if hv_trial > best_hv + tol:
                    best_hv = hv_trial
                    best_candidate = int(cand)
                    best_local_improvement = hv_trial - current_hv
                # small budget: if we already found an improving swap, we can prefer the best among evaluated neighbors
            if best_candidate is not None:
                # accept swap
                selected_global[pos_idx] = best_candidate
                current_hv = best_hv
                improved_overall = True
                # update for next iterations
        # end for selected
    # Ensure uniqueness (if duplicates due to replacements, fix by filling with best remaining)
    selected_global = list(dict.fromkeys(selected_global))
    if len(selected_global) < k:
        # fill by best box volumes among remaining valid points
        remaining = [i for i in range(N) if i not in selected_global and np.all(points[i] <= reference)]
        if remaining:
            rem_vols = np.prod(np.maximum(0.0, reference - points[remaining]), axis=1)
            need = k - len(selected_global)
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_global.append(int(remaining[idx]))
    # If too many (rare), trim by box volume small-to-large
    if len(selected_global) > k:
        sel_arr = np.array(selected_global, dtype=int)
        sel_box = np.prod(np.maximum(0.0, reference - points[sel_arr]), axis=1)
        keep_order = np.argsort(-sel_box)[:k]
        selected_global = sel_arr[keep_order].tolist()

    # Final safety: if still insufficient, pad with best by box volume across all valid points
    if len(selected_global) < k:
        already = set(selected_global)
        valid_all = np.where(np.all(points <= reference, axis=1))[0]
        vols = np.prod(np.maximum(0.0, reference - points[valid_all]), axis=1)
        order = np.argsort(-vols)
        for idx in order:
            gi = int(valid_all[idx])
            if gi in already:
                continue
            selected_global.append(gi)
            already.add(gi)
            if len(selected_global) >= k:
                break

    selected_global = selected_global[:k]
    subset = points[selected_global].copy()
    return np.asarray(subset, dtype=float)


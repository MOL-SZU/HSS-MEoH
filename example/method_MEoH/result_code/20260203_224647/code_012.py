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

    EPS = 1e-12

    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k <= 0 or N == 0:
        return np.zeros((0, D), dtype=float)

    if reference_point is None:
        reference_point = pts.max(axis=0) * 1.1
    ref = np.asarray(reference_point, dtype=float).reshape(D,)
    if ref.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Build axis-aligned boxes between each point and the reference point
    lows = np.minimum(pts, ref)
    highs = np.maximum(pts, ref)
    extents = np.maximum(highs - lows, 0.0)
    indiv_vols = np.prod(extents, axis=1)

    # Degenerate cases
    if np.all(indiv_vols <= EPS):
        idxs = np.argsort(np.sum((ref - pts) ** 2, axis=1))[:min(k, N)]
        return pts[idxs].copy()

    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    global_span = np.maximum(global_high - global_low, 0.0)
    if np.any(global_span <= 0):
        # fallback: pick largest indiv vols
        idxs = np.argsort(-indiv_vols)[:min(k, N)]
        return pts[idxs].copy()
    denom_dist = np.linalg.norm(global_span)
    if denom_dist <= 0:
        denom_dist = 1.0

    # Monte-Carlo sampling budget (adaptive to k and D)
    rng = np.random.default_rng()
    M = int(np.clip(2000 + 350 * min(k, 100) + 80 * D, 1500, 10000))
    samples = global_low + rng.random((M, D)) * global_span  # (M, D)

    # Precompute coverage matrix: (N, M) boolean, True if sample inside box of point i
    # We'll compute in a memory-efficient vectorized manner
    ge_low = samples[:, None, :] >= (lows[None, :, :] - EPS)
    le_high = samples[:, None, :] <= (highs[None, :, :] + EPS)
    within = np.logical_and(ge_low, le_high).all(axis=2)  # shape (M, N)
    covers = within.T  # (N, M)

    # Pre-filter candidates with zero individual volume or zero coverage on samples
    valid_mask = (indiv_vols > EPS) & (covers.sum(axis=1) > 0)
    if not np.any(valid_mask):
        idxs = np.argsort(-indiv_vols)[:min(k, N)]
        return pts[idxs].copy()
    remaining = valid_mask.copy()

    # Greedy selection maximizing marginal coverage on samples
    cover_counts = np.zeros(M, dtype=int)  # counts of how many selected boxes cover each sample
    selected = []
    selected_set = set()

    # diversity weight for tie-breaking
    gamma = 0.35

    max_select = min(k, int(remaining.sum()))

    for it in range(max_select):
        rem_idx = np.where(remaining)[0]
        if rem_idx.size == 0:
            break
        # uncovered samples
        uncovered = cover_counts == 0
        if np.any(uncovered):
            # marginal contributions: for each remaining candidate, how many currently uncovered samples it would cover
            # shape (N_rem,)
            contribs = covers[rem_idx][:, uncovered].sum(axis=1)
        else:
            contribs = np.zeros(rem_idx.size, dtype=int)

        # If all contributions zero, fall back to individual volumes (prefer diversity)
        if np.all(contribs == 0):
            # use indiv_vols with diversity bonus
            if len(selected) == 0:
                best_local_idx = int(np.argmax(indiv_vols[rem_idx]))
                best = int(rem_idx[best_local_idx])
            else:
                sel_arr = np.array(selected, dtype=int)
                diffs = pts[rem_idx][:, None, :] - pts[sel_arr][None, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                min_dists = np.min(dists, axis=1)
                dist_factor = np.clip(min_dists / denom_dist, 0.0, 1.0)
                scores = indiv_vols[rem_idx] * (1.0 + gamma * dist_factor)
                best_local_idx = int(np.argmax(scores))
                best = int(rem_idx[best_local_idx])
        else:
            # tie-break among maximal contribs by (indiv_vol * (1 + gamma * min_dist))
            max_contrib = int(contribs.max())
            candidates = np.where(contribs == max_contrib)[0]
            if candidates.size == 1:
                best = int(rem_idx[candidates[0]])
            else:
                cand_idx = rem_idx[candidates]
                if len(selected) == 0:
                    # choose largest individual volume
                    best = int(cand_idx[np.argmax(indiv_vols[cand_idx])])
                else:
                    sel_arr = np.array(selected, dtype=int)
                    diffs = pts[cand_idx][:, None, :] - pts[sel_arr][None, :, :]
                    dists = np.linalg.norm(diffs, axis=2)
                    min_dists = np.min(dists, axis=1)
                    dist_factor = np.clip(min_dists / denom_dist, 0.0, 1.0)
                    scores = indiv_vols[cand_idx] * (1.0 + gamma * dist_factor)
                    best = int(cand_idx[np.argmax(scores)])

        # If best yields no additional samples (safety), break
        added = int(np.sum((cover_counts == 0) & covers[best]))
        if added <= 0 and len(selected) > 0:
            # nothing more to add in terms of uncovered samples
            break

        # select it
        selected.append(int(best))
        selected_set.add(int(best))
        remaining[best] = False
        cover_counts += covers[best].astype(int)

    # If we didn't reach k, pad by best remaining individual volumes
    if len(selected) < k:
        rem_idx = np.where(np.array([i for i in range(N) if i not in selected_set]))[0]
        if rem_idx.size > 0:
            # sort by individual volume desc, but only those with positive volume
            order = rem_idx[np.argsort(-indiv_vols[rem_idx])]
            for idx in order:
                if len(selected) >= k:
                    break
                selected.append(int(idx))
                selected_set.add(int(idx))
                cover_counts += covers[int(idx)].astype(int)

    # Limit selected to k
    selected = selected[:k]
    selected_set = set(selected)
    cover_counts = np.zeros(M, dtype=int)
    for s in selected:
        cover_counts += covers[s].astype(int)
    curr_covered = int(np.count_nonzero(cover_counts > 0))

    # Local 1-for-1 swap refinement: try to swap any selected with promising unselected candidates
    max_swaps = min(6 * k + 100, 600)
    swaps = 0
    improved = True
    # Precompute a ranking of unselected candidates by individual volumes and sample coverage
    all_idx = np.arange(N)
    unselected = np.array([i for i in all_idx if i not in selected_set], dtype=int)
    if unselected.size > 0:
        un_scores = indiv_vols[unselected] + 0.5 * covers[unselected].sum(axis=1)
        un_order = unselected[np.argsort(-un_scores)]
    else:
        un_order = np.array([], dtype=int)

    while improved and swaps < max_swaps:
        improved = False
        swaps += 1
        # iterate selected indices (try to replace each)
        for s in list(selected):
            # remove s contribution
            cover_excl = cover_counts - covers[s].astype(int)
            # precompute uncovered after removal
            uncovered_after_removal = cover_excl == 0
            best_gain = 0
            best_cand = None
            # try top candidates only for speed
            top_try = un_order[:min(300, un_order.size)]
            for cand in top_try:
                if cand in selected_set:
                    continue
                # new covered count when swapping s out and cand in:
                new_counts = cover_excl + covers[cand].astype(int)
                new_covered = int(np.count_nonzero(new_counts > 0))
                gain = new_covered - curr_covered
                if gain > best_gain:
                    best_gain = gain
                    best_cand = int(cand)
                    # early exit for big gain
                    if best_gain > 0:
                        break
            if best_gain > 0 and best_cand is not None:
                # perform swap s -> best_cand
                selected_set.discard(int(s))
                selected_set.add(int(best_cand))
                selected = [int(best_cand) if x == s else x for x in selected]
                cover_counts = cover_excl + covers[best_cand].astype(int)
                curr_covered = int(np.count_nonzero(cover_counts > 0))
                # refresh unselected order
                unselected = np.array([i for i in all_idx if i not in selected_set], dtype=int)
                if unselected.size > 0:
                    un_scores = indiv_vols[unselected] + 0.5 * covers[unselected].sum(axis=1)
                    un_order = unselected[np.argsort(-un_scores)]
                else:
                    un_order = np.array([], dtype=int)
                improved = True
                break  # restart loop over selected

    final_idx = np.array(list(selected_set), dtype=int)
    # If we have more than k (shouldn't usually happen), pick those with largest empirical coverage
    if final_idx.size > k:
        indiv_counts = covers[final_idx, :].sum(axis=1)
        order = np.argsort(-indiv_counts)
        final_idx = final_idx[order[:k]]
    elif final_idx.size < k:
        rem = np.array([i for i in range(N) if i not in set(final_idx)], dtype=int)
        if rem.size > 0:
            pad_order = rem[np.argsort(-indiv_vols[rem])]
            need = k - final_idx.size
            to_add = pad_order[:need]
            final_idx = np.concatenate([final_idx, to_add])

    final_idx = final_idx[:k]
    subset = pts[final_idx].copy()
    return subset


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    EPS = 1e-12

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape

    if k <= 0 or N == 0:
        return np.zeros((0, D), dtype=float)

    if reference_point is None:
        reference_point = points.max(axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(D,)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Boxes anchored between each point and reference
    lows = np.minimum(points, reference_point)
    highs = np.maximum(points, reference_point)
    diffs = np.maximum(highs - lows, 0.0)
    indiv_vols = np.prod(diffs, axis=1)

    # Degenerate cases
    if np.all(indiv_vols <= EPS):
        take = min(k, N)
        return points[:take].copy()
    if k >= N:
        return points.copy()

    # Global sampling bounding box
    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    span = np.maximum(global_high - global_low, 0.0)
    total_box_vol = np.prod(np.maximum(span, 1e-18))
    if total_box_vol <= EPS:
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    # Prune candidates to top-P by individual volume to reduce cost (still include at least some)
    # choose P relative to k and N
    P_max = max(8 * k, 300)
    P = min(N, P_max)
    idx_by_vol = np.argsort(-indiv_vols)
    top_idx = idx_by_vol[:P]
    others_idx = idx_by_vol[P:].tolist()

    vols_p = indiv_vols[top_idx]
    max_vol = vols_p.max() if vols_p.size > 0 else 0.0
    vol_norm = vols_p / max_vol if max_vol > 0 else np.zeros_like(vols_p)

    # Stratified Latin-Hypercube Sampling (LHS) for better uniform coverage
    rng = np.random.default_rng()
    # adaptive sample size: base on D and k, but different schedule than original
    S_base = 2500
    S_by_k = 500 * k
    S_by_dim = 800 * max(1, D)
    S = int(min(45000, max(S_base, S_by_k, S_by_dim)))
    S = max(1500, S)
    # Build LHS samples
    samples = np.empty((S, D), dtype=float)
    for d in range(D):
        perm = rng.permutation(S)
        # jitter in each stratum
        jitter = rng.random(S)
        samples[:, d] = (perm + jitter) / float(S) * span[d] + global_low[d]

    # Precompute inclusion for the pruned candidates (S x P)
    inside = np.zeros((S, P), dtype=bool)
    max_cols_per_chunk = 2000
    start = 0
    while start < P:
        end = min(P, start + max_cols_per_chunk)
        sl = slice(start, end)
        ge = samples[:, None, :] >= lows[top_idx][sl][None, :, :]
        le = samples[:, None, :] <= highs[top_idx][sl][None, :, :]
        inside[:, sl] = np.all(ge & le, axis=2)
        start = end

    sample_counts = inside.sum(axis=0)  # shape (P,)
    nonzero_mask = sample_counts > 0
    if not np.any(nonzero_mask):
        # nothing covered in samples (very unlikely), fall back
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    # Keep only those with at least one sample
    nonzero_idx = np.nonzero(nonzero_mask)[0]
    zero_idx = np.nonzero(~nonzero_mask)[0].tolist()
    inside_p = inside[:, nonzero_idx]
    vols_p = vols_p[nonzero_idx]
    vol_norm = vol_norm[nonzero_idx]
    top_idx_p = top_idx[nonzero_idx]
    Pp = inside_p.shape[1]

    # estimated marginals
    counts_p = inside_p.sum(axis=0).astype(float)
    est_marginals = (counts_p / float(S)) * total_box_vol

    # Diversity additive bonus configuration
    # diversity bonus = diversity_coef * indiv_vol * mean_normalized_distance_to_selected
    diversity_coef = 0.25  # tuned moderate additive encouragement
    # precompute pairwise distances between pruned candidates (for speed)
    # We'll compute distances on original objective space points (L2) normalized by diagonal length
    diag = np.linalg.norm(span) if np.linalg.norm(span) > 0 else 1.0
    cand_points = points[top_idx_p]  # shape (Pp, D)

    # initial scores: just estimated marginal + tiny tie-breaker
    scores = est_marginals.copy()
    # small tie-break using indiv volume
    scores += 1e-18 * vols_p

    # lazy heap
    heap = [(-float(scores[i]), int(i)) for i in range(Pp)]
    heapq.heapify(heap)

    covered = np.zeros(S, dtype=bool)
    selected_local = []
    selected_set = set()

    tol = 1e-12
    # Precompute pairwise distances on demand; but to avoid O(P^2) store candidate coordinates
    while len(selected_local) < k and heap:
        neg_score, local_idx = heapq.heappop(heap)
        if local_idx in selected_set:
            continue
        popped_score = -neg_score
        # recompute actual marginal volume (new samples covered)
        new_cov = (~covered) & inside_p[:, local_idx]
        new_count = int(new_cov.sum())
        cur_marg = (new_count / float(S)) * total_box_vol
        # compute diversity bonus relative to already selected points
        if len(selected_local) == 0:
            diversity_bonus = 0.0
        else:
            sel_pts = cand_points[np.array(selected_local)]
            # distances between candidate and each selected
            dif = cand_points[local_idx][None, :] - sel_pts  # shape (m, D)
            dists = np.linalg.norm(dif, axis=1) / diag  # normalized distances
            mean_norm_dist = float(np.mean(dists)) if dists.size > 0 else 0.0
            diversity_bonus = diversity_coef * vols_p[local_idx] * mean_norm_dist
        cur_score = cur_marg + diversity_bonus
        # if popped stale, push updated
        if not np.isclose(cur_score, popped_score, atol=tol, rtol=1e-9):
            heapq.heappush(heap, (-float(cur_score), int(local_idx)))
            continue
        # if marginal effectively zero, break
        if cur_marg <= EPS:
            break
        # accept
        selected_local.append(int(local_idx))
        selected_set.add(int(local_idx))
        covered |= inside_p[:, local_idx]
        # early stop if all samples covered
        if covered.all():
            break

    # If selected fewer than k, fill from remaining pruned candidates by descending indiv_vol
    if len(selected_local) < k:
        remaining = [i for i in range(Pp) if i not in selected_set]
        if remaining:
            rem_vols = vols_p[remaining]
            need = k - len(selected_local)
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_local.append(int(remaining[int(idx)]))
                selected_set.add(int(remaining[int(idx)]))

    # Map local selected indices back to original indices
    selected_original = list(top_idx_p[selected_local])

    # If still fewer than k, pad from zero-covered pruned ones, then from others by global volume
    if len(selected_original) < k:
        pad_needed = k - len(selected_original)
        pads = []
        # zeros within pruned set
        for z in zero_idx:
            if len(pads) >= pad_needed:
                break
            orig = top_idx[z] if z < len(top_idx) else None
            if orig is not None and orig not in selected_original:
                pads.append(orig)
        # then from others (not in top_idx) by global volume
        if len(pads) < pad_needed:
            for idx in idx_by_vol:
                if idx not in selected_original and idx not in pads:
                    pads.append(int(idx))
                    if len(pads) >= pad_needed:
                        break
        selected_original.extend(pads[:pad_needed])

    # Safety: if still fewer (shouldn't), pad with first points
    if len(selected_original) < k:
        for i in range(N):
            if len(selected_original) >= k:
                break
            if i not in selected_original:
                selected_original.append(int(i))

    selected_original = np.array(selected_original[:k], dtype=int)
    subset = points[selected_original, :].copy()
    return subset


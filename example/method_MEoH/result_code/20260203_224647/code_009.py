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

    # Degenerate handling
    if np.all(indiv_vols <= EPS):
        take = min(k, N)
        return points[:take].copy()
    if k >= N:
        return points.copy()

    # Global sampling region
    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    box_sizes = global_high - global_low
    total_box_vol = np.prod(np.maximum(box_sizes, 0.0))
    if total_box_vol <= EPS:
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    # Adaptive sample count using stratified (Latin Hypercube) sampling for better coverage
    base = 1200
    s_by_k = 250 * max(1, k)
    s_by_dim = 450 * max(1, D)
    S = int(min(40000, max(base, s_by_k, s_by_dim)))
    S = max(800, S)

    rng = np.random.default_rng()
    # Build Latin Hypercube-like samples in bounding box (handle zero-size dims)
    span = np.where(box_sizes > 0, box_sizes, 1.0)
    # Create stratified positions per dimension
    u = np.empty((S, D), dtype=float)
    for d in range(D):
        perm = rng.permutation(S)
        # random jitter within strata
        jitter = rng.random(S)
        u[:, d] = (perm + jitter) / float(S)
    samples = global_low + u * span

    # Precompute inclusion matrix in chunks (S x N) boolean
    max_cols_per_chunk = 2500
    inside = np.zeros((S, N), dtype=bool)
    start = 0
    while start < N:
        end = min(N, start + max_cols_per_chunk)
        sl = slice(start, end)
        ge = samples[:, None, :] >= lows[sl][None, :, :]
        le = samples[:, None, :] <= highs[sl][None, :, :]
        inside[:, sl] = np.all(ge & le, axis=2)
        start = end

    sample_counts = inside.sum(axis=0)  # per candidate coverage
    nonzero_mask = sample_counts > 0
    nonzero_idx = np.nonzero(nonzero_mask)[0]
    zero_idx = np.nonzero(~nonzero_mask)[0].tolist()

    if nonzero_idx.size == 0:
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    # Work only with nonzero candidates
    inside_p = inside[:, nonzero_idx]  # S x P
    vols_p = indiv_vols[nonzero_idx]
    counts_p = sample_counts[nonzero_idx].astype(float)
    P = inside_p.shape[1]

    # Initial estimated marginals
    est_marginals = (counts_p / float(S)) * total_box_vol

    # New hybrid bias (different form): use sqrt-based multiplicative bias and log-additive bias
    # Encourages moderate-volume candidates while penalizing huge dominated boxes
    max_vol = vols_p.max() if vols_p.size > 0 else 0.0
    vol_norm = vols_p / max_vol if max_vol > 0 else np.zeros_like(vols_p)

    mult_strength = 0.28
    mult_exp = 0.55
    add_strength = 0.48
    # use a smoothed log factor to prefer informative mid-size boxes
    add_term = add_strength * np.log1p(1.0 + vol_norm * 9.0)  # in [0, ~2.3*add_strength]
    comb_factor = 1.0 + mult_strength * (vol_norm ** mult_exp) + add_term

    # Initial scores and lightweight candidate pruning
    scores = est_marginals * comb_factor
    max_cand = max(1500, 5 * k * max(1, D))
    if P > max_cand:
        top_idx = np.argsort(-scores)[:max_cand]
        inside_p = inside_p[:, top_idx]
        vols_p = vols_p[top_idx]
        counts_p = counts_p[top_idx]
        vol_norm = vol_norm[top_idx]
        est_marginals = est_marginals[top_idx]
        comb_factor = comb_factor[top_idx]
        scores = scores[top_idx]
        nonzero_idx = nonzero_idx[top_idx]
        P = inside_p.shape[1]

    # Build lazy heap: (-score, local_idx)
    heap = [(-float(scores[i]), int(i)) for i in range(P)]
    heapq.heapify(heap)

    covered = np.zeros(S, dtype=bool)
    selected_local = []
    selected_set = set()
    tol = 1e-12

    # Lazy greedy (CELF-like) selection based on sample marginal gains
    while len(selected_local) < k and heap:
        neg_score, local_idx = heapq.heappop(heap)
        if local_idx in selected_set:
            continue
        popped_score = -neg_score
        new_cov_mask = (~covered) & inside_p[:, local_idx]
        new_count = int(new_cov_mask.sum())
        cur_marg = (new_count / float(S)) * total_box_vol
        cur_score = cur_marg * comb_factor[local_idx]
        # If stale, push updated score back
        if not np.isclose(cur_score, popped_score, atol=tol, rtol=1e-9):
            if cur_score > EPS:
                heapq.heappush(heap, (-float(cur_score), int(local_idx)))
            continue
        if cur_marg <= EPS:
            # No useful marginal left for this candidate; skip it
            continue
        # Select it
        selected_local.append(int(local_idx))
        selected_set.add(int(local_idx))
        covered |= inside_p[:, local_idx]
        if covered.all():
            break

    # Fill remaining slots by largest remaining individual volumes (among considered candidates first)
    if len(selected_local) < k:
        remaining = [i for i in range(P) if i not in selected_set]
        need = k - len(selected_local)
        if remaining:
            rem_vols = vols_p[remaining]
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_local.append(int(remaining[int(idx)]))
                selected_set.add(int(remaining[int(idx)]))

    # Map back to original indices
    selected_original = list(nonzero_idx[selected_local]) if selected_local else []
    # Pad if still fewer than k (use zero-coverage candidates or remaining highest volumes)
    if len(selected_original) < k:
        pad_needed = k - len(selected_original)
        pads = []
        # first use zero-coverage candidates with nonzero volume if possible
        if zero_idx:
            # sort zero_idx by individual volume desc
            zero_sorted = sorted(zero_idx, key=lambda i: -indiv_vols[i])
            for z in zero_sorted:
                if z not in selected_original:
                    pads.append(z)
                    if len(pads) >= pad_needed:
                        break
        if len(pads) < pad_needed:
            # fill with top-volume remaining candidates
            remaining_all = [i for i in range(N) if i not in selected_original and i not in pads]
            if remaining_all:
                rem_sorted = sorted(remaining_all, key=lambda i: -indiv_vols[i])
                for r in rem_sorted:
                    pads.append(r)
                    if len(pads) >= pad_needed:
                        break
        selected_original.extend(pads[:pad_needed])

    # Final selection (ensure length k and valid indices)
    selected_original = np.array(selected_original[:k], dtype=int)
    subset = points[selected_original, :].copy()
    return subset


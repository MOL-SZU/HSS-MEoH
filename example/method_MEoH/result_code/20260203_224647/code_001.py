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

    # Define axis-aligned boxes between each point and reference
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

    # Global sampling region
    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    box_sizes = global_high - global_low
    total_box_vol = np.prod(np.maximum(box_sizes, 0.0))
    if total_box_vol <= EPS:
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    # Adaptive sample size: prioritize quality but keep runtime bounded
    base = 2500
    s_by_k = 500 * k
    s_by_dim = 600 * max(1, D)
    S = int(min(45000, max(base, s_by_k, s_by_dim)))
    S = max(1200, S)

    rng = np.random.default_rng()

    # Latin Hypercube Sampling (LHS) for lower-variance MC estimate
    # create S stratified positions per dimension
    u = (np.arange(S) + rng.random(S)) / float(S)  # base vector
    samples_unit = np.empty((S, D), dtype=float)
    for dim in range(D):
        perm = rng.permutation(S)
        samples_unit[:, dim] = u[perm]
    # scale to global bounding box; avoid zero span dims
    span = np.where(box_sizes > 0, box_sizes, 1.0)
    samples = samples_unit * span + global_low

    # Precompute inclusion S x N matrix in chunks to control memory
    max_cols_per_chunk = 2000
    inside = np.zeros((S, N), dtype=bool)
    start = 0
    while start < N:
        end = min(N, start + max_cols_per_chunk)
        sl = slice(start, end)
        ge = samples[:, None, :] >= lows[sl][None, :, :]
        le = samples[:, None, :] <= highs[sl][None, :, :]
        inside[:, sl] = np.all(ge & le, axis=2)
        start = end

    # Candidate pruning: remove boxes that cover zero samples (no contribution estimate)
    sample_counts = inside.sum(axis=0)
    nonzero_mask = sample_counts > 0
    nonzero_idx = np.nonzero(nonzero_mask)[0]
    zero_idx = np.nonzero(~nonzero_mask)[0].tolist()

    if nonzero_idx.size == 0:
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    inside_p = inside[:, nonzero_idx]  # S x P
    vols_p = indiv_vols[nonzero_idx]
    P = inside_p.shape[1]

    # initial estimated marginal volumes (from samples)
    counts_p = inside_p.sum(axis=0).astype(float)  # shape (P,)
    est_marginals = (counts_p / float(S)) * total_box_vol

    # Score components: log-volume bias and diversity penalty base
    bias_alpha = 0.6
    log_scale = 6.0
    max_vol = vols_p.max() if vols_p.size > 0 else 0.0
    vol_norm = (vols_p / max_vol) if max_vol > 0 else np.zeros_like(vols_p)
    vol_factor = 1.0 + bias_alpha * np.log1p(vol_norm * log_scale)

    # Precompute an ordering of candidates by est_marginals * vol_factor to speed initial heap
    initial_scores = est_marginals * vol_factor

    # Build lazy heap: (-score, local_idx, last_update_round)
    heap = [(-float(initial_scores[i]), int(i)) for i in range(P)]
    heapq.heapify(heap)

    covered = np.zeros(S, dtype=bool)
    selected_local = []
    selected_set = set()

    tol = 1e-12
    # To avoid many tiny reinsertions, allow a small tolerance relative to score
    rel_tol = 1e-8

    # Greedy selection with CELF-style lazy updates
    while len(selected_local) < k and heap:
        neg_score, local_idx = heapq.heappop(heap)
        if local_idx in selected_set:
            continue
        popped_score = -neg_score
        candidate_mask = inside_p[:, local_idx]
        new_cov_mask = (~covered) & candidate_mask
        new_count = int(new_cov_mask.sum())
        cur_marg = (new_count / float(S)) * total_box_vol

        # if original count is 0 (shouldn't happen for nonzero candidates), guard division
        orig_count = counts_p[local_idx] if counts_p[local_idx] > 0 else 1.0
        # unique fraction among this candidate's samples that are not already covered
        unique_frac = (new_count / orig_count) if orig_count > 0 else 0.0

        # Diversity penalty: reward candidates that add unique coverage; use concave transform
        diversity_gamma = 0.9
        diversity_exponent = 0.9
        diversity_bonus = 1.0 + diversity_gamma * (unique_frac ** diversity_exponent)

        cur_score = cur_marg * vol_factor[local_idx] * diversity_bonus

        # If score stale (difference beyond tolerance), reinsert updated score
        if not np.isclose(cur_score, popped_score, atol=tol, rtol=rel_tol):
            if cur_marg <= EPS:
                # no effective marginal contribution => drop candidate
                continue
            heapq.heappush(heap, (-float(cur_score), int(local_idx)))
            continue

        # If marginal is negligible, stop
        if cur_marg <= EPS:
            break

        # Accept candidate
        selected_local.append(int(local_idx))
        selected_set.add(int(local_idx))
        covered |= candidate_mask

        # Early stop if all samples covered
        if covered.all():
            break

    # Fill remaining slots by highest individual volume among remaining (including zero-coverage candidates)
    if len(selected_local) < k:
        remaining_locals = [i for i in range(P) if i not in selected_set]
        need = k - len(selected_local)
        if remaining_locals:
            rem_vols = vols_p[remaining_locals]
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_local.append(int(remaining_locals[int(idx)]))
                selected_set.add(int(remaining_locals[int(idx)]))
        # if still short, pad with zero-coverage original indices
        if len(selected_local) < k:
            pad_needed = k - len(selected_local)
            pads = []
            for z in zero_idx:
                if z not in pads:
                    pads.append(z)
                    if len(pads) >= pad_needed:
                        break
            if len(pads) < pad_needed:
                for i in range(N):
                    if i not in nonzero_idx[selected_local] and i not in pads:
                        pads.append(i)
                        if len(pads) >= pad_needed:
                            break
            # Map pad indices that are from zero_idx are original indices; else map remaining original
            for p in pads[:pad_needed]:
                # if this p refers to original index within nonzero_idx? zero_idx are original
                if p in nonzero_idx:
                    # find local index
                    loc = int(np.where(nonzero_idx == p)[0][0])
                    selected_local.append(loc)
                    selected_set.add(loc)
                else:
                    # p is an original index not in nonzero_idx; append sentinel by mapping to original
                    # We'll handle by extending selected_original later
                    # mark by negative original index (to include later)
                    selected_local.append(('orig', int(p)))

    # Map selected_local (which may contain ('orig', idx)) to original indices
    selected_original = []
    for s in selected_local:
        if isinstance(s, tuple) and s[0] == 'orig':
            selected_original.append(int(s[1]))
        else:
            # local index -> original index via nonzero_idx
            selected_original.append(int(nonzero_idx[int(s)]))

    # Ensure uniqueness and length k: if still fewer than k, pad with remaining original indices by volume
    selected_original = list(dict.fromkeys(selected_original))  # preserve order, unique
    if len(selected_original) < k:
        # pick remaining by highest indiv_vols not already chosen
        need = k - len(selected_original)
        remaining = [i for i in range(N) if i not in selected_original]
        if remaining:
            rem_vols = indiv_vols[remaining]
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_original.append(int(remaining[int(idx)]))

    selected_original = np.array(selected_original[:k], dtype=int)
    subset = points[selected_original, :].copy()
    return subset


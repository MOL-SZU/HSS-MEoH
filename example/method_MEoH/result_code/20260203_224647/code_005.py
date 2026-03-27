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

    # If all volumes are zero, return first k points (degenerate)
    if np.all(indiv_vols <= EPS):
        take = min(k, N)
        return points[:take].copy()

    # If k >= N, just return all points (no selection needed)
    if k >= N:
        return points.copy()

    # Sampling region: must cover union of all boxes -> use global lows/highs
    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    box_sizes = global_high - global_low
    total_box_vol = np.prod(np.maximum(box_sizes, 0.0))
    if total_box_vol <= EPS:
        # Degenerate bounding box: fallback to highest individual volumes
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    # Adaptive Monte Carlo sample count (balanced for quality and speed)
    base = 2000
    s_by_k = 400 * k
    s_by_dim = 800 * max(1, D)
    S = int(min(40000, max(base, s_by_k, s_by_dim)))
    S = max(1200, S)

    rng = np.random.default_rng()
    # Avoid zero-size dimensions for sampling; in those dims samples are constant global_low
    span = np.where(box_sizes > 0, box_sizes, 1.0)
    samples = rng.random((S, D)) * span + global_low

    # Precompute inclusion matrix: samples in each candidate box (S x N) boolean
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

    # Candidate pruning: drop candidates with zero sample coverage (they won't contribute)
    sample_counts = inside.sum(axis=0)  # per candidate how many samples lie in its box
    nonzero_mask = sample_counts > 0
    nonzero_idx = np.nonzero(nonzero_mask)[0]
    zero_idx = np.nonzero(~nonzero_mask)[0].tolist()

    if nonzero_idx.size == 0:
        # No candidate covers any sample (unlikely) => fall back to top individual volumes
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    # Work only with nonzero candidates to save time
    inside_p = inside[:, nonzero_idx]  # S x P
    vols_p = indiv_vols[nonzero_idx]
    P = inside_p.shape[1]

    # Initial estimated marginal volumes based on sample counts
    counts_p = inside_p.sum(axis=0).astype(float)  # shape (P,)
    est_marginals = (counts_p / float(S)) * total_box_vol

    # New scoring: log-scaled volume bias + uniqueness bonus to encourage diverse coverage
    # Parameters chosen to moderately favor larger volumes while rewarding unique coverage
    bias_alpha = 0.65   # strength of log-scaled volume bias
    log_scale = 8.0     # scaling inside log1p for sensitivity
    uniq_beta = 0.55    # uniqueness bonus strength
    uniq_exp = 1.15     # uniqueness exponent

    max_vol = vols_p.max() if vols_p.size > 0 else 0.0
    if max_vol > 0:
        vol_norm = vols_p / max_vol
    else:
        vol_norm = np.zeros_like(vols_p)

    # Log-scaled factor: 1 + bias_alpha * log1p(vol_norm * log_scale)
    combined_factor = 1.0 + bias_alpha * np.log1p(vol_norm * log_scale)

    # Initial uniqueness at start is 1 (new_count == counts_p), so uniqueness_bonus = 1 + uniq_beta
    uniqueness_base = 1.0 + uniq_beta
    initial_scores = est_marginals * combined_factor * uniqueness_base

    # Build lazy heap: (-score, local_idx)
    heap = [(-float(initial_scores[i]), int(i)) for i in range(P)]
    heapq.heapify(heap)

    covered = np.zeros(S, dtype=bool)
    selected_local = []
    selected_set = set()

    tol = 1e-12
    while len(selected_local) < k and heap:
        neg_score, local_idx = heapq.heappop(heap)
        if local_idx in selected_set:
            continue
        popped_score = -neg_score
        # recompute actual marginal on current covered set (count of samples newly covered)
        candidate_mask = inside_p[:, local_idx]
        new_cov = (~covered) & candidate_mask
        new_count = int(new_cov.sum())
        cur_marg = (new_count / float(S)) * total_box_vol
        # uniqueness relative to original estimated coverage (avoid division by zero)
        orig_count = counts_p[local_idx] if counts_p[local_idx] > 0 else 1.0
        uniqueness = (new_count / orig_count) if orig_count > 0 else 0.0
        uniqueness_bonus = 1.0 + uniq_beta * (uniqueness ** uniq_exp)
        # recompute current score with combined factor (unchanged) and uniqueness bonus
        cur_score = cur_marg * combined_factor[local_idx] * uniqueness_bonus
        # if popped score stale, push updated and continue
        if not np.isclose(cur_score, popped_score, atol=tol, rtol=1e-9):
            # If the current marginal is effectively zero, no benefit to reinsert
            if cur_marg <= EPS:
                continue
            heapq.heappush(heap, (-float(cur_score), int(local_idx)))
            continue
        # if current marginal is effectively zero, no further gains
        if cur_marg <= EPS:
            break
        # accept this candidate
        selected_local.append(int(local_idx))
        selected_set.add(int(local_idx))
        # update covered samples
        covered |= candidate_mask

        # early stop: if all samples covered, we can stop
        if covered.all():
            break

    # If selected fewer than k, fill remaining by top individual volumes among remaining non-selected
    if len(selected_local) < k:
        remaining = [i for i in range(P) if i not in selected_set]
        if remaining:
            rem_vols = vols_p[remaining]
            need = k - len(selected_local)
            order = np.argsort(-rem_vols)[:need]
            for idx in order:
                selected_local.append(int(remaining[int(idx)]))
                selected_set.add(int(remaining[int(idx)]))

    # Map selected local indices back to original indices
    selected_original = list(nonzero_idx[selected_local])

    # If still fewer than k (e.g., P < k), pad with zero-coverage candidates or others
    if len(selected_original) < k:
        pad_needed = k - len(selected_original)
        pads = []
        for z in zero_idx:
            if z not in selected_original:
                pads.append(z)
                if len(pads) >= pad_needed:
                    break
        if len(pads) < pad_needed:
            for i in range(N):
                if i not in selected_original and i not in pads:
                    pads.append(i)
                    if len(pads) >= pad_needed:
                        break
        selected_original.extend(pads[:pad_needed])

    selected_original = np.array(selected_original[:k], dtype=int)
    subset = points[selected_original, :].copy()
    return subset


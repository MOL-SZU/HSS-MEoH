import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape
    if k <= 0:
        return np.zeros((0, D))
    if k >= N:
        return points.copy()

    if reference_point is None:
        reference_point = points.max(axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Ensure reference_point is at least slightly worse than points (for minimization convention)
    # If some dims reference <= points.min, expand slightly to avoid degenerate zero-volume sampling space.
    pts_min = points.min(axis=0)
    eps = 1e-9
    ref = reference_point.copy()
    mask_too_small = ref <= pts_min + eps
    if np.any(mask_too_small):
        # Expand those dims
        ref[mask_too_small] = pts_min[mask_too_small] + 1e-6 + 0.1 * np.abs(pts_min[mask_too_small]) + eps

    # Monte Carlo sample size (trade-off between accuracy and speed)
    M = int(min(20000, max(2000, 400 * k)))  # caps at 20k, grows with k for better marginal estimates
    rng = np.random.RandomState(0)  # deterministic for reproducibility

    low = pts_min  # lower bound for sampling (best observed values)
    high = ref

    # If any dimension has zero range, perturb slightly to avoid degenerate sampling
    zero_range = (high - low) <= 0
    if np.any(zero_range):
        high = high.copy()
        low = low.copy()
        high[zero_range] = low[zero_range] + 1e-6 + 0.1 * (np.abs(low[zero_range]) + 1.0)

    # Draw samples uniformly in the hyper-rectangle [low, high]
    samples = low + rng.rand(M, D) * (high - low)

    # For minimization: a point p dominates a sample s if p <= s componentwise
    # Build boolean dominated matrix of shape (N, M)
    # To save memory, compute in blocks if N*M too big
    max_bool_entries = 100_000_000  # cap ~100M booleans (~100MB)
    dominated = np.zeros((N, M), dtype=bool)
    if N * M <= max_bool_entries:
        dominated = (points[:, None, :] <= samples[None, :, :]).all(axis=2)
    else:
        # block over points
        block = int(max(1, max_bool_entries // M))
        i = 0
        while i < N:
            j = min(N, i + block)
            dominated[i:j, :] = (points[i:j, None, :] <= samples[None, :, :]).all(axis=2)
            i = j

    # Only samples that are dominated by at least one candidate matter
    sample_covered_by_any = dominated.any(axis=0)
    if not np.any(sample_covered_by_any):
        # No sample is dominated -> fallback to selecting by box-volume (prod of (ref - p))
        volumes = np.prod(np.maximum(0.0, high - points), axis=1)
        idx = np.argsort(-volumes)[:k]
        return points[idx].copy()

    # Restrict to relevant samples to save work
    relevant_idx = np.where(sample_covered_by_any)[0]
    samples = samples[relevant_idx]  # not used further except for shape
    dominated = dominated[:, relevant_idx]
    M_rel = dominated.shape[1]

    selected = []
    selected_mask = np.zeros(N, dtype=bool)
    covered_by_selected = np.zeros(M_rel, dtype=bool)

    # Precompute box volumes as a cheap tie-breaker / fallback
    box_volumes = np.prod(np.maximum(0.0, high - points), axis=1)

    for _ in range(k):
        # Compute marginal contributions as number of currently-uncovered samples dominated by candidate
        # For speed we compute counts only for unselected candidates
        remaining = np.where(~selected_mask)[0]
        if remaining.size == 0:
            break

        # marginal counts
        # dominated[remaining] is shape (R, M_rel)
        # Use vectorized dot with inverted covered mask
        not_covered = ~covered_by_selected
        if not np.any(not_covered):
            # everything already covered by selected (rare). Choose by box volume among remaining.
            rem_vols = box_volumes[remaining]
            chosen = remaining[np.argmax(rem_vols)]
            selected.append(chosen)
            selected_mask[chosen] = True
            covered_by_selected |= dominated[chosen]
            continue

        # Compute marginal counts
        # For large arrays, compute in blocks to save memory
        # Using sum over axis=1
        marginal_counts = np.empty(remaining.shape[0], dtype=int)
        block_size = 256  # tune to memory/CPU tradeoff
        r = remaining
        for i in range(0, r.size, block_size):
            rb = r[i:i+block_size]
            # (rb.size, M_rel) boolean & not_covered
            # compute counts
            marginal_counts[i:i+rb.size] = (dominated[rb][:, not_covered].sum(axis=1))

        # If all marginal counts zero (due to sampling granularity), fall back to approximate
        if marginal_counts.max() == 0:
            # Fall back: choose point that maximizes box volume not already selected
            rem_vols = box_volumes[remaining]
            # tie-breaker: pick one closest to improving coverage: i.e., minimal sum of distances to reference (smaller is better)
            chosen_rel_idx = np.argmax(rem_vols)
            chosen = remaining[chosen_rel_idx]
            selected.append(chosen)
            selected_mask[chosen] = True
            covered_by_selected |= dominated[chosen]
            continue

        # Choose candidate with max marginal count; tie-breaker uses box volume then lexicographic smaller objectives
        max_count = marginal_counts.max()
        cand_idx = np.where(marginal_counts == max_count)[0]
        if cand_idx.size == 1:
            chosen = remaining[cand_idx[0]]
        else:
            # tie-break by box volume
            tie_remaining = remaining[cand_idx]
            tv = box_volumes[tie_remaining]
            best = np.argmax(tv)
            chosen = tie_remaining[best]

        selected.append(chosen)
        selected_mask[chosen] = True
        covered_by_selected |= dominated[chosen]

        # early stop if we've covered all relevant samples
        if covered_by_selected.all():
            # fill remaining slots by box-volume among remaining points
            rem_left = np.where(~selected_mask)[0]
            if rem_left.size > 0:
                rem_needed = k - len(selected)
                if rem_needed > 0:
                    order = np.argsort(-box_volumes[rem_left])[:rem_needed]
                    for idx_rel in order:
                        sel_idx = rem_left[idx_rel]
                        selected.append(sel_idx)
                        selected_mask[sel_idx] = True
            break

    selected = np.array(selected, dtype=int)[:k]
    return points[selected].copy()


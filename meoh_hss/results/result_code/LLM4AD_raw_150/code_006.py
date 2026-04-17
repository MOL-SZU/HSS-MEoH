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

    # Adaptive Monte Carlo sample count emphasizing dimension and k (slightly reduced for speed)
    base = 2500
    s_by_k = 500 * k
    s_by_dim = 1000 * max(1, D)
    S = int(min(35000, max(base, s_by_k, s_by_dim)))
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

    # New scoring: hybrid multiplicative+additive diversity bias with different params
    # Stronger emphasis on larger individual volumes with two exponents
    diversity_mul = 0.35   # multiplicative diversity strength (increased)
    beta = 0.60            # multiplicative exponent (higher)
    additive_scale = 0.45  # additive-style bias scale
    gamma = 0.90           # additive exponent

    max_vol = vols_p.max() if vols_p.size > 0 else 0.0
    if max_vol > 0:
        vol_norm = vols_p / max_vol
    else:
        vol_norm = np.zeros_like(vols_p)

    # Combined factor: 1 + diversity_mul * vol_norm^beta + additive_scale * vol_norm^gamma
    combined_factor = 1.0 + diversity_mul * (vol_norm ** beta) + additive_scale * (vol_norm ** gamma)

    # Initial scores (hybrid)
    scores = est_marginals * combined_factor

    # Build lazy heap: (-score, local_idx)
    heap = [(-float(scores[i]), int(i)) for i in range(P)]
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
        new_cov = (~covered) & inside_p[:, local_idx]
        new_count = int(new_cov.sum())
        cur_marg = (new_count / float(S)) * total_box_vol
        # recompute combined factor (unchanged per candidate) but used to form cur_score
        cur_score = cur_marg * (1.0 + diversity_mul * (vol_norm[local_idx] ** beta) + additive_scale * (vol_norm[local_idx] ** gamma))
        # if popped score stale, push updated and continue
        if not np.isclose(cur_score, popped_score, atol=tol, rtol=1e-9):
            heapq.heappush(heap, (-float(cur_score), int(local_idx)))
            continue
        # if current marginal is effectively zero, no further gains
        if cur_marg <= EPS:
            break
        # accept this candidate
        selected_local.append(int(local_idx))
        selected_set.add(int(local_idx))
        # update covered samples
        covered |= inside_p[:, local_idx]

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

    # If still fewer than k (e.g., P < k), pad with zero-volume candidates or others
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


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

    # Initial Monte-Carlo sample budget parameters (adaptive allowed)
    S = int(min(30000, max(2000, 1000 * k, 400 * max(1, D))))
    S = max(1000, S)
    S_max = 30000

    rng = np.random.default_rng()

    # Samples generation helper (handles zero-size dims)
    def gen_samples(n):
        span = np.where(box_sizes > 0, box_sizes, 1.0)
        return rng.random((n, D)) * span + global_low

    samples = gen_samples(S)

    # Build inclusion boolean matrix in chunks to avoid memory bursts
    # inside is S x N boolean
    def compute_inside(samples_array):
        sS = samples_array.shape[0]
        inside_mat = np.zeros((sS, N), dtype=bool)
        max_cols_per_chunk = 2000
        start = 0
        while start < N:
            end = min(N, start + max_cols_per_chunk)
            sl = slice(start, end)
            ge = samples_array[:, None, :] >= lows[sl][None, :, :]
            le = samples_array[:, None, :] <= highs[sl][None, :, :]
            inside_mat[:, sl] = np.all(ge & le, axis=2)
            start = end
        return inside_mat

    inside = compute_inside(samples)  # shape (S, N)

    # Candidate pruning: drop candidates with zero sample coverage
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

    # Initial estimates
    counts_p = inside_p.sum(axis=0).astype(float)  # per-candidate sample hits
    est_marg = (counts_p / float(S)) * total_box_vol

    # Bias: encourage larger individual volumes moderately (different exponent than original)
    bias_scale = 0.04
    beta = 0.7
    max_vol = vols_p.max() if vols_p.size > 0 else 0.0
    vol_norm = (vols_p / max_vol) if max_vol > 0 else np.zeros_like(vols_p)
    bias_vals = bias_scale * total_box_vol * (vol_norm ** beta)

    # Diversity penalty parameters: penalize candidates near centroid of selected set
    diversity_scale = 0.06  # relative to total_box_vol
    diversity_power = 1.2

    # Precompute candidate positions for centroid/distance calc
    cand_points = points[nonzero_idx]

    # Build initial scores
    scores = est_marg + bias_vals  # no diversity term initially

    # Build lazy heap: (-score, local_idx)
    heap = [(-float(scores[i]), int(i)) for i in range(P)]
    heapq.heapify(heap)

    covered = np.zeros(S, dtype=bool)
    selected_local = []
    selected_set = set()
    selected_points = []

    tol = 1e-12

    # helper to compute diversity penalty given candidate and current centroid info
    def compute_diversity_penalty(candidate_idx, centroid, max_d):
        if centroid is None or max_d <= 0:
            return 0.0
        # distance normalized in [0,1]
        dist = np.linalg.norm(cand_points[candidate_idx] - centroid)
        dist_norm = min(1.0, dist / max_d)
        # higher distance => smaller penalty; we want to penalize near neighbors => penalty ~ (1-dist_norm)^p
        return diversity_scale * total_box_vol * ((1.0 - dist_norm) ** diversity_power)

    # precompute diameter for normalization
    if P > 1:
        # approximate diameter by max pairwise in candidate set using mins/maxs
        mins = cand_points.min(axis=0)
        maxs = cand_points.max(axis=0)
        max_diameter = np.linalg.norm(maxs - mins)
    else:
        max_diameter = 0.0

    # Lazy greedy with optional adaptive resampling
    while len(selected_local) < k and heap:
        neg_score, local_idx = heapq.heappop(heap)
        if local_idx in selected_set:
            continue
        popped_score = -neg_score

        # recompute exact marginal on current covered set
        new_cov = (~covered) & inside_p[:, local_idx]
        new_count = int(new_cov.sum())
        cur_marg = (new_count / float(S)) * total_box_vol

        # compute current diversity penalty
        if selected_points:
            centroid = np.mean(np.vstack(selected_points), axis=0)
        else:
            centroid = None
        div_pen = compute_diversity_penalty(local_idx, centroid, max_diameter)
        cur_score = cur_marg + bias_vals[local_idx] - div_pen

        # if popped score stale, push updated and continue
        if not np.isclose(cur_score, popped_score, atol=tol, rtol=1e-9):
            heapq.heappush(heap, (-float(cur_score), int(local_idx)))
            continue

        # if marginal is effectively zero, maybe try to refine estimates by adding samples
        small_gain_thresh = 1e-3 * total_box_vol  # if top marginal under this, we may refine
        if cur_marg <= EPS and S < S_max:
            # add samples to improve resolution
            add_S = min(S * 2, S_max) - S
            if add_S > 0:
                extra_samples = gen_samples(add_S)
                extra_inside = compute_inside(extra_samples)
                # append
                inside = np.vstack([inside, extra_inside])
                samples = np.vstack([samples, extra_samples])
                inside_p = inside[:, nonzero_idx]
                S = inside.shape[0]
                # recompute covered and counts_p
                covered = np.zeros(S, dtype=bool)
                for sel in selected_local:
                    covered |= inside_p[:, sel]
                counts_p = inside_p.sum(axis=0).astype(float)
                est_marg = (counts_p / float(S)) * total_box_vol
                scores = est_marg + bias_vals
                # rebuild heap with updated scores for non-selected candidates
                heap = [(-float(scores[i]), int(i)) for i in range(P) if i not in selected_set]
                heapq.heapify(heap)
                continue
            else:
                break

        # if marginal is effectively zero, stop selecting more (no gain)
        if cur_marg <= EPS:
            break

        # accept this candidate
        selected_local.append(int(local_idx))
        selected_set.add(int(local_idx))
        selected_points.append(cand_points[local_idx])

        # update coverage
        covered |= inside_p[:, local_idx]

        # if all samples covered, can stop early
        if covered.all():
            break

        # update counts for remaining candidates (decrement by newly covered samples)
        # Efficient update: subtract newly covered sample mask from counts
        newly_covered_mask = new_cov  # boolean for samples newly covered by this accepted point
        if newly_covered_mask.any():
            # Subtract newly covered contributions from counts_p
            # Convert mask to int and sum along axis for the candidates
            reduce = inside_p[newly_covered_mask, :].sum(axis=0)
            counts_p = counts_p - reduce
            # Ensure non-negative
            counts_p = np.maximum(counts_p, 0.0)
            est_marg = (counts_p / float(S)) * total_box_vol
            # recompute scores for remaining candidates with updated diversity penalties
            if selected_points:
                centroid = np.mean(np.vstack(selected_points), axis=0)
            else:
                centroid = None
            # Update scores array but keep previously selected ones irrelevant
            for i in range(P):
                if i in selected_set:
                    continue
                div_pen = compute_diversity_penalty(i, centroid, max_diameter)
                scores[i] = est_marg[i] + bias_vals[i] - div_pen
            # rebuild heap lazily to reflect changes (could be kept more incremental)
            heap = [(-float(scores[i]), int(i)) for i in range(P) if i not in selected_set]
            heapq.heapify(heap)

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


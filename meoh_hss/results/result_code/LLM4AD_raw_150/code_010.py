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

    # Choose number of Monte Carlo samples adaptively
    # tradeoff: more samples -> better estimates; cap for memory/time
    base = 2000
    s_by_k = 200 * k
    s_by_dim = int(800 * max(1, D // 3))
    S = int(min(20000, max(base, s_by_k, s_by_dim)))
    # Ensure at least a modest number
    S = max(1000, S)

    rng = np.random.default_rng()
    # Avoid zero-size dimensions for sampling; in those dims samples are constant global_low
    span = np.where(box_sizes > 0, box_sizes, 1.0)
    samples = rng.random((S, D)) * span + global_low

    # Precompute inclusion matrix: samples in each candidate box (S x N) boolean
    # sample in box i iff for all dims: samples[:,d] between lows[i,d] and highs[i,d]
    # Do in vectorized manner; may allocate (S,N,D) temporarily but manageable for S,N moderate.
    # To reduce memory, compute comparisons in two steps and combine.
    # Compute mask_low = samples[:, None, :] >= lows[None, :, :]
    # Compute mask_high = samples[:, None, :] <= highs[None, :, :]
    # Then inside = np.all(mask_low & mask_high, axis=2)
    # We'll compute in chunks if N large to save peak memory.
    max_cols_per_chunk = 2000  # tune to avoid huge memory in extreme N
    inside = np.zeros((S, N), dtype=bool)
    start = 0
    while start < N:
        end = min(N, start + max_cols_per_chunk)
        sl = slice(start, end)
        # shape (S, end-start, D)
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

    # Build lazy heap: (-est_marginal, local_idx)
    heap = [(-float(est_marginals[i]), int(i)) for i in range(P)]
    heapq.heapify(heap)

    covered = np.zeros(S, dtype=bool)
    selected_local = []
    selected_set = set()

    # Greedy selection with lazy updates using exact (on samples) recomputation
    tol = 1e-12
    while len(selected_local) < k and heap:
        neg_est, local_idx = heapq.heappop(heap)
        if local_idx in selected_set:
            continue
        popped_est = -neg_est
        # recompute actual marginal on current covered set (count of samples newly covered)
        new_cov = (~covered) & inside_p[:, local_idx]
        new_count = int(new_cov.sum())
        cur_marg = (new_count / float(S)) * total_box_vol
        # if popped estimate stale, push updated and continue
        if not np.isclose(cur_marg, popped_est, atol=tol, rtol=1e-9):
            # push updated value (if zero, it may be ignored later)
            heapq.heappush(heap, (-float(cur_marg), int(local_idx)))
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
        # prefer zero-volume points first
        pads = []
        for z in zero_idx:
            if z not in selected_original:
                pads.append(z)
                if len(pads) >= pad_needed:
                    break
        if len(pads) < pad_needed:
            # add other remaining original indices
            for i in range(N):
                if i not in selected_original and i not in pads:
                    pads.append(i)
                    if len(pads) >= pad_needed:
                        break
        selected_original.extend(pads[:pad_needed])

    selected_original = np.array(selected_original[:k], dtype=int)
    subset = points[selected_original, :].copy()
    return subset


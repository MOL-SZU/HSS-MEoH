import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape

    if k <= 0:
        return np.empty((0, D), dtype=float)
    if k >= N:
        return points.copy()

    if reference_point is None:
        reference_point = points.max(axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Compute positive extents (map boxes to origin-anchored boxes)
    deltas = np.abs(points - reference_point)  # shape (N, D)
    eps = 1e-12
    max_delta = np.maximum(deltas.max(axis=0), eps)

    # analytic volumes
    vols = np.maximum(deltas.prod(axis=1), 0.0)
    max_vol = max(vols.max(), eps)

    # Phase 0: prefilter candidates by analytic volume to reduce N for greedy
    # Keep at least k, but limit to a moderate multiple of k
    prefilter_mult = 5
    top_m = min(N, max(k, min(N, prefilter_mult * k)))
    # Also allow a fraction of N if that yields more candidates for large N
    top_m = min(N, max(top_m, int(min(N, max(10, 0.15 * N)))))
    # Select indices with largest analytic volumes
    order_all = np.argsort(-vols)
    cand_idx = order_all[:top_m]
    M = cand_idx.size
    if M <= k:
        selected_idx = list(cand_idx[:k].astype(int))
        return points[np.array(selected_idx, dtype=int), :].copy()

    # Sampling: Latin Hypercube Sampling (LHS) in [0, max_delta] to estimate union
    rng = np.random.default_rng()
    # choose sample count scaled by D and k but clipped
    samples = int(np.clip(1500 + 400 * D + 60 * k, 1000, 40000))
    s = samples

    # Generate LHS base for s points
    # For each dim, create s strata and permute
    base = (np.arange(s, dtype=float) + rng.random(s)) / float(s)
    sample_points = np.empty((s, D), dtype=float)
    for dim in range(D):
        perm = rng.permutation(s)
        sample_points[:, dim] = base[perm]
    # scale to actual extents
    sample_points = sample_points * max_delta[None, :]

    # Precompute coverage matrix for only candidate indices
    cand_deltas = deltas[cand_idx, :]  # shape (M, D)
    # coverage_samples x candidates: True if sample <= cand_deltas
    # We compute by broadcasting
    coverage = (sample_points[:, None, :] <= cand_deltas[None, :, :]).all(axis=2)  # shape (s, M)

    # Precompute cell volume for mapping sample counts to volume estimate
    total_region_vol = np.prod(max_delta)
    cell_vol = total_region_vol / float(s)

    # Score mixing parameter: combine sampled marginal count with analytic volume
    beta = float(min(6.0, max(0.3, np.log1p(D) * 1.2)))

    # Initial covered mask and marginal counts
    covered = np.zeros(s, dtype=bool)
    not_covered = ~covered
    marginal_counts = (not_covered.astype(int)[:, None] * coverage.astype(int)).sum(axis=0)  # length M

    # Initial scores
    scores = marginal_counts + beta * (vols[cand_idx] / max_vol)

    # Lazy-greedy using max-heap
    heap = []
    for local_i in range(M):
        heap.append((-float(scores[local_i]), int(local_i), int(marginal_counts[local_i])))
    heapq.heapify(heap)

    selected_local_idx = []
    selected_local_set = set()

    # Greedy selection loop
    while len(selected_local_idx) < k and heap:
        neg_score, local_i, cached_marg = heapq.heappop(heap)
        if local_i in selected_local_set:
            continue
        # recompute true marginal for this candidate given current covered mask
        true_marg = int(((~covered) & coverage[:, local_i]).sum())
        if true_marg == cached_marg:
            if true_marg == 0:
                # no new samples covered, stop greedy early
                break
            selected_local_idx.append(int(local_i))
            selected_local_set.add(int(local_i))
            covered |= coverage[:, local_i]
            # we do not update all marginals eagerly; lazy updates handle it
        else:
            new_score = true_marg + beta * (vols[cand_idx[local_i]] / max_vol)
            heapq.heappush(heap, (-float(new_score), int(local_i), int(true_marg)))

    # If selected fewer than k, fill from remaining candidates by analytic volume
    if len(selected_local_idx) < k:
        remaining_local = [i for i in range(M) if i not in selected_local_set]
        if remaining_local:
            rem_vols = vols[cand_idx[remaining_local]]
            order = np.argsort(-rem_vols)
            need = k - len(selected_local_idx)
            for oi in order[:need]:
                selected_local_idx.append(int(remaining_local[int(oi)]))
                selected_local_set.add(int(remaining_local[int(oi)]))

    # Map local indices back to original indices
    selected_idx = [int(cand_idx[i]) for i in selected_local_idx[:k]]
    selected_set = set(selected_idx)

    # Compute sample-based coverage counts for current selection
    sel_local_inds = [int(np.where(cand_idx == si)[0][0]) for si in selected_idx]  # indices in cand_idx
    counts_per_sample = coverage[:, sel_local_inds].sum(axis=1)  # integer counts per sample
    best_count = int((counts_per_sample > 0).sum())

    # Local improvement: deterministic best swaps among candidate pool limited by budget
    # For each selected element, try replacing by top-L non-selected candidates (by vol)
    max_swaps = min(200, 40 * k)
    swaps = 0
    # Precompute candidate ordering by analytic volume (descending)
    cand_vols = vols[cand_idx]
    cand_order_by_vol = np.argsort(-cand_vols)
    # Non-selected local indices
    non_selected_local = [i for i in range(M) if cand_idx[i] not in selected_set]
    # Limit list of replacement candidates tried per swap to top_R
    top_R = min(len(non_selected_local), max(50, 8 * k))

    while swaps < max_swaps:
        improved = False
        # iterate over selected items in deterministic order (by least marginal maybe)
        # choose the selected item with smallest analytic volume contribution first to try replace
        sel_local_by_vol = sorted(sel_local_inds, key=lambda li: cand_vols[li])
        for s_local in sel_local_by_vol:
            # candidate replacements: iterate non-selected top_R by vol
            tried = 0
            for u_local in cand_order_by_vol:
                if tried >= top_R:
                    break
                if u_local in sel_local_inds:
                    continue
                tried += 1
                # compute new counts_per_sample if we remove s_local and add u_local
                # new_counts = counts_per_sample - coverage[:, s_local].astype(int) + coverage[:, u_local].astype(int)
                # use integer arrays for speed
                new_counts = counts_per_sample - coverage[:, s_local].astype(int) + coverage[:, u_local].astype(int)
                new_count = int((new_counts > 0).sum())
                if new_count > best_count:
                    # accept swap: update data structures
                    swaps += 1
                    improved = True
                    best_count = new_count
                    # update selection lists
                    # replace s_local in sel_local_inds with u_local
                    idx_pos = sel_local_inds.index(s_local)
                    sel_local_inds[idx_pos] = int(u_local)
                    # update selected_idx mapping
                    selected_idx[idx_pos] = int(cand_idx[int(u_local)])
                    selected_set = set(selected_idx)
                    # update counts_per_sample
                    counts_per_sample = new_counts
                    break  # break out of candidate loop for this s_local
            if improved:
                break  # restart scanning selected items
        if not improved:
            break

    # Ensure exactly k indices
    selected_idx = selected_idx[:k]
    # Final subset
    subset = points[np.array(selected_idx, dtype=int), :].copy()
    return subset


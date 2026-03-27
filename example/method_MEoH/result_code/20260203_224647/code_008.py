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

    # Axis-aligned boxes per point between point and reference
    lows = np.minimum(points, reference_point)
    highs = np.maximum(points, reference_point)
    diffs = np.maximum(highs - lows, 0.0)
    indiv_vols = np.prod(diffs, axis=1)

    # Degenerate / trivial cases
    if np.all(indiv_vols <= EPS):
        take = min(k, N)
        return points[:take].copy()
    if k >= N:
        return points.copy()

    # Global sampling box
    global_low = np.min(lows, axis=0)
    global_high = np.max(highs, axis=0)
    box_sizes = global_high - global_low
    total_box_vol = np.prod(np.maximum(box_sizes, 0.0))
    if total_box_vol <= EPS:
        order = np.argsort(-indiv_vols)[:k]
        return points[order].copy()

    # Adaptive sample size (smaller baseline than original for speed)
    base = 1200
    s_by_k = 300 * k
    s_by_dim = 300 * max(1, D)
    S = int(min(40000, max(base, s_by_k, s_by_dim)))
    S = max(800, S)

    rng = np.random.default_rng()

    # Latin Hypercube Sampling (low-variance MC)
    u = (np.arange(S) + rng.random(S)) / float(S)
    samples_unit = np.empty((S, D), dtype=float)
    for dim in range(D):
        perm = rng.permutation(S)
        samples_unit[:, dim] = u[perm]
    span = np.where(box_sizes > 0, box_sizes, 1.0)
    samples = samples_unit * span + global_low

    # Compute inclusion matrix in chunks to limit memory
    max_cols = 3000
    inside = np.zeros((S, N), dtype=bool)
    for start in range(0, N, max_cols):
        end = min(N, start + max_cols)
        ge = samples[:, None, :] >= lows[start:end][None, :, :]
        le = samples[:, None, :] <= highs[start:end][None, :, :]
        inside[:, start:end] = np.all(ge & le, axis=2)

    # Prune candidates with zero sampled coverage (they contribute nothing in MC estimate)
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

    counts_p = inside_p.sum(axis=0).astype(float)  # S-count per candidate
    est_marginals = (counts_p / float(S)) * total_box_vol

    # Score bias and diversity hyperparameters (different from original settings)
    bias_beta = 0.55
    log_scale = 5.0
    max_vol = vols_p.max() if vols_p.size > 0 else 0.0
    vol_norm = (vols_p / max_vol) if max_vol > 0 else np.zeros_like(vols_p)
    vol_bias = 1.0 + bias_beta * np.log1p(vol_norm * log_scale)

    # Initial scores
    initial_scores = est_marginals * vol_bias

    # Build lazy heap storing (-score, local_idx)
    heap = [(-float(initial_scores[i]), int(i)) for i in range(P)]
    heapq.heapify(heap)

    covered = np.zeros(S, dtype=bool)
    selected_local = []
    in_selected = np.zeros(P, dtype=bool)

    rel_tol = 1e-8
    atol = 1e-12

    # Greedy selection with lazy updates (CELF-like)
    while len(selected_local) < k and heap:
        neg_score, loc = heapq.heappop(heap)
        if in_selected[int(loc)]:
            continue
        popped_score = -neg_score

        candidate_mask = inside_p[:, loc]
        # new unique samples this candidate would add
        new_mask = (~covered) & candidate_mask
        new_count = int(new_mask.sum())
        cur_marg = (new_count / float(S)) * total_box_vol

        # fraction of this candidate's sampled mass that is unique
        orig_count = counts_p[loc] if counts_p[loc] > 0 else 1.0
        unique_frac = (new_count / orig_count) if orig_count > 0 else 0.0

        # diversity penalty (we reward uniqueness multiplicatively)
        gamma = 0.95
        diversity_mult = 1.0 + gamma * (unique_frac ** 0.9)

        cur_score = cur_marg * vol_bias[loc] * diversity_mult

        # Lazy check: if stale, reinsert updated score
        if not np.isclose(cur_score, popped_score, atol=atol, rtol=rel_tol):
            if cur_marg <= EPS:
                # negligible marginal -> drop
                continue
            heapq.heappush(heap, (-float(cur_score), int(loc)))
            continue

        # negligible marginal -> break
        if cur_marg <= EPS:
            break

        # accept candidate
        selected_local.append(int(loc))
        in_selected[int(loc)] = True
        covered |= candidate_mask

        if covered.all():
            break

    # If not enough selected, fill by best remaining estimated marginal or by individual volume
    if len(selected_local) < k:
        remaining = [i for i in range(P) if not in_selected[i]]
        need = k - len(selected_local)
        if remaining:
            rem_scores = est_marginals[remaining] * vol_bias[remaining]
            order = np.argsort(-rem_scores)[:need]
            for idx in order:
                selected_local.append(int(remaining[int(idx)]))
                in_selected[int(remaining[int(idx)])] = True

    # If still short, use zero-coverage originals by highest individual volume
    selected_original = [int(nonzero_idx[s]) for s in selected_local]
    if len(selected_original) < k:
        need = k - len(selected_original)
        # choose among zero_idx first
        if zero_idx:
            zero_order = np.argsort(-indiv_vols[zero_idx])[:need]
            for zpos in zero_order:
                selected_original.append(int(zero_idx[int(zpos)]))
                if len(selected_original) >= k:
                    break
        # if still short, choose from remaining originals by indiv_vols
        if len(selected_original) < k:
            remaining_orig = [i for i in range(N) if i not in selected_original]
            if remaining_orig:
                rem_vols = indiv_vols[remaining_orig]
                order = np.argsort(-rem_vols)[: (k - len(selected_original))]
                for idx in order:
                    selected_original.append(int(remaining_orig[int(idx)]))

    # Ensure uniqueness and exact length k
    seen = set()
    final_sel = []
    for idx in selected_original:
        if idx not in seen:
            final_sel.append(idx)
            seen.add(idx)
        if len(final_sel) >= k:
            break

    # As a last resort pad with first points
    if len(final_sel) < k:
        for i in range(N):
            if i not in seen:
                final_sel.append(i)
                seen.add(i)
            if len(final_sel) >= k:
                break

    final_sel = np.array(final_sel[:k], dtype=int)
    subset = points[final_sel, :].copy()
    return subset


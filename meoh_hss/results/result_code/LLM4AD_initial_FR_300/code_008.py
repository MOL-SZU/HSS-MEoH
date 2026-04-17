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

    # reference point handling
    if reference_point is None:
        reference_point = points.max(axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # ensure reference dominates all points (minimization)
    pts_min = points.min(axis=0)
    eps = 1e-9
    ref = reference_point.copy()
    mask = ref <= pts_min + eps
    if np.any(mask):
        ref[mask] = pts_min[mask] + 1e-6 + 0.1 * np.abs(pts_min[mask]) + eps

    # ---------- Monte‑Carlo sampling ----------
    # sample count adapts to k and dimensionality
    M = int(min(8000, max(800, int(150 * k / max(1, D)))))
    rng = np.random.RandomState(42)

    low = pts_min
    high = ref
    zero_range = (high - low) <= 0
    if np.any(zero_range):
        high = high.copy()
        low = low.copy()
        high[zero_range] = low[zero_range] + 1e-6 + 0.1 * (np.abs(low[zero_range]) + 1.0)

    samples = low + rng.rand(M, D) * (high - low)

    # ---------- domination masks ----------
    max_bool = 80_000_000  # ~80 M booleans
    if N * M <= max_bool:
        dominated = (points[:, None, :] <= samples[None, :, :]).all(axis=2)  # (N, M)
    else:
        block = max(1, max_bool // M)
        dominated = np.zeros((N, M), dtype=bool)
        for i in range(0, N, block):
            j = min(N, i + block)
            dominated[i:j] = (points[i:j, None, :] <= samples[None, :, :]).all(axis=2)

    # retain only samples covered by at least one point
    sample_used = dominated.any(axis=0)
    if not np.any(sample_used):
        # fallback: pure box‑volume ranking
        box_vol = np.prod(np.maximum(0.0, high - points), axis=1)
        idx = np.argsort(-box_vol)[:k]
        return points[idx].copy()

    samples = samples[sample_used]
    dominated = dominated[:, sample_used]
    M_used = samples.shape[0]

    # ---------- pre‑compute box volumes ----------
    box_volumes = np.prod(np.maximum(0.0, high - points), axis=1)
    # normalize to [0,1]
    bv_min, bv_max = box_volumes.min(), box_volumes.max()
    norm_box = (box_volumes - bv_min) / (bv_max - bv_min + eps)

    # ---------- greedy selection with multiplicative score ----------
    selected_idx = []
    selected_mask = np.zeros(N, dtype=bool)
    covered = np.zeros(M_used, dtype=bool)

    # initial marginal coverage counts
    marginal = dominated.sum(axis=1)

    # lazy‑heap: store (estimated_gain, idx); recompute when popped if stale
    import heapq
    heap = []
    for i in range(N):
        # score = marginal_i * norm_box_i  (multiplicative)
        est = marginal[i] * norm_box[i]
        heapq.heappush(heap, (-est, i))  # max‑heap via negative key

    while len(selected_idx) < k and heap:
        neg_est, idx = heapq.heappop(heap)
        if selected_mask[idx]:
            continue  # already chosen elsewhere
        # recompute true marginal after possible coverage updates
        true_marginal = (dominated[idx] & ~covered).sum()
        true_score = true_marginal * norm_box[idx]
        # if the estimate deviates, push updated value back
        if -neg_est != true_score:
            heapq.heappush(heap, (-true_score, idx))
            continue

        # accept this point
        selected_idx.append(idx)
        selected_mask[idx] = True
        newly_cov = dominated[idx] & ~covered
        if newly_cov.any():
            covered |= newly_cov

        # early stop if all samples already covered
        if covered.all():
            break

    # If we still need more points, fill with highest box‑volume
    if len(selected_idx) < k:
        remaining = np.where(~selected_mask)[0]
        need = k - len(selected_idx)
        extra = remaining[np.argsort(-box_volumes[remaining])[:need]]
        selected_idx.extend(extra.tolist())

    selected_idx = np.array(selected_idx[:k], dtype=int)
    return points[selected_idx].copy()


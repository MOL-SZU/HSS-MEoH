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

    # reference point
    if reference_point is None:
        reference_point = points.max(axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # ensure reference is strictly worse (larger) than any point (minimization)
    pts_min = points.min(axis=0)
    eps = 1e-9
    ref = reference_point.copy()
    mask = ref <= pts_min + eps
    if np.any(mask):
        ref[mask] = pts_min[mask] + 1e-6 + 0.1 * np.abs(pts_min[mask]) + eps

    # ---------- Monte‑Carlo sampling ----------
    # sample count adapts to k and dimensionality
    M = int(min(10000, max(1000, int(200 * k / max(1, D)))))
    rng = np.random.RandomState(0)

    low = pts_min
    high = ref
    zero_range = (high - low) <= 0
    if np.any(zero_range):
        high = high.copy()
        low = low.copy()
        high[zero_range] = low[zero_range] + 1e-6 + 0.1 * (np.abs(low[zero_range]) + 1.0)

    samples = low + rng.rand(M, D) * (high - low)

    # ---------- domination masks ----------
    max_bool = 100_000_000  # ~100 M booleans
    if N * M <= max_bool:
        dominated = (points[:, None, :] <= samples[None, :, :]).all(axis=2)  # (N, M)
    else:
        # block over points
        block = max(1, max_bool // M)
        dominated = np.zeros((N, M), dtype=bool)
        for i in range(0, N, block):
            j = min(N, i + block)
            dominated[i:j] = (points[i:j, None, :] <= samples[None, :, :]).all(axis=2)

    # keep only samples that any point dominates
    sample_used = dominated.any(axis=0)
    if not np.any(sample_used):
        # fallback to pure box‑volume ranking
        box_vol = np.prod(np.maximum(0.0, high - points), axis=1)
        idx = np.argsort(-box_vol)[:k]
        return points[idx].copy()

    samples = samples[sample_used]
    dominated = dominated[:, sample_used]
    M_used = samples.shape[0]

    # ---------- pre‑compute box volumes ----------
    box_volumes = np.prod(np.maximum(0.0, high - points), axis=1)
    # normalized for scoring
    bv_min, bv_max = box_volumes.min(), box_volumes.max()
    norm_box = (box_volumes - bv_min) / (bv_max - bv_min + eps)

    # ---------- greedy selection ----------
    alpha = 0.7  # weight for marginal contribution
    selected_idx = []
    selected_mask = np.zeros(N, dtype=bool)
    covered = np.zeros(M_used, dtype=bool)

    # initial marginal counts
    marginal = dominated.sum(axis=1)

    # heap‑like loop (simple recompute each iteration, still fast)
    for _ in range(k):
        # compute combined scores
        scores = alpha * marginal + (1.0 - alpha) * norm_box
        scores[selected_mask] = -np.inf  # ignore already chosen
        best = int(np.argmax(scores))
        selected_idx.append(best)
        selected_mask[best] = True

        # update coverage
        newly_covered = dominated[best] & ~covered
        if newly_covered.any():
            covered |= newly_covered
            # decrement marginal counts of remaining candidates for those samples
            # vectorized decrement
            affected = ~selected_mask
            if affected.any():
                # sum over newly covered samples for each remaining point
                dec = (dominated[affected][:, newly_covered].sum(axis=1)).astype(int)
                marginal[affected] -= dec

        # early stop if all samples covered
        if covered.all():
            break

    # if we stopped early, fill remaining slots by box‑volume ranking
    if len(selected_idx) < k:
        remaining = np.where(~selected_mask)[0]
        need = k - len(selected_idx)
        extra = remaining[np.argsort(-box_volumes[remaining])[:need]]
        selected_idx.extend(extra.tolist())

    selected_idx = np.array(selected_idx[:k], dtype=int)
    return points[selected_idx].copy()


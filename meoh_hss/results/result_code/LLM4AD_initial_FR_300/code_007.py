import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np

    pts_all = np.asarray(points, dtype=float)
    if pts_all.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts_all.shape

    if k <= 0:
        return np.zeros((0, D))
    if k >= N:
        return pts_all.copy()

    # reference point: ensure it's a little worse than any point (minimization)
    if reference_point is None:
        reference_point = pts_all.max(axis=0) * 1.1
    ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Guarantee reference is strictly worse than observed minima
    pts_min = pts_all.min(axis=0)
    eps = 1e-9
    ref = ref.copy()
    mask_too_small = ref <= pts_min + eps
    if np.any(mask_too_small):
        ref[mask_too_small] = pts_min[mask_too_small] + 1e-6 + 0.1 * (np.abs(pts_min[mask_too_small]) + 1.0)

    # Precompute box volumes for tie-breaking (safe non-negative)
    box_volumes = np.prod(np.maximum(0.0, ref - pts_all), axis=1)

    # Number of directional samples (weights). Tuned to trade speed vs quality.
    # Very different from objective-space MC: we sample on the unit simplex (direction space).
    L = int(min(12000, max(1000, 300 * k, 500 * D)))
    rng = np.random.RandomState(1)  # deterministic

    # Sample weight vectors on the simplex using exponential trick
    # shape (L, D)
    exp_samples = rng.exponential(scale=1.0, size=(L, D))
    weights = exp_samples / exp_samples.sum(axis=1, keepdims=True)

    # For each weight, find the point minimizing the weighted sum (minimization problem)
    # Compute points dot weights.T => (N, L), argmin over axis=0 yields length-L array of winning indices
    # Use float64 for stable sums
    scores = pts_all.dot(weights.T)  # shape (N, L)
    best_per_weight = np.argmin(scores, axis=0)  # which point is best for each directional weight

    # Count number of weights each point wins
    counts = np.bincount(best_per_weight, minlength=N).astype(int)

    # Primary ranking: by counts desc, secondary by box_volume desc
    # lexsort keys: (secondary, primary) and lexsort uses last key as primary
    order = np.lexsort(( -box_volumes, -counts ))  # yields indices sorted by counts desc, then box_vol desc

    # Select top-k distinct winners; if not enough unique winners (rare), fill by box volume among leftovers
    selected = order[:k].tolist()
    if len(selected) < k:
        remaining = np.setdiff1d(np.arange(N), np.array(selected, dtype=int), assume_unique=True)
        if remaining.size > 0:
            extra = remaining[np.argsort(-box_volumes[remaining])][: k - len(selected)]
            selected.extend(extra.tolist())

    selected = np.array(selected, dtype=int)[:k]

    # Final deterministic fine tie-break: ensure deterministic ordering (by counts, box_vol)
    # Re-sort selected according to the same priority to produce stable output order
    sel_counts = counts[selected]
    sel_box = box_volumes[selected]
    sel_order = np.lexsort(( -sel_box, -sel_counts ))
    selected = selected[sel_order]

    return pts_all[selected].copy()


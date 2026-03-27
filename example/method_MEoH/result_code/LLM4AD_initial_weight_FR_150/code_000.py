import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape

    if k <= 0:
        return np.zeros((0, D), dtype=points.dtype)
    if k >= N:
        return points.copy()

    if reference_point is None:
        reference_point = points.max(axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Ensure reference is >= points per-dimension for minimization-case hyperrectangles.
    # If not, clamp reference to at least slightly above the max of points to avoid negative volumes.
    max_pts = points.max(axis=0)
    reference_point = np.maximum(reference_point, max_pts + 1e-12)

    # Sampling count: scale with k but cap to keep runtime reasonable
    M = int(min(20000, max(2000, 500 * k)))
    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    # Sample uniformly in the box between lower (min of points) and reference (upper)
    lower = points.min(axis=0)
    upper = reference_point
    # If any dimension has zero range, jitter upper a little (to avoid degenerate sampling)
    zero_range = (upper - lower) <= 0
    if np.any(zero_range):
        upper = upper.copy()
        upper[zero_range] = lower[zero_range] + 1e-6

    samples = rng.uniform(low=lower, high=upper, size=(M, D))

    # covers[sample, candidate] == True if sample is inside the hyper-rectangle [point, reference]
    # For minimization hypervolume rectangles: sample >= point (componentwise)
    # Use broadcasting to compute boolean mask efficiently
    # shape (M, N, D) would be large; instead compute in chunks if necessary
    try:
        covers = np.all(samples[:, None, :] >= points[None, :, :], axis=2)  # shape (M, N)
    except MemoryError:
        # fallback to chunking over samples
        covers = np.zeros((M, N), dtype=bool)
        chunk_size = max(1000, M // 10)
        for start in range(0, M, chunk_size):
            end = min(M, start + chunk_size)
            covers[start:end, :] = np.all(samples[start:end, None, :] >= points[None, :, :], axis=2)

    selected = []
    covered_mask = np.zeros(M, dtype=bool)
    remaining = np.ones(N, dtype=bool)

    # Precompute rectangle volumes for tie-breaking (volume of [point, reference])
    rect_sizes = np.prod(np.clip(upper - points, a_min=0.0, a_max=None), axis=1)

    for _ in range(k):
        # Compute new coverage counts for remaining candidates
        # new_covered = covers[:, remaining] & ~covered_mask[:, None]
        # counts = new_covered.sum(axis=0)
        rem_idx = np.nonzero(remaining)[0]
        if rem_idx.size == 0:
            break
        # Use matrix multiplication trick to count uncovered samples per candidate
        uncovered = ~covered_mask
        if uncovered.any():
            # counts = np.sum(covers[uncovered][:, rem_idx], axis=0)
            counts = covers[uncovered][:, rem_idx].sum(axis=0)
        else:
            counts = np.zeros(rem_idx.shape[0], dtype=int)

        # If all counts are zero (no sample distinguishes candidates), break ties with rect_sizes
        if counts.max() == 0:
            # choose remaining candidate with largest rect_sizes
            rem_sizes = rect_sizes[rem_idx]
            best_local = int(np.argmax(rem_sizes))
        else:
            # pick candidate with largest new coverage; tie-break by rect_sizes
            max_count = counts.max()
            candidates = np.nonzero(counts == max_count)[0]
            if candidates.size == 1:
                best_local = int(candidates[0])
            else:
                # among tied, pick one with largest rect size
                tied_idx = rem_idx[candidates]
                best_tie = int(np.argmax(rect_sizes[tied_idx]))
                best_local = int(candidates[best_tie])

        best = rem_idx[best_local]
        selected.append(best)
        remaining[best] = False
        # update covered mask
        covered_mask |= covers[:, best]

    # If we selected fewer than k due to numeric issues, fill remaining by rect size
    if len(selected) < k:
        rem_idx = np.nonzero(remaining)[0]
        if rem_idx.size > 0:
            rem_sorted = rem_idx[np.argsort(-rect_sizes[rem_idx])]
            need = k - len(selected)
            to_add = rem_sorted[:need].tolist()
            selected.extend(to_add)

    selected = selected[:k]
    return points[np.array(selected, dtype=int)].copy()


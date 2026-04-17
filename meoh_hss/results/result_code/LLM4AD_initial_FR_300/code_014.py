import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2‑D array")
    N, D = pts.shape
    if k <= 0 or k > N:
        raise ValueError("k must satisfy 1 ≤ k ≤ N")

    # reference point
    if reference_point is None:
        reference_point = pts.max(axis=0) * 1.1
    ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # bounding box for MC sampling
    low = pts.min(axis=0)
    high = ref.copy()
    span = high - low
    eps = 1e-12
    span[span <= 0] = eps

    # number of samples (trade‑off between accuracy & speed)
    M = int(min(12000, max(2000, 300 * k, 1000 * D)))
    rng = np.random.default_rng(0)
    samples = rng.random((M, D)) * span + low   # shape (M, D)

    # pre‑compute domination masks (point → samples)
    max_bool = 8e7                     # ≈ 80 M booleans → ~10 MiB
    block = max(1, int(max_bool // M))
    masks = np.empty((N, M), dtype=bool)

    for s in range(0, N, block):
        e = min(N, s + block)
        chunk = pts[s:e]                                   # (b, D)
        masks[s:e] = np.all(chunk[:, None, :] <= samples[None, :, :], axis=2)

    # lazy‑greedy selection with max‑heap
    init_gain = masks.sum(axis=1).astype(np.int32)
    heap = [(-g, i) for i, g in enumerate(init_gain)]
    heapq.heapify(heap)

    selected = []
    covered = np.zeros(M, dtype=bool)

    while len(selected) < k and heap:
        neg_gain, idx = heapq.heappop(heap)
        true_gain = int(np.count_nonzero(masks[idx] & ~covered))
        if -neg_gain != true_gain:
            heapq.heappush(heap, (-true_gain, idx))
            continue
        selected.append(idx)
        newly = masks[idx] & ~covered
        if newly.any():
            covered |= newly

    # if heap exhausted before k, fill with highest box volume
    if len(selected) < k:
        remaining = np.setdiff1d(np.arange(N), selected, assume_unique=True)
        box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)
        extra = remaining[np.argsort(-box_vol[remaining])[:k - len(selected)]]
        selected.extend(extra.tolist())

    selected = np.array(selected, dtype=int)

    # ---------- local‑swap refinement ----------
    # shared data for refinement
    sel_mask = np.zeros(N, dtype=bool)
    sel_mask[selected] = True
    # scaled coordinates for distance computation
    mins = pts.min(axis=0)
    denom = (ref - mins).copy()
    denom[denom <= 0] = 1.0
    scaled = (pts - mins) / denom

    # helper: approximate hypervolume of a set using the shared samples
    def approx_hv(indices):
        if indices.size == 0:
            return 0.0
        sub_masks = masks[indices]                     # (m, M)
        dom = np.any(sub_masks, axis=0)                # (M,)
        frac = dom.sum() / M
        return frac * np.prod(span)

    # current hv
    current_hv = approx_hv(selected)

    # refinement parameters
    L = 8                     # number of nearest neighbours to examine per selected point
    max_passes = 2
    tol = 1e-12

    for _ in range(max_passes):
        improved = False
        for pos, idx in enumerate(selected):
            # distances to all other points
            dists = np.linalg.norm(scaled - scaled[idx], axis=1)
            dists[sel_mask] = np.inf
            cand_idxs = np.argpartition(dists, L)[:L]
            best_hv = current_hv
            best_cand = None
            for cand in cand_idxs:
                trial = selected.copy()
                trial[pos] = cand
                hv = approx_hv(trial)
                if hv > best_hv + tol:
                    best_hv = hv
                    best_cand = cand
            if best_cand is not None:
                # perform swap
                sel_mask[idx] = False
                sel_mask[best_cand] = True
                selected[pos] = best_cand
                current_hv = best_hv
                improved = True
        if not improved:
            break
    # ------------------------------------------------------------------

    return pts[selected[:k]].copy()


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    # ------------------------------------------------------------------ #
    # 1️⃣  basic checks & reference point
    # ------------------------------------------------------------------ #
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2‑D array")
    N, D = pts.shape
    if k <= 0 or k > N:
        raise ValueError("k must satisfy 1 ≤ k ≤ N")
    if reference_point is None:
        reference_point = pts.max(axis=0) * 1.1
    ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # ------------------------------------------------------------------ #
    # 2️⃣  Pareto‑front pruning (minimisation)
    # ------------------------------------------------------------------ #
    def nondominated_idx(arr):
        idx = np.lexsort(arr.T[::-1])
        front = []
        for i in idx:
            p = arr[i]
            if not front:
                front.append(i)
                continue
            front_pts = arr[np.array(front)]
            if np.any((front_pts <= p).all(axis=1)):
                continue
            dom = (p <= front_pts).all(axis=1) & (p < front_pts).any(axis=1)
            if np.any(dom):
                keep = np.where(~dom)[0]
                front = list(np.array(front)[keep])
            front.append(i)
        return np.array(front, dtype=int)

    nd_idx = nondominated_idx(pts)
    if nd_idx.size < N:
        pts = pts[nd_idx]
        N = pts.shape[0]
        if N <= k:
            return pts.copy()

    # ------------------------------------------------------------------ #
    # 3️⃣  Monte‑Carlo samples (shared for all points)
    # ------------------------------------------------------------------ #
    low = pts.min(axis=0)
    high = ref.copy()
    span = high - low
    eps = 1e-12
    span[span <= 0] = eps
    M = int(min(12000, max(2000, 300 * k, 1000 * D)))      # sample count
    rng = np.random.default_rng(0)
    samples = rng.random((M, D)) * span + low               # (M, D)

    # ------------------------------------------------------------------ #
    # 4️⃣  Pre‑compute domination masks (point → samples)
    # ------------------------------------------------------------------ #
    max_bool = 8e7            # ≈ 10 MiB
    block = max(1, int(max_bool // M))
    masks = np.empty((N, M), dtype=bool)

    for s in range(0, N, block):
        e = min(N, s + block)
        chunk = pts[s:e]                               # (b, D)
        masks[s:e] = np.all(chunk[:, None, :] <= samples[None, :, :], axis=2)

    # ------------------------------------------------------------------ #
    # 5️⃣  Lazy‑greedy selection (max‑heap of marginal gains)
    # ------------------------------------------------------------------ #
    init_gain = masks.sum(axis=1).astype(np.int32)
    heap = [(-g, i) for i, g in enumerate(init_gain)]
    heapq.heapify(heap)

    selected_idx = []
    covered = np.zeros(M, dtype=bool)

    while len(selected_idx) < k and heap:
        neg_gain, idx = heapq.heappop(heap)
        true_gain = np.count_nonzero(masks[idx] & ~covered)
        if -neg_gain != true_gain:
            heapq.heappush(heap, (-true_gain, idx))
            continue
        selected_idx.append(idx)
        newly = masks[idx] & ~covered
        if newly.any():
            covered |= newly

    # If heap exhausted before k (rare), fill with highest box volume
    if len(selected_idx) < k:
        remaining = np.setdiff1d(np.arange(N), selected_idx, assume_unique=True)
        box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)
        extra = remaining[np.argsort(-box_vol[remaining])[:k - len(selected_idx)]]
        selected_idx.extend(extra.tolist())

    selected_idx = np.array(selected_idx, dtype=int)[:k]

    # ------------------------------------------------------------------ #
    # 6️⃣  Build a short‑list for refinement
    # ------------------------------------------------------------------ #
    # candidates not yet selected, ordered by current marginal gain
    not_sel = np.setdiff1d(np.arange(N), selected_idx, assume_unique=True)
    # recompute marginal gains for those candidates (cheap)
    marginal = (masks[not_sel] & ~covered).sum(axis=1)
    order = np.argsort(-marginal)
    shortlist_extra = not_sel[order][:min(2 * k, not_sel.size)]
    shortlist_idx = np.unique(np.concatenate([selected_idx, shortlist_extra]))
    shortlist = pts[shortlist_idx]

    # map selected indices to positions inside shortlist
    sel_pos = [int(np.where(shortlist_idx == i)[0][0]) for i in selected_idx]

    # ------------------------------------------------------------------ #
    # 7️⃣  Optional exact refinement (swap) if pygmo is available
    # ------------------------------------------------------------------ #
    try:
        import pygmo as pg
        have_pygmo = True
    except Exception:
        have_pygmo = False

    if have_pygmo and shortlist.shape[0] > k:
        def exact_hv(subset):
            if subset.size == 0:
                return 0.0
            hv = pg.hypervolume(subset)
            return hv.compute(ref)

        current_set = shortlist[sel_pos]
        best_hv = exact_hv(current_set)
        improved = True
        max_iters = 50
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            for i, s_pos in enumerate(sel_pos):
                for c_pos in range(shortlist.shape[0]):
                    if c_pos in sel_pos:
                        continue
                    trial = sel_pos.copy()
                    trial[i] = c_pos
                    hv = exact_hv(shortlist[trial])
                    if hv > best_hv + 1e-12:
                        best_hv = hv
                        sel_pos = trial
                        improved = True
                        break
                if improved:
                    break

        refined = shortlist[sel_pos]
    else:
        refined = pts[selected_idx]

    # ------------------------------------------------------------------ #
    # 8️⃣  Restore original indices if Pareto pruning was applied
    # ------------------------------------------------------------------ #
    if nd_idx.size < points.shape[0]:
        return points[nd_idx[refined.astype(int)]].copy()
    return refined.copy()


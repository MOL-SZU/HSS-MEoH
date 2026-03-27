import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np, heapq

    # --------------------------------------------------------------
    # 0. basic checks & reference point
    # --------------------------------------------------------------
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2‑D array")
    N, D = pts.shape
    if not (1 <= k <= N):
        raise ValueError("k must satisfy 1 ≤ k ≤ N")
    if reference_point is None:
        reference_point = pts.max(axis=0) * 1.1
    ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # --------------------------------------------------------------
    # 1. shared Monte‑Carlo samples (size tuned to k and D)
    # --------------------------------------------------------------
    low = pts.min(axis=0)
    span = ref - low
    eps = 1e-12
    span[span <= 0] = eps
    M = int(min(12000, max(2000, 300 * k, 800 * D)))   # fast yet accurate
    rng = np.random.default_rng(0)
    samples = rng.random((M, D)) * span + low

    # --------------------------------------------------------------
    # 2. block‑wise mask → sparse index lists (memory‑safe)
    # --------------------------------------------------------------
    max_bool = int(8e7)            # ~10 MiB budget
    block = max(1, int(max_bool // M))
    point_inds = [None] * N
    init_gain = np.empty(N, dtype=int)

    for s in range(0, N, block):
        e = min(N, s + block)
        chunk = pts[s:e]                               # (b, D)
        mask = np.all(chunk[:, None, :] <= samples[None, :, :], axis=2)  # (b, M)
        for i in range(e - s):
            idxs = np.nonzero(mask[i])[0].astype(np.int32)
            point_inds[s + i] = idxs
            init_gain[s + i] = idxs.size

    # --------------------------------------------------------------
    # 3. lazy‑greedy selection with max‑heap (gain, box‑volume tie‑break)
    # --------------------------------------------------------------
    box_vol = np.prod(np.maximum(0.0, ref - pts), axis=1)
    heap = [(-int(g), -float(box_vol[i]), i) for i, g in enumerate(init_gain)]
    heapq.heapify(heap)

    selected = []
    covered = np.zeros(M, dtype=bool)

    while len(selected) < k and heap:
        neg_gain, neg_vol, idx = heapq.heappop(heap)
        inds = point_inds[idx]
        true_gain = int(np.count_nonzero(~covered[inds])) if inds.size else 0
        if -neg_gain != true_gain:
            heapq.heappush(heap, (-true_gain, neg_vol, idx))
            continue
        if true_gain == 0:          # no uncovered contribution → skip
            continue
        selected.append(idx)
        if inds.size:
            covered[inds] = True

    # fill remaining slots with highest box volume (rare)
    if len(selected) < k:
        remaining = np.setdiff1d(np.arange(N), selected, assume_unique=True)
        extra = remaining[np.argsort(-box_vol[remaining])[:k - len(selected)]]
        selected.extend(extra.tolist())

    selected = np.array(selected[:k], dtype=int)

    # --------------------------------------------------------------
    # 4. cheap local‑swap polish (nearest‑neighbour candidates)
    # --------------------------------------------------------------
    # scaled coordinates for Euclidean distances
    mins = pts.min(axis=0)
    denom = (ref - mins).copy()
    denom[denom <= 0] = 1.0
    scaled = (pts - mins) / denom

    sel_mask = np.zeros(N, dtype=bool)
    sel_mask[selected] = True

    def approx_hv(idxs):
        if idxs.size == 0:
            return 0.0
        union = np.zeros(M, dtype=bool)
        for i in idxs:
            union[point_inds[i]] = True
        return union.mean() * np.prod(span)

    cur_hv = approx_hv(selected)
    L = 6                     # neighbours examined per point
    max_passes = 2
    tol = 1e-12

    for _ in range(max_passes):
        improved = False
        for pos, idx in enumerate(selected):
            # nearest neighbours among *unselected* points
            dists = np.linalg.norm(scaled - scaled[idx], axis=1)
            dists[sel_mask] = np.inf
            cand = np.argpartition(dists, L)[:L]
            best_hv = cur_hv
            best_cand = None
            for c in cand:
                trial = selected.copy()
                trial[pos] = c
                hv = approx_hv(trial)
                if hv > best_hv + tol:
                    best_hv = hv
                    best_cand = c
            if best_cand is not None:
                sel_mask[idx] = False
                sel_mask[best_cand] = True
                selected[pos] = best_cand
                cur_hv = best_hv
                improved = True
        if not improved:
            break

    # --------------------------------------------------------------
    # 5. optional exact refinement on a tiny shortlist (pygmo)
    # --------------------------------------------------------------
    try:
        import pygmo as pg
        have_pg = True
    except Exception:
        have_pg = False

    if have_pg and N > k:
        # build tiny shortlist: selected + top‑margin candidates
        not_sel = np.setdiff1d(np.arange(N), selected, assume_unique=True)
        if not_sel.size:
            marg = np.empty(not_sel.size, dtype=int)
            for i, idx in enumerate(not_sel):
                marg[i] = int(np.count_nonzero(~covered[point_inds[idx]]))
            top_extra = not_sel[np.argsort(-marg)[:min(2 * k, not_sel.size)]]
            shortlist_idx = np.unique(np.concatenate([selected, top_extra]))
        else:
            shortlist_idx = selected.copy()
        shortlist = pts[shortlist_idx]

        # map selected positions inside shortlist
        sel_pos = [int(np.where(shortlist_idx == i)[0][0]) for i in selected]

        def exact_hv(arr):
            return float(pg.hypervolume(arr).compute(ref))

        cur_set = shortlist[sel_pos]
        best_hv = exact_hv(cur_set)
        improved = True
        it = 0
        max_it = 30
        while improved and it < max_it:
            improved = False
            it += 1
            for s_i, s_pos in enumerate(sel_pos):
                for c_pos in range(len(shortlist_idx)):
                    if c_pos in sel_pos:
                        continue
                    trial = sel_pos.copy()
                    trial[s_i] = c_pos
                    hv = exact_hv(shortlist[trial])
                    if hv > best_hv + 1e-12:
                        best_hv = hv
                        sel_pos = trial
                        improved = True
                        break
                if improved:
                    break
        selected = shortlist_idx[np.array(sel_pos, dtype=int)]

    return pts[selected[:k]].copy()


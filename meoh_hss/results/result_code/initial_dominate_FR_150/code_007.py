import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    # reproducibility
    rng = np.random.default_rng(12345)

    # pygmo for hypervolume
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 0))

    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D).")
    N, D = points.shape

    if not isinstance(k, int):
        raise ValueError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be > 0")

    pad_if_needed = False
    if k > N:
        pad_if_needed = True
        k_eff = N
    else:
        k_eff = k

    # reference point
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape((D,))
    if reference_point.shape[0] != D:
        raise ValueError("reference_point must have dimension D = points.shape[1].")

    # Efficient nondominated filter (vectorized-ish, assumes minimization)
    def nondominated_indices(arr: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        if n == 0:
            return np.array([], dtype=int)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            # any j that dominates i
            comp = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
            if np.any(comp):
                mask[i] = False
            else:
                # remove points dominated by i
                dominated_by_i = np.all(arr >= arr[i], axis=1) & np.any(arr > arr[i], axis=1)
                mask[dominated_by_i] = False
                mask[i] = True
        return np.nonzero(mask)[0]

    # Small N edge handling
    if N == 0:
        return np.empty((0, D))
    if N <= 1:
        if k == 1 or pad_if_needed:
            out = np.tile(points[0].reshape(1, D), (1 if k_eff == 1 else k if pad_if_needed else 1, 1))
            if pad_if_needed and k > 1:
                out = np.vstack([points[0].reshape(1, D)] * k)
            return out.reshape(k, D)

    # 1) Pareto filter
    nd_idx = nondominated_indices(points)
    nd_points = points[nd_idx]
    m = nd_points.shape[0]

    # If few nondominated, include them and fill with surrogate
    if m <= k_eff:
        selected_indices = list(nd_idx.tolist())
        remaining = [i for i in range(N) if i not in selected_indices]
        if remaining:
            eps = 1e-12
            diffs = np.maximum(reference_point - points[remaining], eps)
            # New surrogate: inverse-square range weights (sharper), additive smoothing, different temp
            data_min = np.min(points, axis=0)
            data_max = np.max(points, axis=0)
            ranges = np.maximum(data_max - data_min, eps)
            weights = 1.0 / (ranges ** 2)
            weights = weights / np.sum(weights)
            logprod = np.sum(weights * np.log(diffs + 1e-9), axis=1)
            min_gap = np.min(diffs, axis=1) / (np.max(reference_point - data_min) + eps)
            temp = 0.75
            scores = temp * logprod + 0.25 * min_gap
            order = np.argsort(-scores)
            for idx in order:
                if len(selected_indices) >= k_eff:
                    break
                selected_indices.append(int(remaining[int(idx)]))
        while len(selected_indices) < k_eff:
            selected_indices.append(selected_indices[-1] if selected_indices else 0)
        final_idx = selected_indices[:k_eff]
        if pad_if_needed:
            while len(final_idx) < k:
                final_idx.append(final_idx[-1])
        return points[np.array(final_idx, dtype=int)]

    # 2) Shortlist by sharper surrogate + bounded diversity (operate on nondominated indices)
    eps = 1e-12
    diffs_nd = np.maximum(reference_point - nd_points, eps)

    # surrogate settings:
    # - per-dimension inverse-square-range weights (sharper emphasis on narrow dims)
    # - temperature gamma to blend logprod and min-gap normalized
    data_min = np.min(points, axis=0)
    data_max = np.max(points, axis=0)
    ranges = np.maximum(data_max - data_min, eps)
    weights = 1.0 / (ranges ** 2)
    weights = weights / np.sum(weights)
    logprod = np.sum(weights * np.log(diffs_nd + 1e-9), axis=1)  # base log-product with small smoothing
    min_gap = np.min(diffs_nd, axis=1)
    # temperature parameter: lower => sharper focus on top logprod; gamma blends
    gamma = 0.75
    surrogate = gamma * logprod + (1 - gamma) * (min_gap / (np.max(min_gap) + eps))

    # shortlist size: more conservative than original (smaller), adaptive
    candidate_count = min(m, max(3 * k_eff, 60))
    candidate_count = max(candidate_count, k_eff)

    # stable ordering by surrogate and deterministic tie-break using index
    order_sur = np.argsort(-surrogate, kind='stable')
    top_sur = int(order_sur[0])

    # seed second seed as the best min_gap among top fraction to diversify
    top_fraction = max(1, min(m, int(np.ceil(0.1 * m))))
    top_frac_idx = order_sur[:top_fraction]
    if top_frac_idx.size > 0:
        best_min_gap_idx = int(top_frac_idx[np.argmax(min_gap[top_frac_idx])])
    else:
        best_min_gap_idx = top_sur

    candidates_local = []
    chosen_local = np.zeros(m, dtype=bool)

    # add seeds deterministically (top surrogate first)
    seeds = []
    seeds.append(top_sur)
    if best_min_gap_idx != top_sur:
        seeds.append(best_min_gap_idx)
    for s in seeds:
        if not chosen_local[s]:
            candidates_local.append(int(s))
            chosen_local[s] = True

    # Farthest-first (squared euclidean) sampling to diversify shortlist (vectorized)
    if m > 1:
        nd_pts = nd_points
        # initialize distances to +inf, then set based on existing candidates
        dists = np.full(m, np.inf)
        if candidates_local:
            # compute pairwise squared dists to current seed set and take min
            seed_pts = nd_pts[np.array(candidates_local)]
            # compute squared distances to seeds
            diff = nd_pts[:, None, :] - seed_pts[None, :, :]
            d_sq = np.sum(diff ** 2, axis=2)  # shape (m, nseeds)
            dists = np.min(d_sq, axis=1)
        else:
            dists = np.full(m, np.inf)
        # mark chosen
        dists[chosen_local] = -1.0
        while len(candidates_local) < candidate_count:
            cand = int(np.argmax(dists))
            if dists[cand] < 0:
                break
            candidates_local.append(cand)
            chosen_local[cand] = True
            # update dists with new center
            center = nd_pts[cand]
            d = np.sum((nd_pts - center) ** 2, axis=1)
            dists = np.minimum(dists, d)
            dists[chosen_local] = -1.0
            if np.all(chosen_local):
                break

    # Map to global indices and unique-preserving order
    candidates_global = [int(nd_idx[i]) for i in candidates_local]
    seen = set()
    candidates_global = [x for x in candidates_global if not (x in seen or seen.add(x))]

    # 3) CELF lazy greedy on indices with HV cache and precomputed singletons
    hv_cache = {}

    def hv_of_indexset(index_iterable):
        key = tuple(sorted(map(int, index_iterable)))
        if key in hv_cache:
            return hv_cache[key]
        if len(key) == 0:
            val = 0.0
        else:
            arr = points[np.array(key, dtype=int)]
            val = float(pg.hypervolume(arr).compute(reference_point))
        hv_cache[key] = val
        return val

    # precompute singleton HVs for candidates
    singleton_hv = {}
    for c in candidates_global:
        val = hv_of_indexset((c,))
        singleton_hv[c] = val

    selected_set = set()
    selected_list = []

    # initialize heap with singleton gains (hv({i}) - hv(empty)=hv({i}))
    heap = []
    for order_tiebreak, c in enumerate(candidates_global):
        g = singleton_hv.get(c, hv_of_indexset((c,)))
        # tuple: (-estimated_gain, tie_order, idx, last_eval_size)
        heap.append((-g, int(order_tiebreak), int(c), 0))
    heapq.heapify(heap)

    # CELF lazy greedy
    # Keep track of hv_S to avoid repeated empty recompute
    hv_S = 0.0
    while len(selected_list) < k_eff and heap:
        neg_gain_est, tie, idx_c, last_eval = heapq.heappop(heap)
        gain_est = -neg_gain_est
        # if the candidate was already selected skip
        if idx_c in selected_set:
            continue
        if last_eval == len(selected_list):
            # gain_est is up-to-date
            if idx_c in selected_set:
                continue
            selected_list.append(int(idx_c))
            selected_set.add(int(idx_c))
            # update hv_S incrementally
            hv_S = hv_of_indexset(selected_set)
            continue
        # recompute true marginal
        hv_S_cur = hv_of_indexset(selected_set)
        new_set = set(selected_set)
        new_set.add(int(idx_c))
        hv_new = hv_of_indexset(new_set)
        true_gain = hv_new - hv_S_cur
        # push back updated with current last_eval marker
        heapq.heappush(heap, (-true_gain, int(tie), int(idx_c), len(selected_list)))
        # early stopping if top marginal non-positive and current true_gain non-positive
        top_neg = heap[0][0] if heap else 0.0
        if gain_est <= 1e-12 and true_gain <= 1e-12 and -top_neg <= 1e-12:
            break

    # If not enough selected yet, fill from global remaining by surrogate deterministic order
    if len(selected_list) < k_eff:
        selected_set_local = set(selected_list)
        remaining_global = [i for i in range(N) if i not in selected_set_local]
        if remaining_global:
            rem_pts = points[remaining_global]
            rem_diffs = np.maximum(reference_point - rem_pts, eps)
            rem_logprod = np.sum(weights * np.log(rem_diffs + 1e-9), axis=1)
            rem_min_gap = np.min(rem_diffs, axis=1)
            rem_scores = gamma * rem_logprod + (1 - gamma) * (rem_min_gap / (np.max(rem_min_gap) + eps))
            order_rem = np.argsort(-rem_scores, kind='stable')
            for idx in order_rem:
                if len(selected_list) >= k_eff:
                    break
                selected_list.append(int(remaining_global[int(idx)]))
        while len(selected_list) < k_eff:
            selected_list.append(selected_list[-1] if selected_list else 0)

    # Ensure uniqueness and fill if duplicates reduced count
    final_sel = selected_list[:k_eff]
    unique_final = list(dict.fromkeys(final_sel))
    if len(unique_final) < k_eff:
        leftover = [i for i in range(N) if i not in unique_final]
        for idx in leftover:
            if len(unique_final) >= k_eff:
                break
            unique_final.append(int(idx))
        while len(unique_final) < k_eff:
            unique_final.append(unique_final[-1])
        final_sel = unique_final[:k_eff]

    # pad if original k > N
    if pad_if_needed:
        while len(final_sel) < k:
            final_sel.append(final_sel[-1])

    final_sel = np.array(final_sel, dtype=int)[:k]
    subset = points[final_sel]
    return subset.reshape((k, D))


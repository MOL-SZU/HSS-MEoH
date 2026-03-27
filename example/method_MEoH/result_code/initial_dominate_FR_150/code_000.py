import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    rng = np.random.default_rng(42)  # seed for reproducibility

    N, D = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point).reshape((D,))

    # Helper: Pareto (nondominated) filtering (minimization assumed)
    def nondominated_indices(pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # j dominates i if all(pts[j] <= pts[i]) and any strictly <
            # skip comparing i with itself by masking
            le = np.all(pts <= pts[i], axis=1)
            lt = np.any(pts < pts[i], axis=1)
            # if any other j dominates i -> dominated[i] = True
            comp = le & lt
            comp[i] = False
            if np.any(comp):
                dominated[i] = True
            else:
                # mark any points dominated by i (optional speed up)
                dominated = dominated | ((np.all(pts[i] <= pts, axis=1)) & (np.any(pts[i] < pts, axis=1)))
                dominated[i] = False
        return np.where(~dominated)[0]

    # Edge cases
    if N == 0 or k == 0:
        return np.empty((0, D))

    # Filter dominated points first
    nd_idx = nondominated_indices(points)
    nd_points = points[nd_idx]

    # If few points overall, just return random or all
    if N <= k:
        # if not enough points, return all and pad by sampling remaining (if any duplicates allowed)
        if N == k:
            return points.copy()
        # pick all then fill random from themselves (no duplicates) - sample from entire set without replacement
        selected_idx = np.arange(N)
        if N < k:
            remaining = np.setdiff1d(np.arange(N), selected_idx)
            need = k - len(selected_idx)
            if len(remaining) > 0:
                extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
                final_idx = np.concatenate([selected_idx, extra])
            else:
                # all points already selected (N < k), replicate last row to match shape
                extra = np.repeat(points[-1][None, :], k - N, axis=0)
                return np.vstack([points, extra])
            return points[final_idx]

    # Stage-1: fast surrogate scoring to shortlist candidates
    # Surrogate: log product of (ref - p) (larger -> more contribution), stable via small eps
    eps = 1e-12
    diffs = np.maximum(reference_point - nd_points, eps)
    approx_logprod = np.sum(np.log(diffs), axis=1)  # proxy for dominated hypervolume per point
    # Candidate pool size: a tunable parameter (different from original algorithm's epsilon/delta/T)
    max_candidates = min(len(nd_idx), max(8 * k, 50))  # aggressive shortlist but bounded
    order = np.argsort(-approx_logprod)  # descending
    cand_idx = nd_idx[order[:max_candidates]]  # global indices of candidates
    cand_set = set(cand_idx.tolist())

    # If candidate pool smaller than k, augment with other nondominated points
    if len(cand_idx) < k:
        needed = k - len(cand_idx)
        others = [i for i in nd_idx if i not in cand_set]
        if len(others) > 0:
            add = others[:needed]
            cand_idx = np.concatenate([cand_idx, np.array(add, dtype=int)])
            cand_set = set(cand_idx.tolist())

    # Caching hypervolume values for sets to avoid recomputation
    hv_cache = {}
    def hv_of(indices):
        key = frozenset(indices)
        if key in hv_cache:
            return hv_cache[key]
        if len(indices) == 0:
            val = 0.0
        else:
            data = points[np.array(list(indices), dtype=int)]
            val = pg.hypervolume(data).compute(reference_point)
        hv_cache[key] = val
        return val

    # Greedy selection w.r.t marginal HV contribution (exact)
    selected = []
    selected_set = set()
    base_hv = hv_of(frozenset())  # zero
    # Precompute candidate list
    candidates = list(cand_idx)

    for _ in range(min(k, len(candidates))):
        best_c = None
        best_gain = -np.inf
        hv_S = hv_of(selected_set)
        # compute marginal contributions
        for c in candidates:
            if c in selected_set:
                continue
            new_set = set(selected_set)
            new_set.add(int(c))
            hv_new = hv_of(new_set)
            gain = hv_new - hv_S
            if gain > best_gain:
                best_gain = gain
                best_c = int(c)
        if best_c is None:
            break
        # If no positive gain, still pick best to reach k (but prefer positive)
        if best_gain <= 0:
            # if no positive marginal HV remains, stop greedy early
            # fill remaining slots later
            break
        selected.append(best_c)
        selected_set.add(best_c)

    # If we don't yet have k points, fill with best remaining by approximate score (or dominated points if needed)
    if len(selected) < k:
        remaining_global = [i for i in range(N) if i not in selected_set]
        # score remaining by surrogate (use same logprod proxy, compute for all remaining)
        rem_points = points[remaining_global]
        rem_diffs = np.maximum(reference_point - rem_points, eps)
        rem_score = np.sum(np.log(rem_diffs), axis=1)
        order_rem = np.argsort(-rem_score)
        for idx in order_rem:
            if len(selected) >= k:
                break
            selected.append(int(remaining_global[idx]))
            selected_set.add(int(remaining_global[idx]))
    # If still not enough (shouldn't happen), pad randomly
    if len(selected) < k:
        leftover = [i for i in range(N) if i not in selected_set]
        need = k - len(selected)
        if len(leftover) > 0:
            extra = rng.choice(leftover, size=min(need, len(leftover)), replace=False)
            selected.extend([int(x) for x in extra])
            selected_set.update(extra)
        else:
            # replicate last point if necessary
            while len(selected) < k:
                selected.append(selected[-1])

    # Local swap refinement: try to improve HV by swapping one selected with one not selected (from candidate pool + some neighbors)
    max_swaps = 200
    improved = True
    swaps = 0
    current_set = set(selected[:k])
    current_hv = hv_of(current_set)
    # Build external pool for swaps: use top M nondominated points (by approx) to consider as outsiders
    swap_pool = set(cand_idx) | set(nd_idx)  # reasonable pool
    swap_pool = set([int(x) for x in swap_pool])
    non_selected_pool = list(swap_pool - current_set)
    while improved and swaps < max_swaps:
        improved = False
        swaps += 1
        # iterate over selected points, try replacing with any non-selected in pool
        outer_loop_break = False
        for a in list(current_set):
            for b in non_selected_pool:
                if b in current_set:
                    continue
                new_set = set(current_set)
                new_set.remove(a)
                new_set.add(b)
                hv_new = hv_of(new_set)
                if hv_new > current_hv + 1e-12:
                    # accept swap
                    current_set = new_set
                    current_hv = hv_new
                    non_selected_pool = list(swap_pool - current_set)
                    improved = True
                    outer_loop_break = True
                    break
            if outer_loop_break:
                break

    final_selected = list(current_set)
    # Ensure exactly k elements
    if len(final_selected) > k:
        final_selected = final_selected[:k]
    elif len(final_selected) < k:
        # fill from remaining by surrogate
        remaining = [i for i in range(N) if i not in final_selected]
        if remaining:
            rem_points = points[remaining]
            rem_diffs = np.maximum(reference_point - rem_points, eps)
            rem_score = np.sum(np.log(rem_diffs), axis=1)
            order_rem = np.argsort(-rem_score)
            for idx in order_rem:
                if len(final_selected) >= k:
                    break
                final_selected.append(int(remaining[idx]))
        while len(final_selected) < k:
            final_selected.append(final_selected[-1])

    subset = points[np.array(final_selected[:k], dtype=int)]
    return subset


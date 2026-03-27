import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import pygmo as pg
    import heapq
    import time

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D).")
    n, d = points.shape
    if not (isinstance(k, int) and 0 < k <= n):
        raise ValueError("k must be an integer with 0 < k <= number of points")

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,) matching points' dimension")

    # Helper to compute hypervolume for a set of indices
    def hv_of_indices(indices):
        if len(indices) == 0:
            return 0.0
        try:
            arr = points[np.array(indices, dtype=int)]
            return float(pg.hypervolume(arr).compute(reference_point))
        except Exception:
            return 0.0

    # Precompute individual hypervolumes for all singletons
    individual_hv = np.zeros(n, dtype=float)
    for i in range(n):
        try:
            individual_hv[i] = float(pg.hypervolume(points[[i]]).compute(reference_point))
        except Exception:
            individual_hv[i] = 0.0

    # Trivial case
    if k == n:
        return points.copy()

    start_time = time.time()
    time_limit = 5.0  # total allowed time for selection + refinement

    # CELF-style lazy greedy selection
    # Heap entries: (-gain, idx, last_updated_iter)
    heap = []
    for i in range(n):
        # initial gain = individual hv (as if selected set is empty)
        heap.append((-individual_hv[i], int(i), 0))
    heapq.heapify(heap)

    selected = []
    selected_set = set()
    current_hv = 0.0
    iter_id = 0
    # To avoid infinite loops, cap the number of heap pops
    max_pops = n * k * 2 + 1000

    pops = 0
    while len(selected) < k and (time.time() - start_time) < time_limit:
        if not heap:
            break
        pops += 1
        if pops > max_pops:
            break
        neg_gain, idx, last_upd = heapq.heappop(heap)
        gain = -neg_gain
        # If this entry was last updated in the current iteration, accept directly
        if last_upd == iter_id:
            # Accept this index
            if idx in selected_set:
                # Already selected via some duplicate; skip
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            # update current_hv exactly
            current_hv = hv_of_indices(selected)
            iter_id += 1
            continue
        else:
            # Need to recompute true marginal gain with respect to current selection
            if idx in selected_set:
                continue
            new_hv = hv_of_indices(list(selected) + [int(idx)])
            true_gain = new_hv - current_hv
            # push back with updated gain and mark last_updated = iter_id
            heapq.heappush(heap, (-true_gain, int(idx), iter_id))
            continue

    # If we didn't fill k elements (due to timeouts or heap exhaustion), fill by top individual hv
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: individual_hv[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))
        current_hv = hv_of_indices(selected)

    # Limited refinement: try replacements from top individual-HV outsiders
    # We'll allow a bounded number of successful swaps
    max_swaps = max(5, 5 * k)
    swaps_done = 0
    tol = 1e-12
    # Prepare outsiders sorted by individual hv
    outsiders = [i for i in range(n) if i not in selected_set]
    outsiders_sorted = sorted(outsiders, key=lambda x: individual_hv[x], reverse=True)
    # Candidate pool size: limit to reasonable number to control runtime
    pool_size = min(len(outsiders_sorted), max(50, 10 * k))
    candidate_pool = outsiders_sorted[:pool_size]

    # Run iterative single-swap improvements until no improvement or swap budget exhausted or time exceeded
    improvement = True
    while improvement and swaps_done < max_swaps and (time.time() - start_time) < time_limit:
        improvement = False
        # iterate selected items and try to replace with best candidate
        for sel_pos, sel_idx in enumerate(list(selected)):
            best_local_gain = 0.0
            best_cand = None
            # Optionally consider candidates in order of individual hv
            for cand in candidate_pool:
                if cand in selected_set:
                    continue
                # form new selection by replacement
                new_sel = list(selected)
                new_sel[sel_pos] = int(cand)
                hv_new = hv_of_indices(new_sel)
                gain = hv_new - current_hv
                if gain > best_local_gain + tol:
                    best_local_gain = gain
                    best_cand = int(cand)
                    # early break if significant improvement
                    if best_local_gain >= 1e-6:
                        break
            if best_cand is not None and best_local_gain > tol:
                # apply swap
                old_idx = selected[sel_pos]
                selected[sel_pos] = int(best_cand)
                selected_set.remove(int(old_idx))
                selected_set.add(int(best_cand))
                current_hv += best_local_gain
                swaps_done += 1
                improvement = True
                # update candidate pool (remove chosen)
                if best_cand in candidate_pool:
                    candidate_pool = [c for c in candidate_pool if c != best_cand]
                # break to restart scanning selected list after change (greedy)
                break

    # Final safeguard: ensure exactly k points
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: individual_hv[x], reverse=True)
        for idx in remaining_sorted:
            if len(selected) >= k:
                break
            selected.append(int(idx))
            selected_set.add(int(idx))

    selected = selected[:k]
    subset = points[np.array(selected, dtype=int)]
    return subset


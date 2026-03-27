import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq
    from typing import List, Tuple
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations") from e

    # Basic checks
    if points is None:
        return np.empty((0, 0))
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape
    if n == 0 or k <= 0:
        return np.zeros((0, d))

    # reference point handling
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float).reshape(-1)
    if reference_point.shape[0] != d:
        raise ValueError("reference_point must have same dimension as points")

    # helper: hypervolume computation (robust)
    def hv_compute(arr: np.ndarray) -> float:
        if arr is None or len(arr) == 0:
            return 0.0
        try:
            arr2 = np.asarray(arr, dtype=float)
            if arr2.ndim == 1:
                arr2 = arr2.reshape(1, -1)
            hv = pg.hypervolume(arr2)
            return float(hv.compute(reference_point))
        except Exception:
            return 0.0

    # fast nondominated filter (assume minimization)
    def nondominated_indices(arr: np.ndarray) -> List[int]:
        N = arr.shape[0]
        if N == 0:
            return []
        is_nd = np.ones(N, dtype=bool)
        for i in range(N):
            if not is_nd[i]:
                continue
            # j dominates i if all arr[j] <= arr[i] and any <
            comp = (np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1))
            if np.any(comp):
                is_nd[i] = False
                continue
            # remove those dominated by i
            dominated_by_i = (np.all(arr[i] <= arr, axis=1) & np.any(arr[i] < arr, axis=1))
            is_nd[dominated_by_i] = False
            is_nd[i] = True
        return list(np.where(is_nd)[0])

    # cheap box-volume proxy: product(max(0, ref - x))
    ref_diff_all = np.maximum(reference_point - points, 0.0)
    box_vol_all = np.prod(ref_diff_all, axis=1)

    # Step 1: pareto-filter candidates
    nd_idx = nondominated_indices(points)
    candidates = list(nd_idx)
    # If not enough nondominated, append dominated sorted by box volume
    if len(candidates) < max(1, min(k, n)):
        dominated = [i for i in range(n) if i not in candidates]
        if len(dominated) > 0:
            order = np.argsort(-box_vol_all[dominated])
            candidates.extend([dominated[i] for i in order])

    if len(candidates) == 0:
        candidates = list(range(n))

    # Limit candidate set size to control runtime (keeps complexity bounded)
    cand_limit = max(min(len(candidates), max(10 * k, 200)), k)
    if len(candidates) > cand_limit:
        candidates = sorted(candidates, key=lambda i: -box_vol_all[i])[:cand_limit]

    candidate_set = set(candidates)

    # CELF-style lazy greedy heap:
    # heap entries: (-key, idx, last_eval_size, last_gain)
    # initial key = box_vol (upper bound)
    heap: List[Tuple[float, int, int, float]] = []
    for idx in candidates:
        key = float(box_vol_all[idx])
        heap.append((-key, int(idx), -1, 0.0))
    heapq.heapify(heap)

    selected_indices: List[int] = []
    selected_points: List[np.ndarray] = []
    current_hv = 0.0

    # Cache for last computed marginal gains: idx -> (last_size, last_gain)
    marginal_cache = {}

    target_k = min(k, n)
    max_pops = max(100000, len(heap) * target_k * 5)
    pops = 0

    while len(selected_indices) < target_k and heap and pops < max_pops:
        pops += 1
        neg_key, idx, last_size, last_gain = heapq.heappop(heap)
        if idx not in candidate_set:
            continue
        est = -neg_key

        # If last evaluated at current selection size and exact, accept directly
        if last_size == len(selected_indices) and last_size != -1:
            # Accept candidate
            selected_indices.append(int(idx))
            selected_points.append(points[idx])
            candidate_set.remove(idx)
            current_hv += float(last_gain)  # last_gain is hv(after) - hv(before at eval time)
            marginal_cache.pop(idx, None)
            continue

        # Quick prune: if estimated upper bound is tiny, stop early
        if est <= 1e-14:
            break

        # Compute exact marginal contribution w.r.t current selected set
        if len(selected_points) == 0:
            # singleton HV equals box_vol (if inside reference)
            hv_after = float(box_vol_all[idx])
            true_gain = hv_after
        else:
            tmp = np.vstack([np.asarray(selected_points), points[idx]])
            hv_after = hv_compute(tmp)
            true_gain = hv_after - current_hv

        # store into cache and push back with updated timestamp
        marginal_cache[idx] = (len(selected_indices), float(true_gain))
        heapq.heappush(heap, (-float(true_gain), int(idx), len(selected_indices), float(true_gain)))

    # If not enough selected (heap exhausted), fill by best remaining box-volume among all points
    if len(selected_indices) < target_k:
        remaining = [i for i in range(n) if i not in selected_indices]
        if remaining:
            order = np.argsort(-box_vol_all[remaining])
            for j in order:
                if len(selected_indices) >= target_k:
                    break
                selected_indices.append(int(remaining[j]))
                selected_points.append(points[remaining[j]])
        # recompute current_hv if any selected
        if len(selected_points) > 0:
            current_hv = hv_compute(np.vstack(selected_points))

    selected_array = np.array(selected_points) if len(selected_points) > 0 else np.empty((0, d))

    # Bounded swap refinement: try swapping each selected with a few best non-selected by proxy
    swap_budget = max(10, 5 * min(10, target_k))
    swaps_done = 0
    if len(selected_array) > 0:
        non_selected = [i for i in range(n) if i not in selected_indices]
        # Prioritize non-selected by box volume
        non_selected = sorted(non_selected, key=lambda i: -box_vol_all[i])[:max(50, len(selected_array) * 5)]
        improved = True
        while improved and swaps_done < swap_budget:
            improved = False
            for si in range(len(selected_array)):
                if swaps_done >= swap_budget:
                    break
                cur_idx = selected_indices[si]
                for ns_idx in non_selected:
                    if ns_idx in selected_indices:
                        continue
                    trial = selected_array.copy()
                    trial[si] = points[ns_idx]
                    trial_hv = hv_compute(trial)
                    if trial_hv > current_hv + 1e-12:
                        # accept swap
                        old_idx = selected_indices[si]
                        selected_indices[si] = int(ns_idx)
                        selected_array = trial
                        current_hv = trial_hv
                        swaps_done += 1
                        improved = True
                        # update non_selected pool
                        try:
                            non_selected.remove(ns_idx)
                        except ValueError:
                            pass
                        if old_idx not in non_selected:
                            non_selected.append(int(old_idx))
                        break
                if improved:
                    break

    final = selected_array
    # Ensure exactly k rows in the result (pad by best remaining or repeat last)
    final_count = final.shape[0]
    if final_count > k:
        final = final[:k]
    elif final_count < k:
        needed = k - final_count
        remaining = [i for i in range(n) if i not in selected_indices]
        if remaining:
            order = np.argsort(-box_vol_all[remaining])
            take = min(len(order), needed)
            to_add = [points[remaining[i]] for i in order[:take]]
            if final_count == 0 and len(to_add) > 0:
                final = np.vstack(to_add)
            elif len(to_add) > 0:
                final = np.vstack([final, np.vstack(to_add)])
            needed = k - final.shape[0]
        # if still short (n < k), repeat last row
        while final.shape[0] < k:
            if final.shape[0] == 0:
                final = np.zeros((1, d))
            final = np.vstack([final, final[-1]])
    return np.asarray(final)


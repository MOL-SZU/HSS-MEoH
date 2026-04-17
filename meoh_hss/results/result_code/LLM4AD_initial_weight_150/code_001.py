import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import math
    import random
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape

    if k <= 0:
        return np.empty((0, d))
    k = min(k, n)

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def hv_of(arr_like):
        if arr_like is None:
            return 0.0
        arr = np.asarray(arr_like)
        if arr.size == 0:
            return 0.0
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        hv = pg.hypervolume(arr)
        return float(hv.compute(reference_point))

    # Precompute single-point hypervolumes (useful heuristics / weights)
    single_hv = np.zeros(n, dtype=float)
    for i in range(n):
        single_hv[i] = hv_of(points[i])

    # Parameters (tunable)
    n_restarts = max(3, min(12, n // max(1, k)))  # number of multi-start restarts
    iters_per_restart = max(60, 8 * k)            # iterations per restart
    sample_unselected = min(max(20, n // 10), n - k) if n - k > 0 else 0
    sample_selected = min(6, k)
    initial_temp = max(1e-6, max(single_hv) * 0.5 if single_hv.size > 0 else 1.0)
    final_temp = 1e-4
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max(1, iters_per_restart))
    rng = random.Random()

    best_global_subset = None
    best_global_hv = -math.inf

    # Helper: k-means++ style diverse seeding (use L2 distance in objective space)
    def kmeanspp_seed():
        # choose first as the one with largest single hv (diverse + high quality)
        first = int(np.argmax(single_hv))
        seeds = [first]
        if k == 1:
            return seeds
        # distances squared from nearest seed
        dist2 = np.sum((points - points[first]) ** 2, axis=1)
        for _ in range(1, k):
            # prevent selecting already chosen
            dist2[list(seeds)] = 0.0
            if dist2.sum() <= 0:
                # choose random remaining
                choices = [i for i in range(n) if i not in seeds]
                if not choices:
                    break
                nxt = rng.choice(choices)
                seeds.append(nxt)
                continue
            probs = dist2 / dist2.sum()
            # sample proportional to distance (diversity)
            nxt = rng.choices(range(n), weights=probs, k=1)[0]
            if nxt in seeds:
                # fallback uniform pick from remaining
                choices = [i for i in range(n) if i not in seeds]
                if not choices:
                    break
                nxt = rng.choice(choices)
            seeds.append(int(nxt))
        # if duplicates or <k, fill by top single_hv
        if len(seeds) < k:
            for idx in np.argsort(-single_hv):
                if int(idx) not in seeds:
                    seeds.append(int(idx))
                if len(seeds) == k:
                    break
        return seeds[:k]

    # Main multi-start simulated annealing loop
    for restart in range(n_restarts):
        # Seed initial subset
        seed_indices = kmeanspp_seed()
        selected_indices = list(seed_indices)
        selected_mask = np.zeros(n, dtype=bool)
        selected_mask[selected_indices] = True
        selected_points = points[selected_indices].copy()
        curr_hv = hv_of(selected_points)
        best_local_hv = curr_hv
        best_local_indices = selected_indices.copy()
        best_local_points = selected_points.copy()

        T = initial_temp

        for it in range(iters_per_restart):
            # propose move: with prob 0.6 do greedy-sampled replacement, else random swap
            if n - k == 0:
                break  # nothing to swap
            if rng.random() < 0.6:
                # Greedy sampled replacement:
                # sample a small set of selected candidates to consider removing
                sel_pool = rng.sample(selected_indices, min(sample_selected, len(selected_indices)))
                # sample a pool of unselected candidates to insert
                unselected_all = [i for i in range(n) if not selected_mask[i]]
                samp_un = rng.sample(unselected_all, min(sample_unselected, len(unselected_all)))
                best_move = None
                best_move_hv = -math.inf
                # evaluate replacement for sampled pairs (remove one sel, insert one unselected)
                for rem in sel_pool:
                    # base set without 'rem'
                    base_idx = [idx for idx in selected_indices if idx != rem]
                    base_pts = points[base_idx]
                    for ins in samp_un:
                        cand_set = np.vstack([base_pts, points[ins]])
                        hv_cand = hv_of(cand_set)
                        if hv_cand > best_move_hv:
                            best_move_hv = hv_cand
                            best_move = (rem, ins, base_idx, cand_set)
                if best_move is None:
                    T *= cooling_rate
                    continue
                rem, ins, base_idx, cand_set = best_move
                delta = best_move_hv - curr_hv
                accept = False
                if delta >= 0:
                    accept = True
                else:
                    # accept with simulated annealing probability
                    try:
                        prob = math.exp(delta / T) if T > 0 else 0.0
                    except OverflowError:
                        prob = 0.0
                    if rng.random() < prob:
                        accept = True
                if accept:
                    # apply move
                    selected_indices = base_idx + [ins]
                    selected_mask[rem] = False
                    selected_mask[ins] = True
                    selected_points = cand_set.copy()
                    curr_hv = best_move_hv
                    if curr_hv > best_local_hv:
                        best_local_hv = curr_hv
                        best_local_indices = selected_indices.copy()
                        best_local_points = selected_points.copy()
            else:
                # Random swap: pick random selected pos and random unselected candidate
                rem = rng.choice(selected_indices)
                unselected_all = [i for i in range(n) if not selected_mask[i]]
                ins = rng.choice(unselected_all)
                # create candidate set
                base_idx = [idx for idx in selected_indices if idx != rem]
                cand_set = np.vstack([points[base_idx], points[ins]])
                hv_cand = hv_of(cand_set)
                delta = hv_cand - curr_hv
                accept = False
                if delta >= 0:
                    accept = True
                else:
                    try:
                        prob = math.exp(delta / T) if T > 0 else 0.0
                    except OverflowError:
                        prob = 0.0
                    if rng.random() < prob:
                        accept = True
                if accept:
                    selected_indices = base_idx + [ins]
                    selected_mask[rem] = False
                    selected_mask[ins] = True
                    selected_points = cand_set.copy()
                    curr_hv = hv_cand
                    if curr_hv > best_local_hv:
                        best_local_hv = curr_hv
                        best_local_indices = selected_indices.copy()
                        best_local_points = selected_points.copy()
            # occasional pure greedy insertion pass to escape bad local minima
            if it % max(1, (iters_per_restart // 5)) == 0:
                # try to greedily replace the worst of a sampled selection by the best unselected (sampled)
                if n - k > 0:
                    sel_pool = rng.sample(selected_indices, min(sample_selected, len(selected_indices)))
                    unselected_all = [i for i in range(n) if not selected_mask[i]]
                    samp_un = rng.sample(unselected_all, min(sample_unselected, len(unselected_all)))
                    best_impr = 0.0
                    best_swap = None
                    for rem in sel_pool:
                        base_idx = [idx for idx in selected_indices if idx != rem]
                        base_pts = points[base_idx]
                        for ins in samp_un:
                            cand_set = np.vstack([base_pts, points[ins]])
                            hv_cand = hv_of(cand_set)
                            improvement = hv_cand - curr_hv
                            if improvement > best_impr:
                                best_impr = improvement
                                best_swap = (rem, ins, cand_set, hv_cand, base_idx)
                    if best_swap is not None and best_impr > 0:
                        rem, ins, cand_set, hv_cand, base_idx = best_swap
                        selected_indices = base_idx + [ins]
                        selected_mask[rem] = False
                        selected_mask[ins] = True
                        selected_points = cand_set.copy()
                        curr_hv = hv_cand
                        if curr_hv > best_local_hv:
                            best_local_hv = curr_hv
                            best_local_indices = selected_indices.copy()
                            best_local_points = selected_points.copy()

            # cool down
            T *= cooling_rate

        # keep best of this restart
        if best_local_hv > best_global_hv:
            best_global_hv = best_local_hv
            best_global_subset = best_local_points.copy()

    # Final fallback: if none found (shouldn't), pick top-k by single_hv
    if best_global_subset is None:
        idxs = np.argsort(-single_hv)[:k]
        best_global_subset = points[idxs].copy()

    # Ensure shape (k, d)
    best_global_subset = np.asarray(best_global_subset)
    if best_global_subset.shape[0] != k:
        # if it's smaller or larger, adjust
        if best_global_subset.shape[0] < k:
            # fill with best remaining
            sel_mask = np.zeros(n, dtype=bool)
            # try to identify included ones by exact matching (slower but safe)
            for i in range(best_global_subset.shape[0]):
                # find matching row
                matches = np.all(points == best_global_subset[i], axis=1)
                idxs = np.where(matches)[0]
                if idxs.size > 0:
                    sel_mask[idxs[0]] = True
            remaining = [i for i in range(n) if not sel_mask[i]]
            need = k - best_global_subset.shape[0]
            add_idx = np.argsort(-single_hv[remaining])[:need] if remaining else []
            if len(add_idx) > 0:
                add_idx = [remaining[i] for i in add_idx]
                best_global_subset = np.vstack([best_global_subset, points[add_idx]])
            # if still not enough, pad with zeros (shouldn't happen)
            if best_global_subset.shape[0] < k:
                pad = np.zeros((k - best_global_subset.shape[0], d))
                best_global_subset = np.vstack([best_global_subset, pad])
        else:
            best_global_subset = best_global_subset[:k]

    return np.array(best_global_subset)


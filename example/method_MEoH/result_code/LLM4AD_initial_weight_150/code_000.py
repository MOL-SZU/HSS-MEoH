import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    from typing import List
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume calculations") from e

    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape
    if k <= 0 or k > n:
        raise ValueError("k must be > 0 and <= number of points")

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    # Precompute individual box volumes as quick single-point HVs (upper bounds for marginal)
    box_sizes = np.prod(np.maximum(0.0, reference_point - points), axis=1)

    # If all zero volumes, return first k deterministically
    if np.all(box_sizes <= 0):
        return points[:k].copy()

    # Initialize candidate pool and eliminated set
    candidates = list(range(n))
    eliminated: List[int] = []

    rng = np.random.default_rng()

    # Tournament reduction: repeatedly pair candidates and select winners based on pairwise synergy
    # Stop when pool size <= pool_target to allow exact greedy selection
    pool_target = max(min(n, max(2 * k, k + 10)), k)  # do not force below k, but aim for reduction
    # compute single hv cache (use box_sizes for speed)
    hv_single = box_sizes.copy()

    while len(candidates) > pool_target:
        rng.shuffle(candidates)
        next_round: List[int] = []
        i = 0
        while i + 1 < len(candidates):
            a = candidates[i]
            b = candidates[i + 1]
            # compute pair hv
            try:
                hv_pair = pg.hypervolume(np.vstack([points[a], points[b]])).compute(reference_point)
            except Exception:
                hv_pair = pg.hypervolume(np.array([points[a], points[b]])).compute(reference_point)
            # synergy for a when paired with b = hv_pair - hv_single[b]
            # similarly for b
            s_a = hv_pair - hv_single[b]
            s_b = hv_pair - hv_single[a]
            # choose winner: the one with larger synergy; tie-breaker by larger single hv
            if s_a > s_b:
                winner = a
                loser = b
            elif s_b > s_a:
                winner = b
                loser = a
            else:
                # tie: pick one with larger individual box size
                if hv_single[a] >= hv_single[b]:
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
            next_round.append(winner)
            eliminated.append(loser)
            i += 2
        # odd element passes automatically
        if i < len(candidates):
            next_round.append(candidates[i])
        candidates = next_round

        # safety: if reduction stalls (no progress), break
        if len(candidates) <= pool_target:
            break

    # Now we have a reduced pool of candidates (size <= pool_target)
    pool = candidates.copy()

    # If pool size < k, fill pool with best eliminated by single-box hv to ensure enough for greedy
    if len(pool) < k:
        remaining_elim = [idx for idx in eliminated if idx not in pool]
        remaining_elim_sorted = sorted(remaining_elim, key=lambda ii: -hv_single[ii])
        to_add = k - len(pool)
        for idx in remaining_elim_sorted[:to_add]:
            pool.append(idx)

    # Greedy selection from the reduced pool with simple pruning by single-box upper bound
    selected_idx: List[int] = []
    selected_points: List[np.ndarray] = []

    remaining = pool.copy()
    # Precompute individual hv for remaining (hv_single already)
    for _ in range(k):
        if not remaining:
            break
        # current hv
        if selected_points:
            try:
                hv_cur = pg.hypervolume(np.array(selected_points)).compute(reference_point)
            except Exception:
                hv_cur = pg.hypervolume(np.vstack(selected_points)).compute(reference_point)
        else:
            hv_cur = 0.0

        # order candidates by decreasing upper bound
        remaining.sort(key=lambda ii: -hv_single[ii])

        best_contrib = -np.inf
        best_idx = None

        for idx in remaining:
            ub = hv_single[idx]
            # pruning: if ub <= best_contrib, further candidates (with smaller ub) cannot beat
            if ub <= best_contrib + 1e-15:
                break
            # compute marginal hv
            if selected_points:
                try:
                    hv_after = pg.hypervolume(np.vstack([selected_points, points[idx]])).compute(reference_point)
                except Exception:
                    hv_after = pg.hypervolume(np.vstack([selected_points, points[idx]])).compute(reference_point)
            else:
                hv_after = hv_single[idx]
            contrib = hv_after - hv_cur
            if contrib > best_contrib + 1e-15:
                best_contrib = contrib
                best_idx = idx

        if best_idx is None:
            # fallback: choose highest single hv
            best_idx = remaining[0]

        selected_idx.append(best_idx)
        selected_points.append(points[best_idx])
        remaining = [r for r in remaining if r != best_idx]

    # If still short (unlikely), fill from all points by largest box sizes not already selected
    if len(selected_idx) < k:
        avail = [i for i in range(n) if i not in selected_idx]
        avail_sorted = sorted(avail, key=lambda ii: -hv_single[ii])
        for idx in avail_sorted:
            if len(selected_idx) >= k:
                break
            selected_idx.append(idx)
            selected_points.append(points[idx])

    # Final ensure exactly k rows
    final_selected = np.array(selected_points)
    if final_selected.shape[0] > k:
        final_selected = final_selected[:k]
    elif final_selected.shape[0] < k:
        # pad (should not happen) with best remaining by hv_single
        remaining_all = [i for i in range(n) if i not in selected_idx]
        remaining_sorted = sorted(remaining_all, key=lambda ii: -hv_single[ii])
        need = k - final_selected.shape[0]
        if remaining_sorted and need > 0:
            extras = np.array([points[i] for i in remaining_sorted[:need]])
            if final_selected.size == 0:
                final_selected = extras
            else:
                final_selected = np.vstack([final_selected, extras])

    return final_selected.copy()


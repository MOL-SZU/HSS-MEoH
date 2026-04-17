import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation.") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = pts.shape

    if k <= 0:
        return np.empty((0, D), dtype=pts.dtype)
    if N == 0:
        return np.empty((0, D), dtype=pts.dtype)
    if k >= N:
        return pts.copy()

    # reference point handling
    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # helper: compute hypervolume of given points array or index list
    def hv_of_array(arr):
        if arr is None:
            return 0.0
        arr = np.asarray(arr)
        if arr.size == 0:
            return 0.0
        hv = pg.hypervolume(arr)
        return float(hv.compute(reference_point))

    def hv_of_indices(idxs):
        if len(idxs) == 0:
            return 0.0
        return hv_of_array(pts[np.array(idxs, dtype=int), :])

    # Precompute individual hypervolumes (singletons)
    individual_hv = np.empty(N, dtype=float)
    for i in range(N):
        individual_hv[i] = hv_of_indices([i])

    # Normalize points for distance computations to avoid scale imbalance
    # If bbox is degenerate, add tiny eps to avoid division by zero
    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    span = bbox_max - bbox_min
    eps = 1e-12
    span[span <= 0] = 1.0  # if zero span, avoid division by zero (all same coord)
    pts_norm = (pts - bbox_min) / span

    # Farthest-first traversal seeded by the highest individual hypervolume
    selected = []
    selected_set = set()
    # Start with the point of maximum individual hypervolume (tie-break by index)
    start_idx = int(np.argmax(individual_hv))
    selected.append(start_idx)
    selected_set.add(start_idx)

    # Precompute pairwise squared distances to speed up? For large N we avoid full NxN matrix.
    # We'll compute distances on the fly but cache distances to selected points.
    # Maintain array of min squared distances to current selected set.
    diff = pts_norm - pts_norm[start_idx]
    min_sqdist = np.sum(diff * diff, axis=1)

    # parameter to balance diversity vs individual hv when selecting farthest-first
    gamma = 0.6  # weight exponent for hv influence; 0 => pure diversity, 1 => diversity weighted by hv
    # We select next point maximizing (min_dist * (individual_hv ** gamma))
    for _ in range(1, k):
        # compute score for each unselected candidate
        unselected_mask = np.ones(N, dtype=bool)
        for idx in selected_set:
            unselected_mask[idx] = False
        candidates = np.nonzero(unselected_mask)[0]
        if candidates.size == 0:
            break
        # compute min squared distances to selected set (maintain incrementally)
        # min_sqdist already contains current mins, update if new selected added earlier
        # compute score
        # use sqrt of min_sqdist to get distance
        dists = np.sqrt(np.maximum(min_sqdist[candidates], 0.0))
        hv_boost = np.maximum(individual_hv[candidates], 0.0) ** gamma
        scores = dists * hv_boost
        # pick candidate with largest score (tie-break by highest hv then lowest index)
        max_idx = np.argmax(scores)
        chosen = int(candidates[max_idx])
        selected.append(chosen)
        selected_set.add(chosen)
        # update min_sqdist
        newdiff = pts_norm - pts_norm[chosen]
        new_sq = np.sum(newdiff * newdiff, axis=1)
        min_sqdist = np.minimum(min_sqdist, new_sq)

    # Ensure we have exactly k (in rare degenerate numerical cases)
    if len(selected) < k:
        for i in range(N):
            if i not in selected_set:
                selected.append(i)
                selected_set.add(i)
                if len(selected) >= k:
                    break

    # Compute current hypervolume
    current_hv = hv_of_indices(selected)

    # Build Voronoi-like clusters: assign each point to nearest selected (by normalized distance)
    def assign_clusters(sel_indices):
        sel_pts = pts_norm[np.array(sel_indices, dtype=int), :]
        # compute distances N x k
        # to avoid huge memory for extreme N*k, do in chunks if necessary (but keep simple here)
        dists = np.linalg.norm(pts_norm[:, None, :] - sel_pts[None, :, :], axis=2)
        # closest selected index per point
        closest = np.argmin(dists, axis=1)
        clusters = {si: [] for si in range(len(sel_indices))}
        for pt_idx, c in enumerate(closest):
            clusters[c].append(pt_idx)
        return clusters

    clusters = assign_clusters(selected)

    # Local improvement: medoid-like best 1-for-1 swaps within cluster-local candidate pools
    MAX_ITERS = 60
    iter_num = 0
    improved = True

    # Candidate pool sizes
    cluster_pool_max = 60  # per cluster, top candidates by individual_hv to consider
    global_pool_max = 200  # global top candidates in case cluster pool insufficient

    # Precompute global candidate ranking by individual_hv (descending)
    global_candidates_by_hv = list(np.argsort(-individual_hv))

    while improved and iter_num < MAX_ITERS:
        iter_num += 1
        improved = False
        # reassign clusters each iteration
        clusters = assign_clusters(selected)
        # iterate clusters and try to improve by replacing the medoid (selected element for this cluster)
        for cluster_id, sel_idx in enumerate(selected):
            # points assigned to this cluster (indices in original pts)
            cluster_points = clusters.get(cluster_id, [])
            # build candidate pool: unselected points inside this cluster ordered by individual_hv
            cluster_unselected = [i for i in cluster_points if i not in selected_set]
            if len(cluster_unselected) == 0:
                # fallback to some global candidates
                candidate_pool = [i for i in global_candidates_by_hv if i not in selected_set][:global_pool_max]
            else:
                # take top by individual_hv in cluster
                cluster_unselected_sorted = sorted(cluster_unselected, key=lambda x: -individual_hv[x])
                candidate_pool = cluster_unselected_sorted[:cluster_pool_max]
            if not candidate_pool:
                continue
            # Evaluate best replacement in the pool for this selected point
            best_local_hv = current_hv
            best_replacement = None
            # prepare base set (all selected except the one to replace)
            base_set = [idx for idx in selected if idx != sel_idx]
            # Try each candidate in the candidate pool
            for cand in candidate_pool:
                new_set = base_set + [int(cand)]
                new_hv = hv_of_indices(new_set)
                if new_hv > best_local_hv + 1e-12:
                    best_local_hv = new_hv
                    best_replacement = int(cand)
            if best_replacement is not None:
                # perform the swap
                selected_set.remove(sel_idx)
                selected_set.add(best_replacement)
                # replace in selected list at the correct position (preserve order where possible)
                for p, val in enumerate(selected):
                    if val == sel_idx:
                        selected[p] = best_replacement
                        break
                current_hv = best_local_hv
                improved = True
                # update clusters after a successful swap and break to restart loop (best-improvement per iter)
                break
        # if we performed a swap, continue to next iteration to recompute clusters and candidates
    # Final selection: return points for selected indices (in the order of selected list)
    final_indices = [int(x) for x in selected[:k]]
    subset = pts[np.array(final_indices, dtype=int), :].copy()
    return subset


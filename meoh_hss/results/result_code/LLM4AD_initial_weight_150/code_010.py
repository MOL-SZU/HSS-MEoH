import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import random
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computation") from e

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = points.shape
    if k <= 0:
        return np.empty((0, d))
    k = int(min(k, n))

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def hv_of(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return 0.0
        arr2 = np.atleast_2d(arr)
        hv = pg.hypervolume(arr2)
        return float(hv.compute(reference_point))

    # Precompute one-point hv to guide representative selection
    indiv_hv = np.zeros(n, dtype=float)
    for i in range(n):
        indiv_hv[i] = hv_of(points[i:i+1])

    # If k == n simply return all points
    if k >= n:
        return points.copy()

    # Lightweight KMeans++ initialization and Lloyd's iteration
    def kmeans_pp(X, n_centroids, max_iter=30):
        # X: (n, d)
        n_samples = X.shape[0]
        if n_centroids <= 0:
            return np.zeros((0, d))
        # choose first center deterministically: the point with max L2 norm (spread)
        centers = np.empty((n_centroids, d), dtype=float)
        first_idx = int(np.argmax(np.linalg.norm(X, axis=1)))
        centers[0] = X[first_idx]
        # distances squared to nearest center
        closest_dist_sq = np.sum((X - centers[0])**2, axis=1)
        for c in range(1, n_centroids):
            # probabilistic selection proportional to distance^2
            probs = closest_dist_sq / (closest_dist_sq.sum() + 1e-12)
            # choose index by probabilities (but deterministic fallback if degenerate)
            try:
                idx = np.random.choice(n_samples, p=probs)
            except Exception:
                idx = int(np.argmax(closest_dist_sq))
            centers[c] = X[idx]
            # update distances
            dist_sq = np.sum((X - centers[c])**2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
        # Lloyd's iterations
        labels = np.full(n_samples, -1, dtype=int)
        for it in range(max_iter):
            # assign
            dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)  # (n, k)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # update centers, handle empty clusters
            for j in range(n_centroids):
                members = X[labels == j]
                if members.shape[0] == 0:
                    # reinitialize to a random sample
                    centers[j] = X[np.random.randint(0, n_samples)]
                else:
                    centers[j] = members.mean(axis=0)
        return labels, centers

    # cluster_count = k (we want k groups to choose one representative from each)
    cluster_count = k
    # If n is much larger than k, kmeans is useful; else trivial grouping
    labels, centers = kmeans_pp(points, cluster_count, max_iter=40)

    # For each cluster choose the point with maximum individual hv as initial representative
    selected_indices = []
    selected_mask = np.zeros(n, dtype=bool)
    for cl in range(cluster_count):
        members = np.where(labels == cl)[0]
        if members.size == 0:
            continue
        best = members[int(np.argmax(indiv_hv[members]))]
        selected_indices.append(int(best))
        selected_mask[best] = True

    # If due to empty clusters we selected fewer than k, fill remaining with highest indiv_hv
    if len(selected_indices) < k:
        remaining = [i for i in np.argsort(-indiv_hv) if not selected_mask[i]]
        for idx in remaining[:(k - len(selected_indices))]:
            selected_indices.append(int(idx))
            selected_mask[idx] = True

    selected_indices = selected_indices[:k]
    selected_points = points[np.array(selected_indices)]
    curr_hv = hv_of(selected_points)

    tol = 1e-12

    # Focused intra-cluster greedy replacements:
    # For each cluster, try to replace the current representative with another member of same cluster if it improves global hv.
    max_intra_iters = 100
    improved = True
    intra_iter = 0
    # Build mapping from cluster -> indices
    cluster_members = {}
    for cl in range(cluster_count):
        cluster_members[cl] = list(np.where(labels == cl)[0])
    # Map selected index to its cluster (if any)
    idx_to_cluster = {idx: int(labels[idx]) for idx in selected_indices}

    while improved and intra_iter < max_intra_iters:
        intra_iter += 1
        improved = False
        # iterate clusters; for each cluster where we have a selected index, search for best local replacement
        for sel_pos, sel_idx in enumerate(selected_indices):
            cl = int(labels[sel_idx]) if sel_idx < n else None
            if cl is None:
                continue
            best_gain = 0.0
            best_cand = None
            for cand in cluster_members.get(cl, []):
                if selected_mask[cand]:
                    continue
                # test swap sel_idx -> cand
                candidate_set = selected_points.copy()
                candidate_set[sel_pos] = points[cand]
                hv_cand = hv_of(candidate_set)
                gain = hv_cand - curr_hv
                if gain > best_gain + tol:
                    best_gain = gain
                    best_cand = (cand, hv_cand)
            if best_cand is not None:
                cand_idx, hv_after = best_cand
                # perform swap
                old_idx = selected_indices[sel_pos]
                selected_mask[old_idx] = False
                selected_indices[sel_pos] = int(cand_idx)
                selected_mask[cand_idx] = True
                selected_points[sel_pos] = points[cand_idx].copy()
                curr_hv = hv_after
                improved = True
        # end for
    # end while

    # Randomized cross-cluster greedy swaps: try to improve by swapping any unselected with any selected
    max_cross_iters = 500
    cross_iter = 0
    unselected_pool = [i for i in range(n) if not selected_mask[i]]
    # If there are no unselected, nothing to do
    while cross_iter < max_cross_iters and len(unselected_pool) > 0:
        cross_iter += 1
        # sample a small batch of unselected candidates to test (diversity)
        sample_unselected = random.sample(unselected_pool, min(40, len(unselected_pool)))
        improved_this_iter = False
        # Iterate selected positions and try to replace with best from sample_unselected
        for pos in range(len(selected_indices)):
            best_gain = 0.0
            best_cand = None
            for cand in sample_unselected:
                candidate_set = selected_points.copy()
                candidate_set[pos] = points[cand]
                hv_cand = hv_of(candidate_set)
                gain = hv_cand - curr_hv
                if gain > best_gain + tol:
                    best_gain = gain
                    best_cand = (cand, hv_cand)
            if best_cand is not None:
                cand_idx, hv_after = best_cand
                old_idx = selected_indices[pos]
                selected_mask[old_idx] = False
                selected_indices[pos] = int(cand_idx)
                selected_mask[cand_idx] = True
                selected_points[pos] = points[cand_idx].copy()
                curr_hv = hv_after
                # update unselected pool
                unselected_pool = [i for i in range(n) if not selected_mask[i]]
                improved_this_iter = True
                break  # restart sampling after successful swap
        if not improved_this_iter:
            # small probability to reshuffle sample to escape local plateaus
            if random.random() < 0.1:
                continue
            else:
                break

    # Final deterministic pass: try every possible replacement with a limited budget to catch missed improvements
    budget = 200
    tried = 0
    unselected_indices = [i for i in range(n) if not selected_mask[i]]
    for cand in unselected_indices:
        if tried >= budget:
            break
        tried += 1
        best_gain = 0.0
        best_pos = None
        best_hv = None
        for pos in range(len(selected_indices)):
            candidate_set = selected_points.copy()
            candidate_set[pos] = points[cand]
            hv_cand = hv_of(candidate_set)
            gain = hv_cand - curr_hv
            if gain > best_gain + tol:
                best_gain = gain
                best_pos = pos
                best_hv = hv_cand
        if best_pos is not None:
            old_idx = selected_indices[best_pos]
            selected_mask[old_idx] = False
            selected_indices[best_pos] = int(cand)
            selected_mask[cand] = True
            selected_points[best_pos] = points[cand].copy()
            curr_hv = best_hv
            # update unselected list
            unselected_indices = [i for i in range(n) if not selected_mask[i]]

    # Ensure we return exactly k points (pad if necessary)
    if len(selected_indices) < k:
        for i in range(n):
            if not selected_mask[i]:
                selected_indices.append(i)
                selected_mask[i] = True
                selected_points = np.vstack([selected_points, points[i].copy()])
                if len(selected_indices) == k:
                    break
    # Trim in case of any accidental extra
    selected_points = np.asarray(selected_points, dtype=float)[:k]

    return selected_points.copy()


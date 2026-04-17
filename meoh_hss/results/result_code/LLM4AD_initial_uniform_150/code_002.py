import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np

    # Deterministic behavior
    np.random.seed(2027)

    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")

    N, D = points.shape
    if k <= 0 or N == 0:
        return np.zeros((0, D), dtype=points.dtype)

    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    if k >= N:
        return points.copy()

    # --- Parameters (changed settings) ---
    # fewer directions but still scaling with D, using sparse Dirichlet sampling
    num_dir = int(min(250, max(40, 6 * D)))
    eps = 1e-12
    power_p = 0.5  # sqrt concave transform to soften marginal gains more strongly

    # Sample positive directions using Dirichlet with alpha < 1 to encourage sparsity/diversity
    alpha = max(0.5, 0.7)  # concentration parameter (can be tuned)
    G = np.random.dirichlet(alpha=np.full(D, alpha), size=num_dir).astype(float)
    # normalize directions in L2 to give balanced axis contributions
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    W = G / norms  # shape (num_dir, D), positive directions

    # Compute positive differences from reference
    delta = points - reference_point  # (N, D)
    delta_pos = np.clip(delta, a_min=0.0, a_max=None)

    # Prefilter: use a cheap proxy (L1 sum of positive deltas) to reduce candidate set for speed
    # keep at least max(5*k, 20) candidates but not more than N
    if N > max(5 * k, 50):
        score_proxy = np.sum(delta_pos, axis=1)  # cheap proxy for relevance
        cand_size = min(N, max(5 * k, 20))
        cand_idx = np.argsort(-score_proxy)[:cand_size]
    else:
        cand_idx = np.arange(N, dtype=int)

    # Compute directional scalarizations only for candidates for efficiency
    # hv_vals_cand shape: (Nc, num_dir)
    delta_cand = delta_pos[cand_idx]  # (Nc, D)
    with np.errstate(divide='ignore', invalid='ignore'):
        hv_vals_cand = np.min(delta_cand[:, None, :] / (W[None, :, :] + eps), axis=2)
    hv_vals_cand = np.clip(hv_vals_cand, a_min=0.0, a_max=None)

    # Normalize per-direction by median (more robust than mean) to avoid domination by outliers
    dir_med = np.median(hv_vals_cand, axis=0)  # (num_dir,)
    dir_scale = np.where(dir_med <= 0.0, 1.0, dir_med)
    hv_norm_cand = hv_vals_cand / (dir_scale[None, :] + eps)

    # Direction weights: use softmax over log-median with a temperature to favor informative directions
    temp = 0.6
    log_med = np.log(dir_scale + eps)
    shifted = log_med - np.max(log_med)
    exps = np.exp(shifted / max(temp, eps))
    dir_weights = exps / (np.sum(exps) + eps)
    if not np.isfinite(np.sum(dir_weights)) or np.sum(dir_weights) <= 0.0:
        dir_weights = np.ones_like(dir_weights) / float(len(dir_weights))

    # Precompute concave-transformed values (power) for faster marginal gain computation
    hv_pow_cand = np.power(hv_norm_cand, power_p)  # (Nc, num_dir)

    # Greedy selection based on marginal gains in transformed space among candidates
    Nc = hv_pow_cand.shape[0]
    cur_best = np.zeros(num_dir, dtype=float)
    selected_indices = []
    available = np.ones(Nc, dtype=bool)

    for _ in range(k):
        dif = hv_pow_cand - cur_best[None, :]  # (Nc, num_dir)
        dif_clipped = np.clip(dif, a_min=0.0, a_max=None)
        gains = dif_clipped.dot(dir_weights)  # (Nc,)

        # mask already chosen
        gains[~available] = -np.inf

        best_local = int(np.argmax(gains))
        if not available[best_local] or gains[best_local] == -np.inf or gains[best_local] <= 0.0:
            break

        chosen_global = int(cand_idx[best_local])
        selected_indices.append(chosen_global)
        available[best_local] = False

        # update current best in transformed space with the chosen candidate's hv_pow
        cur_best = np.maximum(cur_best, hv_pow_cand[best_local, :])

    # Fallback: if fewer than k selected, fill remaining using global proxy scores (L1) excluding already selected
    if len(selected_indices) < k:
        remaining_needed = k - len(selected_indices)
        total_proxy = np.sum(delta_pos, axis=1)  # (N,)
        # exclude already selected
        if selected_indices:
            total_proxy[np.array(selected_indices, dtype=int)] = -np.inf
        extra_idx = list(np.argsort(-total_proxy)[:remaining_needed])
        for idx in extra_idx:
            if total_proxy[idx] > -np.inf:
                selected_indices.append(int(idx))
            if len(selected_indices) == k:
                break

    selected_indices = selected_indices[:k]
    subset = points[np.array(selected_indices, dtype=int), :].copy()
    return subset


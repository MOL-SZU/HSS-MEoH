import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    N, D = points.shape
    if N == 0 or k <= 0:
        return np.zeros((0, D)) if k == 0 else np.zeros((k, D))

    # 确定参考点
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (D,):
        raise ValueError("reference_point must have shape (D,)")

    # Deterministic RNG
    rng = np.random.RandomState(2025)

    # 参数：权重向量数（近似精度），可以调节以改变评分精细度/速度权衡
    num_weights = 200  # changed parameter: use more weights for better HVC approx

    # 生成正向权重向量（保持正数并归一化）
    # 使用 Dirichlet 风格生成确保正数并可控
    W = rng.gamma(shape=1.0, scale=1.0, size=(num_weights, D))
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12  # 单位向量，避免零除

    # 1) 预筛选：过滤被其他点严格支配的点（假设目标为最小化）
    # dominated: 如果存在 j != i 使得 points[j] <= points[i] 所有维，并且某维严格<
    pts = points
    is_dominated = np.zeros(N, dtype=bool)
    # 使用批量比较以加速
    for i in range(N):
        if is_dominated[i]:
            continue
        # 如果 any other dominates i:
        le = np.all(pts <= pts[i], axis=1)
        lt = np.any(pts < pts[i], axis=1)
        dom_by = np.where(le & lt)[0]
        # Exclude self
        dom_by = dom_by[dom_by != i]
        if dom_by.size > 0:
            is_dominated[i] = True
    nd_indices = np.where(~is_dominated)[0]
    dom_indices = np.where(is_dominated)[0]

    # 2) 计算基于权重的标量化量 G[i,w] = min_j ( (ref_j - points[i,j]) / W[w,j] )
    # 这用于近似超体积贡献（achievement scalarizing-like approx）
    # Compute diff = ref - points -> shape (N, D)
    diff = reference_point.reshape(1, D) - pts  # (N, D)
    # Broadcast divide by W (num_weights, D): result shape (N, num_weights, D)
    # G = min over D -> shape (N, num_weights)
    # Ensure W positive (it is), and protect divide by small values
    eps = 1e-12
    W_safe = np.maximum(W, eps)
    diff_div = diff[:, None, :] / W_safe[None, :, :]  # (N, num_weights, D)
    G = np.min(diff_div, axis=2)  # (N, num_weights)
    # Negative values correspond to points outside reference dominance in some directions; clamp at 0 for HV contribution
    G_pos = np.maximum(G, 0.0)

    # 3) 贪心选择：维护 current_max over weights; 增量贡献为 sum(max(0, G_i - current_max))
    current_max = np.zeros(num_weights, dtype=float)
    selected = []
    selected_mask = np.zeros(N, dtype=bool)

    # Helper: greedy selection from a pool of indices
    def greedy_from_pool(pool_indices):
        nonlocal current_max, selected, selected_mask
        pool = list(pool_indices)
        # Precompute sum of G_pos for quick single-point contributions if needed
        # Iterate until pool exhausted or selected size reaches k
        while pool and len(selected) < k:
            # Compute incremental deltas for the pool
            # delta_i = sum(max(0, G_pos[i] - current_max))
            # Vectorized: G_pos[pool] shape (P, num_weights)
            G_pool = G_pos[pool]  # (P, W)
            # Compute differences and sum positives
            diffs = G_pool - current_max[None, :]
            deltas = np.sum(np.maximum(diffs, 0.0), axis=1)
            # Choose best (deterministic tie-breaker: lower index wins because argmax returns first)
            best_idx_in_pool = int(np.argmax(deltas))
            best_global_idx = pool[best_idx_in_pool]
            # If delta is non-positive and nothing more can be gained, we still select deterministically highest single scalar
            # (This handles degenerate cases)
            if deltas[best_idx_in_pool] <= 0:
                # fallback: pick point with largest single G_pos sum among remaining
                sums = np.sum(G_pool, axis=1)
                best_idx_in_pool = int(np.argmax(sums))
                best_global_idx = pool[best_idx_in_pool]
            # Select it
            selected.append(best_global_idx)
            selected_mask[best_global_idx] = True
            # Update current_max
            current_max = np.maximum(current_max, G_pos[best_global_idx])
            # Remove from pool
            pool.pop(best_idx_in_pool)

    # First, greedy on non-dominated set
    if nd_indices.size > 0:
        greedy_from_pool(nd_indices)

    # If still need more selections, consider dominated points
    if len(selected) < k and dom_indices.size > 0:
        # Greedy on dominated pool
        greedy_from_pool(dom_indices)

    # If still fewer than k (e.g., N < k), pad deterministically by cycling selected indices or repeating best point
    if len(selected) < k:
        if len(selected) == 0:
            # No selection was made (shouldn't happen unless N==0), pick the single best by sum G_pos
            best_single = int(np.argmax(np.sum(G_pos, axis=1)))
            selected.append(best_single)
        # Pad by cycling through current selections deterministically
        i = 0
        while len(selected) < k:
            selected.append(selected[i % len(selected)])
            i += 1

    # Ensure exactly k and preserve order of selection (first k)
    selected = selected[:k]
    subset = points[np.array(selected, dtype=int), :]

    return subset


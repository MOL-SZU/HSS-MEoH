import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    try:
        import pygmo as pg
    except Exception as e:
        raise ImportError("pygmo is required for hypervolume computations.") from e

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, D)")
    n, d = pts.shape

    if k <= 0:
        return np.empty((0, d), dtype=pts.dtype)
    if n == 0:
        return np.empty((0, d), dtype=pts.dtype)
    if k >= n:
        return pts.copy()

    if reference_point is None:
        reference_point = np.max(pts, axis=0) * 1.1
    reference_point = np.asarray(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,)")

    def hv_of_indices(idxs):
        if len(idxs) == 0:
            return 0.0
        data = pts[np.array(idxs, dtype=int), :]
        hv = pg.hypervolume(data)
        return float(hv.compute(reference_point))

    # 预计算单点超体积作为启发式评分
    individual_hv = np.empty(n, dtype=float)
    for i in range(n):
        individual_hv[i] = hv_of_indices([i])
    # 防止数值问题
    eps = 1e-12

    # === Phase 1: Farthest-first seeding (k-center) but seeded by high individual HV ===
    # 选择第一个中心为最大 individual_hv 的点以鼓励高贡献点被覆盖
    first_idx = int(np.argmax(individual_hv))
    centers = [first_idx]
    if k > 1:
        # 迭代选取 farthest-first centers
        # 计算每个点到最近中心的距离
        pts_sq = pts  # already np.array
        for _ in range(1, min(k, n)):
            # 计算到现有中心的最小距离
            centers_pts = pts[np.array(centers, dtype=int), :]
            # 使用广播计算距离
            # dist_matrix shape (n, len(centers))
            diff = pts[:, None, :] - centers_pts[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            min_dist2 = np.min(dist2, axis=1)
            # 排除已选中心
            min_dist2[np.array(centers, dtype=int)] = -1.0
            # 选取使距离最大且 individual_hv 较大的点作为折中：
            # 使用 distance * (1 + small_factor * normalized_hv) 做排名
            max_hv = float(np.max(individual_hv)) if n > 0 else 1.0
            hv_norm = individual_hv / (max_hv + eps)
            scores = min_dist2 * (1.0 + 0.2 * hv_norm)
            # 若所有距离都相同或为 -1，则 fallback 选最大 individual_hv 中未选的点
            if np.all(scores <= -0.5):
                candidates = [i for i in range(n) if i not in centers]
                if not candidates:
                    break
                next_idx = int(max(candidates, key=lambda i: individual_hv[i]))
            else:
                next_idx = int(np.argmax(scores))
                if next_idx in centers:
                    # safety fallback
                    candidates = [i for i in range(n) if i not in centers]
                    if not candidates:
                        break
                    next_idx = int(max(candidates, key=lambda i: individual_hv[i]))
            centers.append(next_idx)
            if len(centers) >= k:
                break

    # 如果 centers 少于 k（当 n<k），后面会补齐
    # 将每个点归到最近中心，按簇内 individual_hv 选代表
    centers_arr = np.array(centers, dtype=int)
    # 计算距离到每个 center
    centers_pts = pts[centers_arr, :]
    diff = pts[:, None, :] - centers_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    assign = np.argmin(dist2, axis=1)

    selected_indices = []
    selected_set = set()
    for c_idx_pos, c_global_idx in enumerate(centers_arr):
        members = np.where(assign == c_idx_pos)[0]
        if members.size == 0:
            continue
        # 选簇内 individual_hv 最大的点
        best_local = int(members[np.argmax(individual_hv[members])])
        if best_local not in selected_set:
            selected_indices.append(best_local)
            selected_set.add(best_local)
        if len(selected_indices) >= k:
            break

    # 如果不足 k，则补充剩余 individual_hv 最大的点
    if len(selected_indices) < k:
        remaining = [i for i in range(n) if i not in selected_set]
        remaining_sorted = sorted(remaining, key=lambda x: (-individual_hv[x], x))
        need = k - len(selected_indices)
        for idx in remaining_sorted[:need]:
            selected_indices.append(int(idx))
            selected_set.add(int(idx))

    # 截断或扩展保证长度为 k
    if len(selected_indices) > k:
        selected_indices = selected_indices[:k]
        selected_set = set(selected_indices)
    if len(selected_indices) < k:
        # 若仍然不足，重复最后一个索引直到满足（极端情况）
        while len(selected_indices) < k:
            selected_indices.append(selected_indices[-1] if selected_indices else 0)
            selected_set.add(selected_indices[-1])

    # 计算当前超体积
    hv_current = hv_of_indices(selected_indices)

    # === Phase 2: Focused 1-swap local search among high-potential unselected points ===
    max_swap_iters = 50
    swap_iter = 0
    improved = True
    tol = 1e-12
    candidate_limit = min(200, n)

    # 预先构造候选集：按 individual_hv 降序的未选点
    while improved and swap_iter < max_swap_iters:
        improved = False
        swap_iter += 1
        unselected = [i for i in range(n) if i not in selected_set]
        if not unselected:
            break
        # 只考虑前 candidate_limit 个高潜力点
        unselected_sorted = sorted(unselected, key=lambda x: -individual_hv[x])[:candidate_limit]
        # 遍历候选替换 (尝试快速改善)
        outer_break = False
        for u in unselected_sorted:
            # 估算 u 与当前选集的最小距离（作为多样性启发），可以用于早期跳过非常相似的点
            # 但这里直接尝试替换，优先考虑与个体 hv 高的替换
            for s_pos, s in enumerate(list(selected_indices)):
                if u == s:
                    continue
                # 构造替换后的索引集
                base = selected_indices.copy()
                base[s_pos] = int(u)
                swapped_hv = hv_of_indices(base)
                if swapped_hv > hv_current + tol:
                    # 执行替换
                    old_idx = selected_indices[s_pos]
                    selected_indices[s_pos] = int(u)
                    selected_set.remove(int(old_idx))
                    selected_set.add(int(u))
                    hv_current = swapped_hv
                    improved = True
                    outer_break = True
                    break
            if outer_break:
                break

    # 保证结果顺序稳定：按选出顺序返回前 k 个
    final_indices = [int(idx) for idx in selected_indices[:k]]
    subset = pts[np.array(final_indices, dtype=int), :].copy()
    return subset


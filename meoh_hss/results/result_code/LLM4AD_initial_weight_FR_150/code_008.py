import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    N, D = points.shape

    if k <= 0:
        return np.zeros((0, D), dtype=float)
    if k >= N:
        return points.copy()

    # 参考点处理：如果没有给定，则使用各维最大值的 1.1 倍
    if reference_point is None:
        ref = points.max(axis=0) * 1.1
    else:
        ref = np.asarray(reference_point, dtype=float)
        if ref.ndim == 0:
            ref = np.full((D,), float(ref))
        if ref.size != D:
            raise ValueError("reference_point must have same dimensionality as points")
        # 确保参考点在每维大于等于点的最大值
        max_p = points.max(axis=0)
        if np.any(ref <= max_p):
            ref = np.maximum(ref, max_p * 1.1)

    # 采样区间下界为所有候选点在每维的最小值
    low = points.min(axis=0)

    # Monte-Carlo 样本数量，依据 k 和维度自适应，受上限限制以控制时间
    M = int(min(20000, max(2000, 500 * k * D)))
    # 若区间在某维上宽度为 0，仍可采样（样本沿该维恒等）
    rng = np.random.default_rng()
    spans = ref - low
    # 为避免零区间造成除零，直接构造 samples = low + u * spans（若 spans=0 则为 low）
    u = rng.random((M, D))
    samples = low + u * spans

    # 预计算每个样本是否落在每个点定义的 dominated 超长方体 [point, ref]
    # 条件为 samples >= point 在所有维度上
    # dominated: shape (M, N), dtype=bool
    dominated = (samples[:, None, :] >= points[None, :, :]).all(axis=2)

    # 个体超体积近似（以样本计数表示）
    individual_counts = dominated.sum(axis=0)

    covered = np.zeros(M, dtype=bool)
    selected_indices = []
    available = np.ones(N, dtype=bool)

    for _ in range(min(k, N)):
        # 计算每个可用候选点的边际增益（覆没新样本的数量）
        # 对于可用点 j ， marginal[j] = sum( dominated[:,j] & ~covered )
        not_covered = ~covered
        # 避免创建太大临时数组，使用 vectorized sum on boolean
        new_cover_counts = np.where(available, (dominated & not_covered[:, None]).sum(axis=0), -1)
        # 选取边际贡献最大的点（若所有边际为0则也选择最大个体贡献的点）
        best_idx = int(np.argmax(new_cover_counts))
        # 如果最佳仍不可用（应不会发生）或者为 -1，则退而选个体贡献最大的可用点
        if (not available[best_idx]) or (new_cover_counts[best_idx] < 0):
            avail_idx = np.where(available)[0]
            if avail_idx.size == 0:
                break
            best_idx = int(avail_idx[np.argmax(individual_counts[avail_idx])])

        selected_indices.append(best_idx)
        available[best_idx] = False
        # 更新已覆盖样本集
        covered |= dominated[:, best_idx]

        # 提前停止：如果所有样本都被覆盖，则后续点边际为0，直接用剩余最大个体贡献点补齐
        if covered.all():
            break

    # 如尚未选够 k 个点（例如已覆盖所有样本），补齐剩余点按个体贡献降序选择
    if len(selected_indices) < k:
        remaining = np.where(available)[0]
        if remaining.size > 0:
            order = remaining[np.argsort(-individual_counts[remaining])]
            need = k - len(selected_indices)
            selected_indices.extend(list(order[:need]))

    selected_indices = selected_indices[:k]
    return points[np.array(selected_indices, dtype=int), :]


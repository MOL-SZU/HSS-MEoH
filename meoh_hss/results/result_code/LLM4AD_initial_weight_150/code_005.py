import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import heapq

    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype=float)
    else:
        points = points.astype(float, copy=False)

    n, d = points.shape
    if k <= 0 or k > n:
        raise ValueError("k must be > 0 and <= number of points")

    # 处理参考点
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    reference_point = np.array(reference_point, dtype=float)
    if reference_point.shape != (d,):
        raise ValueError("reference_point must have shape (D,) matching point dimensionality")

    # 确保 reference_point 在每个维度上至少略大于点的最小值
    lower = np.min(points, axis=0)
    eps = 1e-12
    reference_point = np.maximum(reference_point, lower + eps)

    # 采样数：使用与维度和 k 相关的经验公式，但与原来不同以改变参数设定
    # 使用一个更保守的上限，且与维度相关以应对高维
    sample_count = int(min(15000, max(1000, 200 * max(1, min(k, d)))))
    sample_count = max(800, sample_count)

    # 使用 Latin Hypercube Sampling (LHS) 以提高样本利用率（相比纯随机）
    rng = np.random.RandomState(123)
    span = reference_point - lower
    zero_span = span <= 0
    span_adj = span.copy()
    span_adj[zero_span] = 1.0  # 防止0跨度，方便采样
    m = sample_count

    # LHS implementation: 对每一维生成一个随机排列并在每个区间内随机采样
    samples_unit = np.empty((m, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(m)
        # within each stratum we pick a random offset
        offsets = rng.rand(m)
        samples_unit[:, j] = (perm + offsets) / float(m)
    samples = lower + samples_unit * span_adj

    # 计算 coverage 矩阵：coverage[i, t] = True 如果样本 t 在点 i 的盒子内（points[i] <= sample <= reference_point）
    def compute_coverage_matrix(pts: np.ndarray, smpls: np.ndarray) -> np.ndarray:
        N, D = pts.shape
        T = smpls.shape[0]
        cov = np.zeros((N, T), dtype=bool)
        # 分块，避免内存峰值
        # chunk_size heuristics: 目标是使 block (chunk_size * T * D) 大小合理
        # 这里使用以 T 为基准的经验值
        max_mem_elements = int(1e7)
        chunk_size = max(1, int(max_mem_elements // max(1, T)))
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            # 比较 shape = (end-start, T, D)
            block = pts[start:end, None, :] <= smpls[None, :, :]
            cov[start:end] = np.all(block, axis=2)
        return cov

    coverage = compute_coverage_matrix(points, samples)  # shape (n, sample_count)

    # 每个点独立覆盖的样本数（用于初始估计）
    per_point_counts = coverage.sum(axis=1)  # length n

    # 计算每个点对应的真实盒子体积：prod(reference - point)
    raw_vols = reference_point[None, :] - points  # shape (n,d)
    # 若某个维度 reference <= point，则体积为0或负，裁剪为eps以避免数值问题
    raw_vols[raw_vols < eps] = eps
    box_volumes = np.prod(raw_vols, axis=1)
    # 防止零体积
    box_volumes = np.maximum(box_volumes, eps)

    # 新的评分函数参数：将样本覆盖数与盒子体积结合，采用幂律缩放体积（beta）
    beta = 0.5  # 与原算法不同的参数设置，平衡体积和样本覆盖
    initial_scores = per_point_counts * (box_volumes ** beta)

    # 使用 CELF 懒惰贪心，heap 元素: (-score, idx, last_updated_selected_size)
    heap = []
    for i in range(n):
        heap.append((-initial_scores[i], i, 0))
    heapq.heapify(heap)

    selected_set = set()
    covered_by_selected = np.zeros(sample_count, dtype=bool)
    selected_indices = []
    current_selected_size = 0

    while len(selected_indices) < k and heap:
        neg_score, idx, last_updated = heapq.heappop(heap)
        # 如果条目时间戳与当前一致，说明评分是相对于当前已选集合更新后的，可选
        if last_updated == current_selected_size:
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            covered_by_selected |= coverage[idx]
            current_selected_size += 1
            continue
        # 否则重新计算真实边际（覆盖的新增样本数），并用体积加权得到新的评分，压回堆
        newly_covered = np.count_nonzero(coverage[idx] & (~covered_by_selected))
        new_score = newly_covered * (box_volumes[idx] ** beta)
        heapq.heappush(heap, (-new_score, idx, current_selected_size))

    # 如果堆耗尽但还没选够，则补齐剩余未选点（按未修正的初始评分排序补齐）
    if len(selected_indices) < k:
        for _, i, _ in sorted(heap, key=lambda x: x[0]):
            if i not in selected_set:
                selected_indices.append(i)
                selected_set.add(i)
                if len(selected_indices) >= k:
                    break
        # 若仍不足，遍历所有点补齐
        if len(selected_indices) < k:
            for i in range(n):
                if i not in selected_set:
                    selected_indices.append(i)
                    selected_set.add(i)
                    if len(selected_indices) >= k:
                        break

    subset = points[np.array(selected_indices[:k], dtype=int)]
    return subset


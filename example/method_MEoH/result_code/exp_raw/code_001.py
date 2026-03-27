import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    HSS (Hypervolume-based Subset Selection): 贪心选择超体积贡献最大的 k 个点

    参数:
        points: np.ndarray
            所有候选点，形状为 (N, D)，其中 N 是点数，D 是目标维度
        k: int
            需要选择的点数
        reference_point: np.ndarray, optional
            参考点，形状为 (D,)。如果为 None，则使用 points 的最大值 * 1.1

    返回:
        subset: np.ndarray
            选出的子集，形状为 (k, D)
    """
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    N, D = points.shape
    if k >= N:
        return points.copy()
    
    selected = []
    remaining = list(range(N))
    
    while len(selected) < k and remaining:
        best_idx = -1
        best_score = -np.inf
        
        if selected:
            selected_pts = points[selected]
            min_distances = np.min(np.linalg.norm(points[remaining][:, np.newaxis, :] - selected_pts, axis=2), axis=1)
            max_min_dist = np.max(min_distances)
            mean_min_dist = np.mean(min_distances)
            
            if max_min_dist > mean_min_dist:
                spread_factor = (min_distances - mean_min_dist) / (max_min_dist - mean_min_dist)
            else:
                spread_factor = np.ones(len(min_distances))
            
            hv_contribs = np.prod(reference_point - points[remaining], axis=1)
            max_hv = np.max(hv_contribs)
            min_hv = np.min(hv_contribs)
            
            if max_hv > min_hv:
                hv_norm = (hv_contribs - min_hv) / (max_hv - min_hv)
            else:
                hv_norm = np.ones(len(hv_contribs))
            
            alpha = len(selected) / k
            scores = alpha * hv_norm + (1 - alpha) * spread_factor
            
            best_remaining_idx = np.argmax(scores)
            best_idx = remaining[best_remaining_idx]
        else:
            hv_contribs = np.prod(reference_point - points, axis=1)
            best_idx = np.argmax(hv_contribs)
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return points[selected]


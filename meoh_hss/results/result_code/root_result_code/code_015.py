import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    n, m = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    # 1. 贪心初始化
    selected = []
    remaining = list(range(n))
    
    if k == n:
        return points.copy()
    
    # 快速计算初始贡献
    contributions = np.array([np.prod(reference_point - points[idx]) for idx in remaining])
    
    for i in range(k):
        best_pos = np.argmax(contributions)
        best_idx = remaining[best_pos]
        selected.append(best_idx)
        del remaining[best_pos]
        
        # 更新剩余点的贡献
        if i < k-1 and remaining:
            temp_set = points[selected]
            for j, idx in enumerate(remaining):
                combined = np.vstack([temp_set, points[idx].reshape(1, -1)])
                hv = pg.hypervolume(combined)
                contributions[j] = hv.compute(reference_point)
    
    # 2. 局部搜索改进
    current_set = points[selected]
    current_hv = pg.hypervolume(current_set).compute(reference_point)
    
    for _ in range(100):  # 固定迭代次数控制运行时间
        improved = False
        
        # 尝试替换每个选中的点
        for i in range(k):
            for cand in random.sample(remaining, min(5, len(remaining))):
                # 创建新集合
                new_indices = selected.copy()
                new_indices[i] = cand
                new_set = points[new_indices]
                
                # 计算新超体积
                new_hv = pg.hypervolume(new_set).compute(reference_point)
                
                # 如果改进则接受
                if new_hv > current_hv:
                    selected = new_indices
                    remaining[remaining.index(cand)] = selected[i]
                    current_set = new_set
                    current_hv = new_hv
                    improved = True
                    break
            
            if improved:
                break
        
        # 如果没有改进，尝试交换两个选中的点
        if not improved and k >= 2:
            for i in range(k):
                for j in range(i+1, k):
                    # 交换两个选中的点
                    new_indices = selected.copy()
                    new_indices[i], new_indices[j] = new_indices[j], new_indices[i]
                    new_set = points[new_indices]
                    
                    new_hv = pg.hypervolume(new_set).compute(reference_point)
                    
                    if new_hv > current_hv:
                        selected = new_indices
                        current_set = new_set
                        current_hv = new_hv
                        improved = True
                        break
                
                if improved:
                    break
        
        # 如果没有进一步改进，退出
        if not improved:
            break
    
    return points[selected]


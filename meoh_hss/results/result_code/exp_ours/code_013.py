import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    n, m = points.shape
    
    if k <= 0 or k > n:
        raise ValueError("k must be greater than 0 and not larger than the number of points.")
    
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    indices = list(range(n))
    random.shuffle(indices)
    
    # 1. 贪心增量选择
    S = set()
    remaining = set(indices)
    
    # 初始选择：选择第一个点
    if indices:
        S.add(indices[0])
        remaining.remove(indices[0])
    
    # 贪心添加剩余点
    while len(S) < k and remaining:
        best_point = None
        best_hv = -float('inf')
        
        current_points = points[list(S)]
        if current_points.size == 0:
            current_hv = 0.0
        else:
            hv_calc = pg.hypervolume(current_points)
            current_hspecial_v = hv_calc.compute(reference_point)
        
        for r in list(remaining):
            temp_S = S | {r}
            temp_points = points[list(temp_S)]
            
            hv_calc = pg.hypervolume(temp_points)
            temp_hv = hv_calc.compute(reference_point)
            
            if temp_hv > best_hv:
                best_hv = temp_hv
                best_point = r
        
        if best_point is not None:
            S.add(best_point)
            remaining.remove(best_point)
    
    # 2. 替换改进
    improved = True
    while improved:
        improved = False
        current_points = points[list(S)]
        hv_calc = pg.hypervolume(current_points)
        current_hv = hv_calc.compute(reference_point)
        
        # 找到当前子集中贡献最小的点
        min_contrib_point = None
        min_contrib_value = float('inf')
        for s in S:
            temp_S = S - {s}
            temp_points = points[list(temp_S)]
            hv_calc = pg.hypervolume(temp_points)
            temp_hv = hv_calc.compute(reference_point)
            contribution = current_hv - temp_hv
            if contribution < min_contrib_value:
                min_contrib_value = contribution
                min_contrib_point = s
        
        # 尝试用剩余点替换贡献最小的点
        if min_contrib_point is not None:
            best_replace = None
            best_new_hv = current_hv
            for r in remaining:
                new_S = (S - {min_contrib_point}) | {r}
                new_points = points[list(new_S)]
                hv_calc = pg.hypervolume(new_points)
                new_hv = hv_calc.compute(reference_point)
                if new_hv > best_new_hv:
                    best_new_hv = new_hv
                    best_replace = r
            
            if best_replace is not None:
                S.remove(min_contrib_point)
                S.add(best_replace)
                remaining.remove(best_replace)
                remaining.add(min_contrib_point)
                improved = True
    
    # 返回结果
    selected_indices = list(S)
    return points[selected_indices]


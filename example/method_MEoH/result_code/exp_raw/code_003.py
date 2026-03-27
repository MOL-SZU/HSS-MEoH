import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    N, D = points.shape
    if k >= N:
        return points.copy()
    
    selected = []
    remaining = set(range(N))
    alpha = 0.7
    
    while len(selected) < k:
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            contrib = np.prod(reference_point - points[idx])
            dist_to_ref = np.linalg.norm(reference_point - points[idx])
            if dist_to_ref == 0:
                score = contrib
            else:
                score = alpha * contrib + (1 - alpha) * (1 / dist_to_ref)
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    from pymoo.indicators.hv import HV
    hv_calc = HV(ref_point=reference_point)
    
    improved = True
    while improved:
        improved = False
        current_set = points[selected]
        current_hv = hv_calc(current_set)
        
        contribs = []
        for i in range(k):
            subset = np.delete(current_set, i, axis=0)
            contrib = current_hv - hv_calc(subset)
            contribs.append((contrib, i))
        contribs.sort()
        
        worst_contrib, worst_idx = contribs[0]
        worst_original_idx = selected[worst_idx]
        
        best_swap_idx = -1
        best_hv = current_hv
        
        for idx in remaining:
            new_selected = selected.copy()
            new_selected[worst_idx] = idx
            new_set = points[new_selected]
            new_hv = hv_calc(new_set)
            if new_hv > best_hv:
                best_hv = new_hv
                best_swap_idx = idx
        
        if best_swap_idx != -1:
            remaining.remove(best_swap_idx)
            remaining.add(worst_original_idx)
            selected[worst_idx] = best_swap_idx
            improved = True
    
    return points[selected]


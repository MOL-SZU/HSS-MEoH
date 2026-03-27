import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    n, d = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    def compute_hv(indices):
        if not indices:
            return 0.0
        hv = pg.hypervolume(points[list(indices)])
        return hv.compute(reference_point)
    
    selected = set()
    remaining = set(range(n))
    
    while len(selected) < k:
        best_gain = -float('inf')
        best_idx = -1
        current_hv = compute_hv(selected)
        
        for idx in remaining:
            candidate = selected | {idx}
            candidate_hv = compute_hv(candidate)
            gain = candidate_hv - current_hv
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx != -1:
            selected.add(best_idx)
            remaining.remove(best_idx)
    
    while len(selected) > k:
        worst_loss = float('inf')
        worst_idx = -1
        current_hv = compute_hv(selected)
        
        for idx in selected:
            candidate = selected - {idx}
            candidate_hv = compute_hv(candidate)
            loss = current_hv - candidate_hv
            if loss < worst_loss:
                worst_loss = loss
                worst_idx = idx
        
        if worst_idx != -1:
            selected.remove(worst_idx)
            remaining.add(worst_idx)
    
    return points[list(selected)]


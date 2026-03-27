import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    N, D = points.shape
    if k >= N:
        return points.copy()
    
    selected = []
    remaining = list(range(N))
    
    while len(selected) < k and remaining:
        best_idx = -1
        
        if selected:
            selected_pts = points[selected]
            min_distances = np.min(np.linalg.norm(points[remaining][:, np.newaxis, :] - selected_pts, axis=2), axis=1)
            hv_contribs = np.prod(reference_point - points[remaining], axis=1)
            
            max_hv = np.max(hv_contribs)
            min_hv = np.min(hv_contribs)
            if max_hv > min_hv:
                hv_norm = (hv_contribs - min_hv) / (max_hv - min_hv)
            else:
                hv_norm = np.ones(len(hv_contribs))
            
            max_dist = np.max(min_distances)
            min_dist = np.min(min_distances)
            if max_dist > min_dist:
                dist_norm = (min_distances - min_dist) / (max_dist - min_dist)
            else:
                dist_norm = np.ones(len(min_distances))
            
            scores = 0.5 * hv_norm + 0.5 * dist_norm
            best_remaining_idx = np.argmax(scores)
            best_idx = remaining[best_remaining_idx]
        else:
            hv_contribs = np.prod(reference_point - points, axis=1)
            best_idx = np.argmax(hv_contribs)
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return points[selected]


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import pygmo as pg
    import random
    
    n, d = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    # Hypervolume cache using tuple keys
    hv_cache = {}
    
    def compute_hv(indices):
        """Compute hypervolume for a set of indices with caching."""
        key = tuple(sorted(indices))
        if key in hv_cache:
            return hv_cache[key]
        if not indices:
            hv_cache[key] = 0.0
        else:
            hv = pg.hypervolume(points[list(indices)])
            hv_cache[key] = hv.compute(reference_point)
        return hv_cache[key]
    
    # 1. Stochastic greedy initialization
    selected = set()
    remaining = list(range(n))
    random.shuffle(remaining)
    
    for _ in range(k):
        if not remaining:
            break
            
        best_idx = None
        best_contrib = -np.inf
        current_hv = compute_hv(list(selected))
        
        # Evaluate limited random candidates
        candidates = remaining[:min(20, len(remaining))]
        for idx in candidates:
            new_set = selected | {idx}
            new_hv = compute_hv(list(new_set))
            contrib = new_hv - current_hv
            if contrib > best_contrib:
                best_contrib = contrib
                best_idx = idx
        
        if best_idx is None:
            best_idx = remaining[0]
        
        selected.add(best_idx)
        remaining.remove(best_idx)
    
    # 2. Adaptive removal-addition cycles
    best_hv = compute_hv(list(selected))
    
    for cycle in range(30):
        # Compute all contributions
        contributions = {}
        for idx in selected:
            temp_set = selected - {idx}
            contributions[idx] = best_hv - compute_hv(list(temp_set))
        
        # Adaptive removal: remove worst points with probability
        to_remove = []
        for idx in selected:
            if contributions[idx] < np.mean(list(contributions.values())) * 0.5:
                if random.random() < 0.7:
                    to_remove.append(idx)
        
        # Remove at most 2 points per cycle
        if to_remove:
            remove_count = min(2, len(to_remove))
            removed = random.sample(to_remove, remove_count)
            selected.difference_update(removed)
            remaining.extend(removed)
        
        # Greedy addition to fill back to k
        while len(selected) < k and remaining:
            best_idx = None
            best_contrib = -np.inf
            current_hv = compute_hv(list(selected))
            
            candidates = remaining[:min(30, len(remaining))]
            for idx in candidates:
                new_set = selected | {idx}
                new_hv = compute_hv(list(new_set))
                contrib = new_hv - current_hv
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            
            if best_idx is not None:
                selected.add(best_idx)
                remaining.remove(best_idx)
            else:
                selected.add(remaining[0])
                remaining.pop(0)
        
        # Remove excess points if any
        while len(selected) > k:
            # Remove point with smallest contribution
            contributions = {}
            for idx in selected:
                temp_set = selected - {idx}
                contributions[idx] = best_hv - compute_hv(list(temp_set))
            worst = min(contributions, key=contributions.get)
            selected.remove(worst)
            remaining.append(worst)
        
        # Update best hypervolume
        new_hv = compute_hv(list(selected))
        if new_hv > best_hv:
            best_hv = new_hv
    
    # 3. Stochastic local swap optimization
    improved = True
    iteration = 0
    
    while improved and iteration < 50:
        improved = False
        iteration += 1
        
        # Random order of selected points
        selected_list = list(selected)
        random.shuffle(selected_list)
        
        for idx in selected_list[:min(5, len(selected_list))]:  # Limit attempts
            if not remaining:
                break
                
            # Try swaps with random candidates
            candidates = random.sample(remaining, min(10, len(remaining)))
            for cand in candidates:
                new_set = (selected - {idx}) | {cand}
                new_hv = compute_hv(list(new_set))
                
                if new_hv > best_hv + 1e-10:
                    selected = new_set
                    remaining.remove(cand)
                    remaining.append(idx)
                    best_hv = new_hv
                    improved = True
                    break
            
            if improved:
                break
    
    return points[list(selected)]


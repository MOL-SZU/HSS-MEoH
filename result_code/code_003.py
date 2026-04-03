import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import pygmo as pg
    import random
    
    n, d = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    # Efficient hypervolume cache
    hv_cache = {}
    
    def compute_hv(indices):
        key = frozenset(indices)
        if key in hv_cache:
            return hv_cache[key]
        if not indices:
            hv_cache[key] = 0.0
        else:
            hv = pg.hypervolume(points[list(indices)])
            hv_cache[key] = hv.compute(reference_point)
        return hv_cache[key]
    
    # 1. Fast greedy initialization with adaptive sampling
    selected = set()
    remaining = list(range(n))
    random.shuffle(remaining)
    
    # Start with random point
    selected.add(remaining.pop())
    
    while len(selected) < k:
        best_idx = None
        best_contrib = -np.inf
        current_hv = compute_hv(selected)
        
        # Adaptive candidate sampling
        sample_size = min(50, len(remaining))
        candidates = remaining[:sample_size]
        
        for idx in candidates:
            new_set = selected | {idx}
            new_hv = compute_hv(new_set)
            contrib = new_hv - current_hv
            if contrib > best_contrib:
                best_contrib = contrib
                best_idx = idx
        
        if best_idx is None:
            best_idx = remaining[0]
        
        selected.add(best_idx)
        remaining.remove(best_idx)
    
    # 2. Compact evolutionary optimization
    population = [set(selected)]
    # Add diverse solutions
    for _ in range(3):
        pop = set(random.sample(range(n), k))
        population.append(pop)
    
    fitness = [compute_hv(ind) for ind in population]
    best_hv = max(fitness)
    best_set = population[np.argmax(fitness)]
    
    # Adaptive stopping criteria
    no_improve = 0
    max_no_improve = 10
    
    for _ in range(30):
        # Tournament selection
        candidates = random.sample(range(len(population)), 2)
        parent_idx = max(candidates, key=lambda i: fitness[i])
        child = set(population[parent_idx])
        
        # Contribution-aware mutation
        if len(child) != k:
            # Size correction
            if len(child) < k:
                # Add with probability based on available points
                if random.random() < 0.8 and (n - len(child)) > 0:
                    child.add(random.choice(list(set(range(n)) - child)))
            else:
                # Remove with probability based on excess
                if random.random() < 0.8 and len(child) > 1:
                    child.remove(random.choice(list(child)))
        else:
            # Balanced mutation: swap based on contribution ranking
            if random.random() < 0.7:
                # Compute contributions quickly
                child_hv = compute_hv(child)
                contributions = {}
                for idx in child:
                    temp = child - {idx}
                    contributions[idx] = child_hv - compute_hv(temp)
                
                # Remove low contributor
                worst = min(contributions, key=contributions.get)
                child.remove(worst)
                
                # Add from remaining
                remaining_set = set(range(n)) - child
                if remaining_set:
                    # Sample candidates based on potential
                    candidates = list(remaining_set)[:min(20, len(remaining_set))]
                    best_cand = None
                    best_gain = -np.inf
                    for cand in candidates:
                        new_set = child | {cand}
                        gain = compute_hv(new_set) - compute_hv(child)
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = cand
                    if best_cand is not None:
                        child.add(best_cand)
        
        # Ensure correct size
        while len(child) > k:
            child.remove(random.choice(list(child)))
        while len(child) < k:
            child.add(random.choice(list(set(range(n)) - child)))
        
        # Environmental selection
        child_hv = compute_hv(child)
        if child_hv > min(fitness):
            min_idx = np.argmin(fitness)
            population[min_idx] = child
            fitness[min_idx] = child_hv
            
            if child_hv > best_hv:
                best_hv = child_hv
                best_set = child
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        if no_improve >= max_no_improve:
            break
    
    # 3. Focused local refinement on best solution only
    improved = True
    iterations = 0
    
    while improved and iterations < 20:
        improved = False
        iterations += 1
        
        # Compute contributions once per iteration
        contributions = {}
        for idx in best_set:
            temp = best_set - {idx}
            contributions[idx] = best_hv - compute_hv(temp)
        
        # Sort by contribution (ascending)
        sorted_points = sorted(best_set, key=lambda x: contributions[x])
        
        # Try swaps for worst points only
        for idx in sorted_points[:min(3, len(sorted_points))]:
            if improved:
                break
                
            remaining_set = set(range(n)) - best_set
            if not remaining_set:
                break
            
            # Sample promising candidates
            candidates = list(remaining_set)[:min(30, len(remaining_set))]
            for cand in candidates:
                new_set = (best_set - {idx}) | {cand}
                new_hv = compute_hv(new_set)
                if new_hv > best_hv + 1e-10:
                    best_set = new_set
                    best_hv = new_hv
                    improved = True
                    break
    
    return points[list(best_set)]


import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import pygmo as pg
    import random
    
    n, d = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    # Hypervolume cache using frozenset keys
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
    
    # 1. Fast greedy initialization
    selected = set()
    remaining = list(range(n))
    random.shuffle(remaining)
    
    # Start with random point
    selected.add(remaining.pop())
    
    while len(selected) < k:
        best_idx = None
        best_contrib = -np.inf
        current_hv = compute_hv(selected)
        
        # Sample limited candidates for speed
        sample_size = min(30, len(remaining))
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
    
    # 2. Initialize diverse population
    population = [selected]
    # Add diverse solutions: one random, one slightly perturbed greedy
    for i in range(2):
        if i == 0:
            pop = set(random.sample(range(n), k))
        else:
            pop = set(selected)
            # Perturb by swapping one point
            if pop and len(pop) > 1:
                to_remove = random.choice(list(pop))
                pop.remove(to_remove)
                available = set(range(n)) - pop
                if available:
                    pop.add(random.choice(list(available)))
        population.append(pop)
    
    fitness = [compute_hv(ind) for ind in population]
    best_hv = max(fitness)
    best_set = population[np.argmax(fitness)]
    
    # 3. Steady-state evolution with tournament selection
    no_improve = 0
    max_no_improve = 5  # Reduced for faster termination
    
    for _ in range(20):  # Fewer generations
        # Tournament selection
        idx1, idx2 = random.sample(range(len(population)), 2)
        parent = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
        
        child = set(parent)
        
        # Size-aware mutation
        if len(child) != k:
            if len(child) < k:
                if random.random() < 0.7 and (n - len(child)) > 0:
                    child.add(random.choice(list(set(range(n)) - child)))
            else:
                if random.random() < 0.7 and len(child) > 1:
                    child.remove(random.choice(list(child)))
        else:
            # Balanced swap mutation
            if random.random() < 0.6:
                # Remove worst contributor
                child_hv = compute_hv(child)
                contributions = {}
                for idx in child:
                    temp = child - {idx}
                    contributions[idx] = child_hv - compute_hv(temp)
                worst = min(contributions, key=contributions.get)
                child.remove(worst)
                
                # Add promising candidate
                remaining_set = set(range(n)) - child
                if remaining_set:
                    candidates = list(remaining_set)[:min(15, len(remaining_set))]
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
        
        # Replace worst in population if better
        child_hv = compute_hv(child)
        worst_idx = np.argmin(fitness)
        if child_hv > fitness[worst_idx]:
            population[worst_idx] = child
            fitness[worst_idx] = child_hv
            
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
    
    # 4. Focused final refinement (swap worst points first)
    improved = True
    iteration = 0
    
    while improved and iteration < 10:  # Fewer refinement iterations
        improved = False
        iteration += 1
        
        # Compute contributions of current best set
        contributions = {}
        for idx in best_set:
            temp = best_set - {idx}
            contributions[idx] = best_hv - compute_hv(temp)
        
        # Sort points by contribution (ascending)
        sorted_points = sorted(best_set, key=lambda x: contributions[x])
        
        # Try swapping only the worst 2 points
        for worst_idx in sorted_points[:2]:
            remaining_set = set(range(n)) - best_set
            if not remaining_set:
                break
            
            # Sample limited candidates
            candidates = list(remaining_set)[:min(20, len(remaining_set))]
            for cand in candidates:
                new_set = (best_set - {worst_idx}) | {cand}
                new_hv = compute_hv(new_set)
                if new_hv > best_hv + 1e-10:
                    best_set = new_set
                    best_hv = new_hv
                    improved = True
                    break
            
            if improved:
                break
    
    return points[list(best_set)]


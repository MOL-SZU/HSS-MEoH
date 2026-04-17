import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    n, d = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    # Hypervolume cache with frozenset keys
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
    
    # 1. Fast greedy initialization with limited evaluations
    selected = set()
    remaining = list(range(n))
    random.shuffle(remaining)
    
    # Start with random point
    selected.add(remaining.pop())
    
    while len(selected) < k:
        best_idx = None
        best_contrib = -np.inf
        current_hv = compute_hv(selected)
        
        # Evaluate only a subset of remaining points
        eval_count = min(50, len(remaining))
        candidates = remaining[:eval_count]
        
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
    population = [set(selected) for _ in range(6)]
    for i in range(1, 6):
        if i % 2 == 0:
            population[i] = set(random.sample(range(n), k))
        else:
            # Perturb greedy solution
            pop = set(selected)
            changes = random.randint(1, 2)
            for _ in range(changes):
                if pop and random.random() < 0.5:
                    pop.remove(random.choice(list(pop)))
                if len(pop) < k:
                    pop.add(random.choice(list(set(range(n)) - pop)))
            while len(pop) < k:
                pop.add(random.choice(list(set(range(n)) - pop)))
            while len(pop) > k:
                pop.remove(random.choice(list(pop)))
            population[i] = pop
    
    # Evaluate initial population
    fitness = [compute_hv(ind) for ind in population]
    
    # 3. Efficient evolutionary phase
    for _ in range(25):  # Reduced generations
        # Tournament selection
        parent_idx = random.sample(range(len(population)), 2)
        parent_idx = max(parent_idx, key=lambda i: fitness[i])
        child = set(population[parent_idx])
        
        # Size-aware mutation
        current_size = len(child)
        if current_size < k:
            add_prob, remove_prob = 0.8, 0.2
        elif current_size > k:
            add_prob, remove_prob = 0.2, 0.8
        else:
            add_prob, remove_prob = 0.5, 0.5
        
        if random.random() < add_prob and (n - current_size) > 0:
            child.add(random.choice(list(set(range(n)) - child)))
        elif random.random() < remove_prob and current_size > 1:
            child.remove(random.choice(list(child)))
        
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
    
    # 4. Elite local refinement
    best_idx = np.argmax(fitness)
    best_set = population[best_idx]
    best_hv = fitness[best_idx]
    
    improved = True
    while improved:
        improved = False
        
        # Compute contributions efficiently
        contributions = {}
        for idx in best_set:
            temp_set = best_set - {idx}
            contributions[idx] = best_hv - compute_hv(temp_set)
        
        # Sort by contribution
        sorted_points = sorted(contributions.items(), key=lambda x: x[1])
        
        # Try swapping worst points first
        for worst_idx, _ in sorted_points[:3]:  # Check only worst 3
            candidates = list(set(range(n)) - best_set)
            random.shuffle(candidates)
            
            for cand in candidates[:min(30, len(candidates))]:  # Limited checks
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


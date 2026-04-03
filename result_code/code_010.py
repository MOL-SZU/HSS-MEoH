import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    n, d = points.shape
    if k <= 0 or k > n:
        raise ValueError("k must be greater than 0 and not larger than the number of points.")
    
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    # Hypervolume cache with incremental updates
    hv_cache = {}
    
    def compute_hv(indices):
        key = frozenset(indices)
        if key in hv_cache:
            return hv_cache[key]
        if not indices:
            hv = 0.0
        else:
            hv = pg.hypervolume(points[list(indices)]).compute(reference_point)
        hv_cache[key] = hv
        return hv
    
    # 1. Two-phase greedy initialization
    def two_phase_greedy():
        # Phase 1: Forward selection
        selected = set()
        remaining = set(range(n))
        
        # Start with point maximizing hypervolume
        best_hv = -float('inf')
        best_idx = -1
        for idx in remaining:
            hv = compute_hv([idx])
            if hv > best_hv:
                best_hv = hv
                best_idx = idx
        
        selected.add(best_idx)
        remaining.remove(best_idx)
        
        # Greedy forward selection
        while len(selected) < k:
            best_contrib = -float('inf')
            best_idx = -1
            
            current_hv = compute_hv(selected)
            for idx in remaining:
                new_set = selected | {idx}
                new_hv = compute_hv(new_set)
                contrib = new_hv - current_hv
                
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            
            if best_idx != -1:
                selected.add(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        # Phase 2: Backward elimination if needed
        while len(selected) > k:
            worst_contrib = float('inf')
            worst_idx = -1
            
            current_hv = compute_hv(selected)
            for idx in selected:
                temp_set = selected - {idx}
                new_hv = compute_hv(temp_set)
                contrib = current_hv - new_hv
                
                if contrib < worst_contrib:
                    worst_contrib = contrib
                    worst_idx = idx
            
            if worst_idx != -1:
                selected.remove(worst_idx)
                remaining.add(worst_idx)
            else:
                break
        
        return selected
    
    # 2. Initialize population with diverse strategies
    population = []
    population_size = 12
    
    # Strategy 1: Two-phase greedy
    population.append(two_phase_greedy())
    
    # Strategy 2: Random subsets
    for _ in range(3):
        population.append(set(random.sample(range(n), k)))
    
    # Strategy 3: Greedy with random restarts
    for _ in range(4):
        base = set()
        # Start with random point
        base.add(random.randint(0, n-1))
        
        while len(base) < k:
            current_hv = compute_hv(base)
            candidates = random.sample(list(set(range(n)) - base), min(5, n - len(base)))
            best_contrib = -float('inf')
            best_idx = -1
            
            for idx in candidates:
                new_set = base | {idx}
                new_hv = compute_hv(new_set)
                contrib = new_hv - current_hv
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            
            if best_idx != -1:
                base.add(best_idx)
            else:
                available = set(range(n)) - base
                if available:
                    base.add(random.choice(list(available)))
        
        population.append(base)
    
    # Fill remaining with random
    while len(population) < population_size:
        population.append(set(random.sample(range(n), k)))
    
    # 3. Directed mutation operator
    def directed_mutation(indices):
        current = set(indices)
        operation = random.random()
        
        if operation < 0.7:  # Focused mutation
            # Remove worst, add best
            if len(current) > 1:
                # Find worst point
                current_hv = compute_hv(current)
                worst_contrib = float('inf')
                worst_idx = -1
                
                for idx in current:
                    temp_set = current - {idx}
                    new_hv = compute_hv(temp_set)
                    contrib = current_hv - new_hv
                    if contrib < worst_contrib:
                        worst_contrib = contrib
                        worst_idx = idx
                
                if worst_idx != -1:
                    current.remove(worst_idx)
            
            # Add best available point
            if len(current) < n:
                available = set(range(n)) - current
                if available:
                    current_hv = compute_hv(current)
                    best_contrib = -float('inf')
                    best_idx = -1
                    
                    # Sample candidates for efficiency
                    candidates = random.sample(list(available), min(10, len(available)))
                    for idx in candidates:
                        new_set = current | {idx}
                        new_hv = compute_hv(new_set)
                        contrib = new_hv - current_hv
                        if contrib > best_contrib:
                            best_contrib = contrib
                            best_idx = idx
                    
                    if best_idx != -1:
                        current.add(best_idx)
        
        else:  # Exploratory mutation
            # Random swap
            if current and len(current) < n:
                remove_idx = random.choice(list(current))
                current.remove(remove_idx)
                
                available = set(range(n)) - current
                if available:
                    add_idx = random.choice(list(available))
                    current.add(add_idx)
        
        # Ensure correct size
        while len(current) != k:
            if len(current) < k:
                available = set(range(n)) - current
                if available:
                    current.add(random.choice(list(available)))
            else:
                current.remove(random.choice(list(current)))
        
        return current
    
    # 4. Steady-state evolutionary search
    best_solution = None
    best_hv = -float('inf')
    
    # Evaluate initial population
    fitness = [compute_hv(ind) for ind in population]
    best_idx = np.argmax(fitness)
    if fitness[best_idx] > best_hv:
        best_hv = fitness[best_idx]
        best_solution = population[best_idx]
    
    generations = 80
    for gen in range(generations):
        # Select parent using binary tournament
        idx1, idx2 = random.sample(range(len(population)), 2)
        parent = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
        
        # Create offspring
        offspring = directed_mutation(parent)
        offspring_hv = compute_hv(offspring)
        
        # Replace worst in population if offspring is better
        worst_idx = np.argmin(fitness)
        if offspring_hv > fitness[worst_idx]:
            # Check for duplicates
            is_duplicate = False
            for idx, ind in enumerate(population):
                if ind == offspring:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                population[worst_idx] = offspring
                fitness[worst_idx] = offspring_hv
                
                # Update best solution
                if offspring_hv > best_hv:
                    best_hv = offspring_hv
                    best_solution = offspring
    
    # 5. Iterative swap refinement
    if best_solution is None:
        best_solution = population[0]
        best_hv = compute_hv(best_solution)
    
    improvement = True
    while improvement:
        improvement = False
        
        # Sort points by contribution
        contributions = {}
        for idx in best_solution:
            temp_set = best_solution - {idx}
            hv_without = compute_hv(temp_set)
            contributions[idx] = best_hv - hv_without
        
        # Try swapping low contributors
        sorted_points = sorted(contributions.items(), key=lambda x: x[1])
        remaining = set(range(n)) - best_solution
        
        for idx_in, _ in sorted_points[:min(4, k)]:
            best_swap = None
            best_gain = 0
            
            # Evaluate potential replacements
            for idx_out in remaining:
                new_set = (best_solution - {idx_in}) | {idx_out}
                new_hv = compute_hv(new_set)
                gain = new_hv - best_hv
                
                if gain > best_gain:
                    best_gain = gain
                    best_swap = idx_out
            
            if best_swap is not None and best_gain > 1e-10:
                best_solution.remove(idx_in)
                best_solution.add(best_swap)
                best_hv += best_gain
                improvement = True
                break
    
    # Return selected points
    selected_indices = list(best_solution)
    return points[selected_indices]


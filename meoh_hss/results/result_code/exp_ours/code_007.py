import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    import numpy as np
    import random
    import pygmo as pg
    
    n, d = points.shape
    if k <= 0 or k > n:
        raise ValueError("k must be greater than 0 and not larger than the number of points.")
    
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
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
    
    # Phase 1: Greedy initialization with incremental contributions
    def greedy_initialization():
        selected = set()
        # Find the point with maximal hypervolume as starting point
        best_hv = -float('inf')
        best_idx = -1
        for idx in range(n):
            hv = compute_hv([idx])
            if hv > best_hv:
                best_hv = hv
                best_idx = idx
        selected.add(best_idx)
        
        # Incremental greedy selection with candidate sampling
        while len(selected) < k:
            current_hv = compute_hv(selected)
            best_contrib = -float('inf')
            best_idx = -1
            # Adaptive sampling: larger sample when few points remain
            remaining = [i for i in range(n) if i not in selected]
            sample_size = min(25, len(remaining))
            candidates = random.sample(remaining, sample_size)
            for idx in candidates:
                new_set = selected | {idx}
                new_hv = compute_hv(new_set)
                contrib = new_hv - current_hv
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            if best_idx != -1:
                selected.add(best_idx)
            else:
                # Add random point if no improvement found
                if remaining:
                    selected.add(random.choice(remaining))
        return selected
    
    # Initialize diverse population
    population_size = 15
    population = []
    # Include greedy solution
    population.append(greedy_initialization())
    # Include random solutions
    for _ in range(population_size - 1):
        population.append(set(random.sample(range(n), k)))
    
    # Phase 2: Evolutionary optimization with dynamic mutation probabilities
    def dynamic_mutation(indices):
        current = set(indices)
        # Dynamic mutation probability based on subset fitness
        fitness = compute_hv(current)
        if fitness > 0.8 * best_hv:  # Higher fitness -> more focused mutation
            mutation_type_prob = 0.8
        else:
            mutation_type_prob = 0.5
        
        if random.random() < mutation_type_prob:  # Focused mutation: remove worst, add best
            if len(current) > 1:
                # Identify worst contributor in current set
                contributions = {}
                for idx in current:
                    temp_set = current - {idx}
                    hv_without = compute_hv(temp_set)
                    contributions[idx] = fitness - hv_without
                worst_idx = min(contributions, key=contributions.get)
                current.remove(worst_idx)
            
            if len(current) < n:
                available = set(range(n)) - current
                if available:
                    # Find best addition from sampled candidates
                    current_hv = compute_hv(current)
                    best_contrib = -float('inf')
                    best_idx = -1
                    sample_size = min(20, len(available))
                    candidates = random.sample(list(available), sample_size)
                    for idx in candidates:
                        new_set = current | {idx}
                        new_hv = compute_hv(new_set)
                        contrib = new_hv - current_hv
                        if contrib > best_contrib:
                            best_contrib = contrib
                            best_idx = idx
                    if best_idx != -1:
                        current.add(best_idx)
        else:  # Random swap mutation
            if current and len(current) < n:
                remove_idx = random.choice(list(current))
                current.remove(remove_idx)
                available = set(range(n)) - current
                if available:
                    add_idx = random.choice(list(available))
                    current.add(add_idx)
        
        # Adjust size to exactly k
        while len(current) != k:
            if len(current) < k:
                available = set(range(n)) - current
                if available:
                    current.add(random.choice(list(available)))
            else:
                current.remove(random.choice(list(current)))
        return current
    
    # Evaluate initial population
    fitness = [compute_hv(ind) for ind in population]
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx]
    best_hv = fitness[best_idx]
    
    generations = 100
    for gen in range(generations):
        # Tournament selection
        idx1, idx2 = random.sample(range(len(population)), 2)
        parent = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
        # Apply dynamic mutation
        offspring = dynamic_mutation(parent)
        offspring_hv = compute_hv(offspring)
        
        # Replace worst solution if offspring is better and not duplicate
        worst_idx = np.argmin(fitness)
        if offspring_hv > fitness[worst_idx]:
            duplicate = False
            for ind in population:
                if ind == offspring:
                    duplicate = True
                    break
            if not duplicate:
                population[worst_idx] = offspring
                fitness[worst_idx] = offspring_hv
                if offspring_hv > best_hv:
                    best_hv = offspring_hv
                    best_solution = offspring
    
    # Phase 3: Exhaustive swap refinement prioritized by contribution ranking
    improvement = True
    while improvement:
        improvement = False
        # Compute contributions for all points in current best solution
        contributions = {}
        for idx in best_solution:
            temp_set = best_solution - {idx}
            hv_without = compute_hv(temp_set)
            contributions[idx] = best_hv - hv_without
        
        # Sort points by contribution (lowest first)
        sorted_points = sorted(contributions.items(), key=lambda x: x[1])
        remaining = set(range(n)) - best_solution
        
        # Evaluate all swaps for each low-contribution point
        for idx_in, _ in sorted_points:
            best_swap = None
            best_gain = 0
            # Exhaustive evaluation of all possible replacements
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
                break  # Restart with updated contributions
    
    selected_indices = list(best_solution)
    return points[selected_indices]


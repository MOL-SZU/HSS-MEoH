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
    
    def greedy_initialization():
        selected = set()
        remaining = set(range(n))
        
        best_hv = -float('inf')
        best_idx = -1
        for idx in remaining:
            hv = compute_hv([idx])
            if hv > best_hv:
                best_hv = hv
                best_idx = idx
        
        selected.add(best_idx)
        remaining.remove(best_idx)
        
        while len(selected) < k:
            best_contrib = -float('inf')
            best_idx = -1
            current_hv = compute_hv(selected)
            
            eval_points = list(remaining)
            if len(eval_points) > 100:
                eval_points = random.sample(eval_points, 100)
            
            for idx in eval_points:
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
                if remaining:
                    selected.add(random.choice(list(remaining)))
                    remaining.remove(list(selected)[-1])
        
        return selected
    
    population_size = 8
    population = []
    population.append(greedy_initialization())
    
    for _ in range(population_size - 1):
        base = set(random.sample(list(population[0]), min(k, len(population[0]))))
        while len(base) < k:
            available = set(range(n)) - base
            if available:
                base.add(random.choice(list(available)))
        while len(base) > k:
            base.remove(random.choice(list(base)))
        population.append(base)
    
    def adaptive_mutation(indices):
        current_size = len(indices)
        new_indices = set(indices)
        
        if current_size < k:
            add_prob = 0.8
            remove_prob = 0.2
        elif current_size > k:
            add_prob = 0.2
            remove_prob = 0.8
        else:
            add_prob = 0.5
            remove_prob = 0.5
        
        if random.random() < add_prob and len(new_indices) < n:
            available = set(range(n)) - new_indices
            if available:
                current_hv = compute_hv(new_indices)
                best_contrib = -float('inf')
                best_idx = -1
                eval_avail = list(available)
                if len(eval_avail) > 50:
                    eval_avail = random.sample(eval_avail, 50)
                for idx in eval_avail:
                    temp_set = new_indices | {idx}
                    new_hv = compute_hv(temp_set)
                    contrib = new_hv - current_hv
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_idx = idx
                if best_idx != -1:
                    new_indices.add(best_idx)
        
        if random.random() < remove_prob and len(new_indices) > 1:
            current_hv = compute_hv(new_indices)
            worst_contrib = float('inf')
            worst_idx = -1
            for idx in new_indices:
                temp_set = new_indices - {idx}
                new_hv = compute_hv(temp_set)
                contrib = current_hv - new_hv
                if contrib < worst_contrib:
                    worst_contrib = contrib
                    worst_idx = idx
            if worst_idx != -1:
                new_indices.remove(worst_idx)
        
        while len(new_indices) != k:
            if len(new_indices) < k:
                available = set(range(n)) - new_indices
                if available:
                    new_indices.add(random.choice(list(available)))
            else:
                new_indices.remove(random.choice(list(new_indices)))
        
        return new_indices
    
    max_iterations = 100
    no_improvement_limit = 20
    best_solution = None
    best_hv = -float('inf')
    no_improvement = 0
    
    for iteration in range(max_iterations):
        new_population = []
        for _ in range(population_size):
            candidates = random.sample(population, 3)
            candidate_hvs = [compute_hv(s) for s in candidates]
            parent = candidates[np.argmax(candidate_hvs)]
            offspring = adaptive_mutation(parent)
            new_population.append(offspring)
        
        combined = population + new_population
        combined_hvs = [compute_hv(s) for s in combined]
        sorted_indices = np.argsort(combined_hvs)[-population_size:]
        population = [combined[i] for i in sorted_indices]
        
        current_best_hv = max(combined_hvs)
        if current_best_hv > best_hv + 1e-10:
            best_hv = current_best_hv
            best_solution = combined[np.argmax(combined_hvs)]
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement >= no_improvement_limit:
            break
    
    if best_solution is None:
        best_solution = population[0]
        best_hv = compute_hv(best_solution)
    
    improved = True
    while improved:
        improved = False
        contributions = {}
        for idx in best_solution:
            temp_set = best_solution - {idx}
            hv_without = compute_hv(temp_set)
            contributions[idx] = best_hv - hv_without
        
        sorted_points = sorted(contributions.items(), key=lambda x: x[1])
        remaining = set(range(n)) - best_solution
        
        for idx_in, _ in sorted_points[:min(3, k)]:
            best_swap = None
            best_improvement = 0
            eval_remain = list(remaining)
            if len(eval_remain) > 50:
                eval_remain = random.sample(eval_remain, 50)
            
            for idx_out in eval_remain:
                new_set = (best_solution - {idx_in}) | {idx_out}
                new_hv = compute_hv(new_set)
                improvement = new_hv - best_hv
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = idx_out
            
            if best_swap is not None and best_improvement > 1e-10:
                best_solution.remove(idx_in)
                best_solution.add(best_swap)
                best_hv = best_hv + best_improvement
                improved = True
                break
    
    selected_indices = list(best_solution)
    return points[selected_indices]


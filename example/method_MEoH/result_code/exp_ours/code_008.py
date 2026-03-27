import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    n, d = points.shape
    
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    # Hypervolume computation with caching
    hv_cache = {}
    def compute_hv(indices_set):
        key = frozenset(indices_set)
        if key in hv_cache:
            return hv_cache[key]
        if not indices_set:
            hv = 0.0
        else:
            hv = pg.hypervolume(points[list(indices_set)]).compute(reference_point)
        hv_cache[key] = hv
        return hv
    
    # 1. Diverse greedy initialization with adaptive sampling
    def greedy_init():
        indices = set()
        current_hv = 0.0
        for _ in range(k):
            candidates = [i for i in range(n) if i not in indices]
            sample_size = min(25, len(candidates))
            sampled = random.sample(candidates, sample_size)
            best_contrib = -float('inf')
            best_idx = None
            for idx in sampled:
                temp_set = indices | {idx}
                new_hv = compute_hv(temp_set)
                contrib = new_hv - current_hv
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            indices.add(best_idx)
            current_hv = compute_hv(indices)
        return indices
    
    # Initialize diverse population
    population = []
    for _ in range(15):
        if random.random() < 0.7:
            indices = greedy_init()
        else:
            indices = set(random.sample(range(n), k))
        population.append(indices)
    
    # 2. Evolutionary phase with size-aware mutation
    def size_aware_mutation(indices):
        b = len(indices)
        a = n - b
        
        # Adaptive mutation probabilities
        if b < k:
            add_prob = (k - b + 1) / (2 * a) if a > 0 else 0
            remove_prob = 0
        elif b > k:
            add_prob = 0
            remove_prob = (b - k + 1) / (2 * b) if b > 0 else 0
        else:
            add_prob = 1 / (2 * a) if a > 0 else 0
            remove_prob = 1 / (2 * b) if b > 0 else 0
        
        new_indices = set(indices)
        
        # Add best candidate from adaptive sample
        if random.random() < add_prob and a > 0:
            unselected = [i for i in range(n) if i not in new_indices]
            sample_size = min(25, len(unselected))
            sampled = random.sample(unselected, sample_size)
            best_contrib = -float('inf')
            best_idx = None
            current_hv = compute_hv(new_indices)
            for idx in sampled:
                temp_set = new_indices | {idx}
                new_hv = compute_hv(temp_set)
                contrib = new_hv - current_hv
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            if best_idx is not None:
                new_indices.add(best_idx)
        
        # Remove worst contributor
        if random.random() < remove_prob and b > 0:
            worst_contrib = float('inf')
            worst_idx = None
            current_hv = compute_hv(new_indices)
            for idx in new_indices:
                temp_set = new_indices - {idx}
                new_hv = compute_hv(temp_set)
                contrib = current_hv - new_hv
                if contrib < worst_contrib:
                    worst_contrib = contrib
                    worst_idx = idx
            if worst_idx is not None:
                new_indices.remove(worst_idx)
        
        # Ensure exact size k
        while len(new_indices) != k:
            if len(new_indices) < k:
                unselected = [i for i in range(n) if i not in new_indices]
                if unselected:
                    new_indices.add(random.choice(unselected))
            else:
                new_indices.remove(random.choice(list(new_indices)))
        
        return new_indices
    
    # Steady-state evolutionary iterations
    iterations = 100
    for _ in range(iterations):
        new_population = []
        for indices in population:
            mutated = size_aware_mutation(indices)
            new_population.append(mutated)
        
        # Tournament selection
        combined = population + new_population
        hv_values = [compute_hv(indices) for indices in combined]
        
        tournament_size = 3
        selected = []
        for _ in range(15):
            candidates = random.sample(range(len(combined)), tournament_size)
            winner = candidates[np.argmax([hv_values[i] for i in candidates])]
            selected.append(combined[winner])
        population = selected
    
    # 3. Exhaustive swap refinement focusing on low-contribution points
    best_indices = population[0]
    best_hv = compute_hv(best_indices)
    
    # Sort current points by contribution (ascending)
    contributions = []
    for idx in best_indices:
        temp_set = best_indices - {idx}
        loss = best_hv - compute_hv(temp_set)
        contributions.append((idx, loss))
    contributions.sort(key=lambda x: x[1])
    ordered_indices = [c[0] for c in contributions]
    
    improved = True
    while improved:
        improved = False
        
        # Try swapping lowest-contribution points first
        for inner_idx in ordered_indices:
            unselected = [i for i in range(n) if i not in best_indices]
            for outer_idx in unselected:
                temp_set = set(best_indices)
                temp_set.remove(inner_idx)
                temp_set.add(outer_idx)
                new_hv = compute_hv(temp_set)
                
                if new_hv > best_hv + 1e-10:
                    best_indices = temp_set
                    best_hv = new_hv
                    improved = True
                    break
            if improved:
                # Recompute contributions for new set
                contributions = []
                for idx in best_indices:
                    temp_set = best_indices - {idx}
                    loss = best_hv - compute_hv(temp_set)
                    contributions.append((idx, loss))
                contributions.sort(key=lambda x: x[1])
                ordered_indices = [c[0] for c in contributions]
                break
    
    selected_points = points[list(best_indices)]
    return selected_points


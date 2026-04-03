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
    
    # 1. Random initialization
    population = []
    for _ in range(10):
        indices = set(random.sample(range(n), k))
        population.append(indices)
    
    # 2. Evolutionary phase with hypervolume-guided mutation
    def hv_guided_mutation(indices):
        current_hv = compute_hv(indices)
        b = len(indices)
        a = n - b
        
        # Adaptive mutation probabilities based on size difference
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
        
        # Addition: select point maximizing HV contribution
        if random.random() < add_prob and a > 0:
            unselected = [i for i in range(n) if i not in new_indices]
            best_contrib = -float('inf')
            best_idx = None
            for idx in unselected:
                temp_set = new_indices | {idx}
                new_hv = compute_hv(temp_set)
                contrib = new_hv - current_hv
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = idx
            if best_idx is not None:
                new_indices.add(best_idx)
        
        # Removal: select point minimizing HV contribution
        if random.random() < remove_prob and b > 0:
            worst_contrib = float('inf')
            worst_idx = None
            for idx in new_indices:
                temp_set = new_indices - {idx}
                new_hv = compute_hv(temp_set)
                contrib = current_hv - new_hv  # Loss when removed
                if contrib < worst_contrib:
                    worst_contrib = contrib
                    worst_idx = idx
            if worst_idx is not None:
                new_indices.remove(worst_idx)
        
        # Ensure size k by additional random adjustments if needed
        while len(new_indices) != k:
            if len(new_indices) < k:
                unselected = [i for i in range(n) if i not in new_indices]
                if unselected:
                    new_indices.add(random.choice(unselected))
            else:
                new_indices.remove(random.choice(list(new_indices)))
        
        return new_indices
    
    # Evolutionary iterations
    iterations = 100
    for _ in range(iterations):
        new_population = []
        for indices in population:
            mutated = hv_guided_mutation(indices)
            new_population.append(mutated)
        
        # Environmental selection: keep top 10 by HV
        combined = population + new_population
        hv_values = [compute_hv(indices) for indices in combined]
        top_indices = np.argsort(hv_values)[-10:]
        population = [combined[i] for i in top_indices]
    
    # 3. Exhaustive swap refinement
    best_indices = population[0]
    best_hv = compute_hv(best_indices)
    
    improved = True
    while improved:
        improved = False
        current_list = list(best_indices)
        
        # Try all possible swaps
        for i in range(k):
            for j in range(n):
                if j in best_indices:
                    continue
                
                temp_list = current_list.copy()
                temp_list[i] = j
                temp_set = set(temp_list)
                new_hv = compute_hv(temp_set)
                
                if new_hv > best_hv + 1e-10:
                    best_indices = temp_set
                    best_hv = new_hv
                    improved = True
                    break
            if improved:
                break
    
    selected_points = points[list(best_indices)]
    return selected_points


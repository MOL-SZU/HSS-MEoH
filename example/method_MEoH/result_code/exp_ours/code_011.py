import numpy as np

def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    HSS (Hypervolume-based Subset Selection): 贪心选择超体积贡献最大的 k 个点

    参数:
        points: np.ndarray
            所有候选点，形状为 (N, D)，其中 N 是点数，D 是目标维度
        k: int
            需要选择的点数
        reference_point: np.ndarray, optional
            参考点，形状为 (D,)。如果为 None，则使用 points 的最大值 * 1.1

    返回:
        subset: np.ndarray
            选出的子集，形状为 (k, D)
    """
    n, d = points.shape
    if k <= 0 or k > n:
        raise ValueError("k must be greater than 0 and not larger than the number of points.")
    
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1
    
    # Hypervolume cache
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
    
    # 1. Greedy initialization with random restarts
    def greedy_initialization():
        selected = set()
        # Start with point maximizing hypervolume
        best_hv = -float('inf')
        best_idx = -1
        for idx in range(n):
            hv = compute_hv([idx])
            if hv > best_hv:
                best_hv = hv
                best_idx = idx
        selected.add(best_idx)
        
        # Greedy forward selection
        while len(selected) < k:
            current_hv = compute_hv(selected)
            best_contrib = -float('inf')
            best_idx = -1
            
            # Sample candidates for efficiency
            candidates = random.sample(list(set(range(n)) - selected), min(20, n - len(selected)))
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
                # Fallback: add random point
                available = set(range(n)) - selected
                if available:
                    selected.add(random.choice(list(available)))
        
        return selected
    
    # 2. Initialize diverse population
    population_size = 15
    population = []
    
    # Add greedy solution
    population.append(greedy_initialization())
    
    # Add random solutions
    for _ in range(population_size - 1):
        population.append(set(random.sample(range(n), k)))
    
    # 3. Size-aware mutation operator
    def size_aware_mutation(indices):
        current = set(indices)
        op_type = random.random()
        
        if op_type < 0.6:  # Contribution-based mutation
            # Remove worst contributor
            if len(current) > 1:
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
                    candidates = random.sample(list(available), min(15, len(available)))
                    for idx in candidates:
                        new_set = current | {idx}
                        new_hv = compute_hv(new_set)
                        contrib = new_hv - current_hv
                        if contrib > best_contrib:
                            best_contrib = contrib
                            best_idx = idx
                    if best_idx != -1:
                        current.add(best_idx)
        
        elif op_type < 0.9:  # Random swap
            if current and len(current) < n:
                remove_idx = random.choice(list(current))
                current.remove(remove_idx)
                available = set(range(n)) - current
                if available:
                    add_idx = random.choice(list(available))
                    current.add(add_idx)
        
        else:  # Random perturbation
            if random.random() < 0.5 and len(current) > 1:
                current.remove(random.choice(list(current)))
            if len(current) < n:
                available = set(range(n)) - current
                if available:
                    current.add(random.choice(list(available)))
        
        # Ensure correct size
        while len(current) != k:
            if len(current) < k:
                available = set(range(n)) - current
                if available:
                    current.add(random.choice(list(available)))
            else:
                current.remove(random.choice(list(current)))
        
        return current
    
    # 4. Steady-state evolution
    fitness = [compute_hv(ind) for ind in population]
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx]
    best_hv = fitness[best_idx]
    
    generations = 100
    for _ in range(generations):
        # Tournament selection
        idx1, idx2 = random.sample(range(len(population)), 2)
        parent = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
        
        # Mutation
        offspring = size_aware_mutation(parent)
        offspring_hv = compute_hv(offspring)
        
        # Replace worst if offspring is better and not duplicate
        worst_idx = np.argmin(fitness)
        if offspring_hv > fitness[worst_idx]:
            is_duplicate = False
            for ind in population:
                if ind == offspring:
                    is_duplicate = True
                    break
            if not is_duplicate:
                population[worst_idx] = offspring
                fitness[worst_idx] = offspring_hv
                if offspring_hv > best_hv:
                    best_hv = offspring_hv
                    best_solution = offspring
    
    # 5. Exhaustive single-swap local refinement
    improvement = True
    while improvement:
        improvement = False
        remaining = set(range(n)) - best_solution
        
        # Try all possible single swaps
        for idx_in in list(best_solution):
            for idx_out in remaining:
                new_set = (best_solution - {idx_in}) | {idx_out}
                new_hv = compute_hv(new_set)
                if new_hv > best_hv + 1e-10:
                    best_solution = new_set
                    best_hv = new_hv
                    improvement = True
                    remaining = set(range(n)) - best_solution
                    break
            if improvement:
                break
    
    # Return selected points
    selected_indices = list(best_solution)
    return points[selected_indices]


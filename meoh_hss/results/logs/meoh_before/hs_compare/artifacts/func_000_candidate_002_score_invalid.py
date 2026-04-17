def HSS(points, k: int, reference_point=None, initial_temp=415.33953379791535, final_temp=0.17749630379371173, cooling_rate=0.9614296741961484, max_iterations=320.96112191581375) -> np.ndarray:
    import pygmo as pg
    import random
    n, m = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1

    def compute_hv(indices):
        if len(indices) == 0:
            return 0.0
        selected_points = points[list(indices)]
        hv = pg.hypervolume(selected_points)
        return hv.compute(reference_point)
    current_indices = set(random.sample(range(n), k))
    current_hv = compute_hv(current_indices)
    temperature = initial_temp
    best_indices = current_indices.copy()
    best_hv = current_hv
    for iteration in range(max_iterations):
        new_indices = current_indices.copy()
        remove_idx = random.choice(list(new_indices))
        new_indices.remove(remove_idx)
        available = [i for i in range(n) if i not in new_indices]
        if not available:
            continue
        add_idx = random.choice(available)
        new_indices.add(add_idx)
        new_hv = compute_hv(new_indices)
        delta = new_hv - current_hv
        if delta > 0:
            current_indices = new_indices
            current_hv = new_hv
            if new_hv > best_hv:
                best_indices = new_indices.copy()
                best_hv = new_hv
        elif random.random() < np.exp(delta / temperature):
            current_indices = new_indices
            current_hv = new_hv
        temperature *= cooling_rate
        if temperature < final_temp:
            break
    result_indices = list(best_indices)
    return points[result_indices]
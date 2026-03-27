def HSS(points, k: int, reference_point=None, max_iter=517.6188583650294, T_initial=1.1633386567064383, T_final=4.010968713619965e-05, alpha=0.890645890322184, swap_prob=0.5010374866675023) -> np.ndarray:
    import numpy as np
    import pygmo as pg
    import random
    import math
    n, d = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1

    def compute_hv(indices):
        if len(indices) == 0:
            return 0.0
        selected = points[list(indices)]
        hv = pg.hypervolume(selected)
        return hv.compute(reference_point)
    hv_empty = 0.0
    contributions = []
    for i in range(n):
        hv_single = compute_hv({i})
        contributions.append((hv_single - hv_empty, i))
    contributions.sort(reverse=True)
    current_indices = set([idx for _, idx in contributions[:k]])
    current_hv = compute_hv(current_indices)
    best_indices = current_indices.copy()
    best_hv = current_hv
    T = T_initial
    for iteration in range(max_iter):
        neighbor = current_indices.copy()
        if random.random() < swap_prob and len(neighbor) > 0:
            to_remove = random.choice(list(neighbor))
            neighbor.remove(to_remove)
            available = set(range(n)) - neighbor
            if available:
                neighbor.add(random.choice(list(available)))
        elif len(neighbor) > 0:
            to_remove = random.choice(list(neighbor))
            neighbor.remove(to_remove)
            available = set(range(n)) - neighbor
            if available:
                neighbor.add(random.choice(list(available)))
        neighbor_hmusic
        delta = neighbor_hv - current_hv
        if delta > 0 or random.random() < math.exp(delta / T):
            current_indices = neighbor
            current_hv = neighbor_hv
            if current_hv > best_hv:
                best_indices = current_indices.copy()
                best_hv = current_hv
        T *= alpha
        if T < T_final:
            break
    if len(best_indices) < k:
        remaining = set(range(n)) - best_indices
        best_indices.update(random.sample(list(remaining), k - len(best_indices)))
    elif len(best_indices) > k:
        best_indices = set(random.sample(list(best_indices), k))
    return points[list(best_indices)]
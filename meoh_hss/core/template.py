template_program = '''
import numpy as np
import math
import random
import scipy
from typing import List, Set, Any, Dict, Callable
import pygmo as pg
from mat2array import load_mat_to_numpy
import time
from scipy.spatial.distance import cdist
from scipy.special import comb

def HSS(points, k: int, reference_point) -> np.ndarray:
    """Greedily select k points with the largest hypervolume contribution."""
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1

    def hv_cal(objectives_list: List[np.ndarray]) -> float:
        if not objectives_list:
            return 0.0
        objectives_array = np.array(objectives_list)
        hv = pg.hypervolume(objectives_array)
        return hv.compute(reference_point)

    def hypervolume_contribution(point: np.ndarray, selected_subset: List[np.ndarray]) -> float:
        if len(selected_subset) == 0:
            return hv_cal([point])
        return hv_cal(selected_subset + [point]) - hv_cal(selected_subset)

    n, _ = points.shape
    selected_indices: List[int] = []
    selected_points: List[np.ndarray] = []
    while len(selected_indices) < k:
        max_contrib = -np.inf
        max_idx = None
        for idx in range(n):
            if idx in selected_indices:
                continue
            contrib = hypervolume_contribution(points[idx], selected_points)
            if contrib > max_contrib:
                max_contrib = contrib
                max_idx = idx
        if max_idx is None:
            remaining_indices = [i for i in range(n) if i not in selected_indices]
            if not remaining_indices:
                break
            max_idx = remaining_indices[0]
        selected_indices.append(max_idx)
        selected_points.append(points[max_idx])
    return np.array(selected_points)

'''

task_description = 'Implement a function that selects a subset from a set of non-dominated points to maximize their hypervolume.'

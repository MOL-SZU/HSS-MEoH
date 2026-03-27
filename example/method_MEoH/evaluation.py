# name: str: HSS_Evaluation
# Parameters:
# timeout_seconds: int: 60
# k: int: 20
# data_folder: str: ./data/hss_data
# data_key: str: points
# end
from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np

from llm4ad.base import Evaluation
from template import template_program, task_description
from mat2array import load_mat_to_numpy

import time
from typing import Tuple

__all__ = ['HSS_Evaluation']


def evaluate(points: np.ndarray, k: int, reference_point: np.ndarray, hss_func: callable) -> np.ndarray:
    """
    Evaluate HSS function on the given points.
    
    Args:
        points: Numpy array of shape (n, m) where each row is a point in the objective space
        k: Number of points to select
        reference_point: Numpy array of shape (m,) representing the reference point
        hss_func: The HSS function to evaluate (callable)
    
    Returns:
        Numpy array with [hypervolume, -running_time] as objectives
    """
    import pygmo as pg
    start_time = time.time()
    
    try:
        # Call the HSS function
        selected_points = hss_func(points, k, reference_point)
        
        # Ensure selected_points is the correct shape (k, m)
        if selected_points.shape[0] != k:
            # If the function returns wrong shape, return negative infinity
            return np.array([float('-inf'), float('-inf')])
        
        # Calculate hypervolume
        hv = pg.hypervolume(selected_points).compute(reference_point)
        
        running_time = time.time() - start_time
        
        # Return [hypervolume, -running_time] as objectives (maximize HV, minimize time)
        return np.array([hv, -running_time])
    except Exception as e:
        # If evaluation fails, return negative infinity
        return np.array([float('-inf'), float('-inf')])


class HSS_Evaluation(Evaluation):
    """Evaluator for Hypervolume Subset Selection (HSS) problem."""

    def __init__(self,
                 timeout_seconds: int = 60,
                 k: int = 8,
                 data_folder: str = '../../data/train_data',
                 data_key: str = 'points',
                 **kwargs):
        """
        Args:
            timeout_seconds (int): Timeout for evaluation (default: 60).
            k (int): Number of points to select (default: 20).
            data_folder (str): Folder that contains multiple data files (default: './data/hss_data').
            data_key (str): The key of the data to load from each file (default: 'points').

        Notes:
            All files in `data_folder` will be loaded (currently supports '.mat' files),
            evaluated one by one, and the **average** objectives (hypervolume and time)
            over all files will be used as the final score of an algorithm.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.k = k

        # Load all data files from the folder
        if not os.path.isdir(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        self._datasets: list[np.ndarray] = []
        for fname in os.listdir(data_folder):
            if not fname.lower().endswith('.mat'):
                continue
            fpath = os.path.join(data_folder, fname)
            try:
                data = load_mat_to_numpy(fpath, data_key)
                if data is None:
                    continue
                # Keep the first 1000 points as in the original example (if needed)
                # data = data[:1000, :]
                self._datasets.append(data)
            except Exception as e:
                # Skip files that cannot be loaded correctly
                continue

        if len(self._datasets) == 0:
            raise RuntimeError(f"No valid data found in folder: {data_folder} (key='{data_key}')")

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        """
        Evaluate the HSS function on **all** datasets in the folder and
        return the **average** objectives.

        Args:
            program_str: The program string (not used here).
            callable_func: The HSS function to evaluate.

        Returns:
            Numpy array with [avg_hypervolume, -avg_running_time] as objectives.
        """
        hv_list = []
        time_list = []

        for points in self._datasets:
            # For each dataset, compute its own reference point
            reference_point = np.max(points, axis=0) * 1.1
            res = evaluate(points, self.k, reference_point, callable_func)

            hv = res[0]
            runtime = -res[1]   # res[1] is -time

            # Skip invalid evaluations
            if np.isinf(hv) or np.isinf(runtime):
                return np.array([float('-inf'), float('-inf')])

            hv_list.append(hv)
            time_list.append(runtime)

        if len(hv_list) == 0:
            # If all evaluations failed, return -inf
            return np.array([float('-inf'), float('-inf')])

        avg_hv = float(np.mean(hv_list))
        avg_time = float(np.mean(time_list))

        # Return [avg_hv, -avg_time] (maximize HV, minimize time)
        return np.array([avg_hv, -avg_time])


if __name__ == '__main__':
    import numpy as np

    def test_hss(points, k: int, reference_point) -> np.ndarray:
        """
        A simple test HSS function that randomly selects k points.
        
        Args:
            points: Numpy array of shape (n, m) where each row is a point
            k: Number of points to select
            reference_point: Numpy array of shape (m,) representing the reference point
        
        Returns:
            Numpy array of shape (k, m) with selected points
        """
        np.random.seed(2025)
        n = points.shape[0]
        indices = np.random.choice(n, size=k, replace=False)
        return points[indices, :]

    # Example: assume all data files are stored in './data/hss_data'
    hss_eval = HSS_Evaluation(k=20, data_folder='../../data/train_data', data_key='points')
    result = hss_eval.evaluate_program('_', test_hss)
    print(f"Evaluation result: {result}")
    print(f"Hypervolume: {result[0]}")
    print(f"Running time: {-result[1]} seconds")
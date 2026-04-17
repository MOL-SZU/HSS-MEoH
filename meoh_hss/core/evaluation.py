from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = WORKSPACE_DIR.parent
for path in (WORKSPACE_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm4ad.base import Evaluation
from core.template import template_program, task_description
from core.mat2array import load_mat_to_numpy
from paths import TRAIN_DATA_DIR

__all__ = ['HSS_Evaluation']


def evaluate(points: np.ndarray, k: int, reference_point: np.ndarray, hss_func: callable) -> np.ndarray:
    import time
    import pygmo as pg

    start_time = time.time()
    try:
        selected_points = np.asarray(hss_func(points, k, reference_point))
        if selected_points.ndim != 2 or selected_points.shape[0] != k:
            return np.array([float('-inf'), float('-inf')])
        hv = pg.hypervolume(selected_points).compute(reference_point)
        return np.array([hv, -(time.time() - start_time)])
    except Exception:
        return np.array([float('-inf'), float('-inf')])


class HSS_Evaluation(Evaluation):
    def __init__(self, timeout_seconds: int = 60, k: int = 8, data_folder: str | os.PathLike[str] = TRAIN_DATA_DIR, data_key: str = 'points', **kwargs):
        super().__init__(template_program=template_program, task_description=task_description, use_numba_accelerate=False, timeout_seconds=timeout_seconds)
        self.k = k
        self.data_folder = Path(data_folder)
        if not self.data_folder.is_dir():
            raise FileNotFoundError(f'Data folder not found: {self.data_folder}')
        self._datasets: list[np.ndarray] = []
        for fpath in sorted(self.data_folder.iterdir()):
            if fpath.suffix.lower() != '.mat':
                continue
            try:
                data = load_mat_to_numpy(str(fpath), data_key)
            except Exception:
                continue
            if data is not None:
                self._datasets.append(np.asarray(data))
        if not self._datasets:
            raise RuntimeError(f'No valid data found in folder: {self.data_folder}')

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        hv_list = []
        time_list = []
        for points in self._datasets:
            reference_point = np.max(points, axis=0) * 1.1
            res = evaluate(points, self.k, reference_point, callable_func)
            hv = res[0]
            runtime = -res[1]
            if np.isinf(hv) or np.isinf(runtime):
                return np.array([float('-inf'), float('-inf')])
            hv_list.append(hv)
            time_list.append(runtime)
        if not hv_list:
            return np.array([float('-inf'), float('-inf')])
        return np.array([float(np.mean(hv_list)), -float(np.mean(time_list))])

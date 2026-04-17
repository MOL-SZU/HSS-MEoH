from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pygmo as pg

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HSS_benchmark.mat2array import load_mat_to_numpy
from HSS_benchmark.GL_HSS import HSS as GL_HSS
from HSS_benchmark.GHSS import HSS as GHSS
from HSS_benchmark.GAHSS import HSS as GAHSS
from HSS_benchmark.GSI_LS import HSS as GSI_LS
from HSS_benchmark.TPOSS import HSS as TPOSS
from HSS_benchmark.SPESS import HSS as SPESS
from meoh_hss.paths import TRAIN_DATA_DIR, TRAIN_RESULT_DIR, ensure_result_dirs


def hv_cal(selected_objectives: np.ndarray, reference_point: np.ndarray) -> float:
    return pg.hypervolume(selected_objectives).compute(reference_point)


def eval_on_file(row_data: np.ndarray, algo_fn, k: int, ref: float, num_runs: int = 3) -> Dict[str, float]:
    reference_point = np.max(row_data, axis=0) * ref
    hv_list = []
    time_list = []
    for _ in range(num_runs):
        start = time.time()
        subset = algo_fn(row_data, k, reference_point)
        time_list.append(float(time.time() - start))
        hv_list.append(float(hv_cal(subset, reference_point)))
    return {'hv': float(np.mean(hv_list)), 'time': float(np.mean(time_list)), 'runs': {'hv_list': hv_list, 'time_list': time_list, 'num_runs': num_runs}}


def run_benchmark(data_folder: str = str(TRAIN_DATA_DIR), output_folder: str = str(TRAIN_RESULT_DIR), k: int = 8, ref: float = 1.1):
    ensure_result_dirs()
    os.makedirs(output_folder, exist_ok=True)
    mat_files = sorted(glob.glob(os.path.join(data_folder, '*.mat')))
    if not mat_files:
        raise FileNotFoundError(f'No training data found in {data_folder}')
    algos = {'GL_HSS': GL_HSS, 'GHSS': GHSS, 'GAHSS': GAHSS, 'GSI_LS': GSI_LS, 'TPOSS': TPOSS, 'SPESS': SPESS}
    for algo_name, algo_fn in algos.items():
        details: List[Dict[str, float]] = []
        hv_list, time_list = [], []
        for fpath in mat_files:
            try:
                row_data = load_mat_to_numpy(fpath, 'points')
                if row_data is None:
                    continue
                row_data = row_data[:1000, :]
                res = eval_on_file(row_data, algo_fn, k=k, ref=ref)
                res['file'] = os.path.basename(fpath)
                details.append(res)
                hv_list.append(res['hv'])
                time_list.append(res['time'])
            except Exception as exc:
                details.append({'file': os.path.basename(fpath), 'error': str(exc)})
        result = {'summary': {'avg_hv': float(np.mean(hv_list)) if hv_list else float('-inf'), 'avg_time': float(np.mean(time_list)) if time_list else float('inf'), 'num_files': len(hv_list)}, 'details': details}
        out_path = os.path.join(output_folder, f'{algo_name}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'[{algo_name}] wrote result: {out_path}')


if __name__ == '__main__':
    run_benchmark()

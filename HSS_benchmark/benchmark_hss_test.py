from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pygmo as pg
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HSS_benchmark.mat2array import load_mat_to_numpy
from HSS_benchmark.DPP import HSS as DPP
from HSS_benchmark.GHSS import HSS as GHSS
from HSS_benchmark.GAHSS import HSS as GAHSS
from HSS_benchmark.GSI_LS import HSS as GSI_LS
from meoh_hss.paths import TEST_DATA_DIR, TEST_RESULT_DIR, ensure_result_dirs


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


def run_benchmark_on_folder(data_folder: str, algo_name: str, algo_fn, k: int, ref: float = 1.1) -> Dict:
    mat_files = sorted(glob.glob(os.path.join(data_folder, '*.mat')))
    if not mat_files:
        raise FileNotFoundError(f'No test data found in {data_folder}')
    details: List[Dict[str, float]] = []
    hv_list, time_list = [], []
    with tqdm(total=len(mat_files), desc=f'{algo_name} (k={k})', unit='file', leave=False) as pbar:
        for fpath in mat_files:
            fname = os.path.basename(fpath)
            pbar.set_postfix({'file': fname})
            try:
                row_data = load_mat_to_numpy(fpath, 'points')
                if row_data is None:
                    continue
                res = eval_on_file(row_data, algo_fn, k=k, ref=ref)
                res['file'] = fname
                details.append(res)
                hv_list.append(res['hv'])
                time_list.append(res['time'])
            except Exception as exc:
                details.append({'file': fname, 'error': str(exc)})
            finally:
                pbar.update(1)
    return {'summary': {'avg_hv': float(np.mean(hv_list)) if hv_list else float('-inf'), 'avg_time': float(np.mean(time_list)) if time_list else float('inf'), 'num_files': len(hv_list)}, 'details': details}


def run_benchmark(test_data_root: str = str(TEST_DATA_DIR), output_root: str = str(TEST_RESULT_DIR / 'baseline'), k_list: List[int] = [5, 10, 15], ref: float = 1.1):
    ensure_result_dirs()
    test_folders = sorted(d for d in os.listdir(test_data_root) if os.path.isdir(os.path.join(test_data_root, d)))
    algos = {'DPP': DPP, 'GHSS': GHSS, 'GAHSS': GAHSS, 'GSI_LS': GSI_LS}
    for test_folder in test_folders:
        test_data_dir = os.path.join(test_data_root, test_folder)
        for k in k_list:
            out_dir = os.path.join(output_root, f'{test_folder}_{k}')
            os.makedirs(out_dir, exist_ok=True)
            for algo_name, algo_fn in algos.items():
                out_path = os.path.join(out_dir, f'{algo_name}.json')
                if os.path.exists(out_path):
                    continue
                result = run_benchmark_on_folder(test_data_dir, algo_name, algo_fn, k=k, ref=ref)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f'[INFO] saved {algo_name} -> {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate HSS baselines on all test datasets.')
    parser.add_argument('--test_data_root', type=str, default=str(TEST_DATA_DIR))
    parser.add_argument('--output_root', type=str, default=str(TEST_RESULT_DIR / 'baseline'))
    parser.add_argument('--k_list', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--ref', type=float, default=1.1)
    args = parser.parse_args()
    run_benchmark(args.test_data_root, args.output_root, args.k_list, args.ref)

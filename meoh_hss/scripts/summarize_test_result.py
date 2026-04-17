from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = WORKSPACE_DIR.parent
for path in (WORKSPACE_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paths import TEST_RESULT_DIR

BASELINE_ORDER = ['GL_HSS', 'GAHSS', 'GHSS', 'GSI_LS', 'SPESS', 'TPOSS']


def list_dataset_dirs(result_root: str) -> List[str]:
    if not os.path.isdir(result_root):
        raise FileNotFoundError(f'Result folder not found: {result_root}')
    return sorted(name for name in os.listdir(result_root) if os.path.isdir(os.path.join(result_root, name)))


def load_summary(json_path: str) -> Tuple[float, float] | None:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    summary = data.get('summary') if isinstance(data, dict) else None
    if not isinstance(summary, dict):
        return None
    avg_hv = summary.get('avg_hv')
    avg_time = summary.get('avg_time')
    if avg_hv is None or avg_time is None:
        return None
    return float(avg_hv), float(avg_time)


def collect_algorithm_results(dataset_dir: str) -> Dict[str, Tuple[float, float]]:
    results = {}
    for name in sorted(os.listdir(dataset_dir)):
        if not name.lower().endswith('.json'):
            continue
        summary = load_summary(os.path.join(dataset_dir, name))
        if summary is not None:
            results[Path(name).stem] = summary
    return results


def select_fastest(results: Dict[str, Tuple[float, float]]):
    return min(results.items(), key=lambda item: (item[1][1], -item[1][0], item[0])) if results else None


def select_best_performance(results: Dict[str, Tuple[float, float]]):
    return max(results.items(), key=lambda item: (item[1][0], -item[1][1], item[0])) if results else None


def format_result_line(name: str, result: Tuple[float, float]) -> str:
    score, run_time = result
    return f'{name}: [hv={score:.6f}, time={run_time:.6f}]'


def summarize_dataset(dataset_name: str, result_root: str, baseline_root: str) -> str:
    result_algorithms = collect_algorithm_results(os.path.join(result_root, dataset_name))
    baseline_dir = os.path.join(baseline_root, dataset_name)
    baseline_algorithms = collect_algorithm_results(baseline_dir) if os.path.isdir(baseline_dir) else {}
    fastest = select_fastest(result_algorithms)
    best_perf = select_best_performance(result_algorithms)
    lines = [f'{dataset_name}:']
    lines.append(f'Fastest algorithm: {format_result_line(fastest[0], fastest[1])}' if fastest else 'Fastest algorithm: no valid result')
    lines.append(f'Best performance algorithm: {format_result_line(best_perf[0], best_perf[1])}' if best_perf else 'Best performance algorithm: no valid result')
    lines.append('Baselines:')
    for baseline_name in BASELINE_ORDER:
        if baseline_name in baseline_algorithms:
            lines.append(f'  {format_result_line(baseline_name, baseline_algorithms[baseline_name])}')
        else:
            lines.append(f'  {baseline_name}: no result')
    return '\n'.join(lines)


def summarize_result_folder(result_root: str, baseline_root: str) -> str:
    return '\n\n'.join(summarize_dataset(dataset_name, result_root, baseline_root) for dataset_name in list_dataset_dirs(result_root))


def write_text(path: str, content: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description='Summarize the fastest and best-performing algorithms for each test dataset.')
    parser.add_argument('--result_root', type=str, required=True, help='Result folder to analyze')
    parser.add_argument('--baseline_root', type=str, default=str(TEST_RESULT_DIR / 'baseline'), help='Baseline result folder')
    parser.add_argument('--save', type=str, default=None, help='Optional output text path')
    args = parser.parse_args()
    content = summarize_result_folder(args.result_root, args.baseline_root)
    print(content)
    if args.save:
        write_text(args.save, content)
        print(f'\nSaved to: {args.save}')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        result_root = str(TEST_RESULT_DIR / 'exp_raw')
        baseline_root = str(TEST_RESULT_DIR / 'baseline')
        save_path = str(TEST_RESULT_DIR / 'exp_raw_summary.txt')
        content = summarize_result_folder(result_root, baseline_root)
        print(content)
        write_text(save_path, content)
        print(f'\nSaved to: {save_path}')
    else:
        main()

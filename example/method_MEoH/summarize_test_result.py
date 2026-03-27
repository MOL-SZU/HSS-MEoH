from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple


BASELINE_ORDER = [
    "GL_HSS",
    "GAHSS",
    "GHSS",
    "GSI_LS",
    "SPESS",
    "TPOSS",
]


def list_dataset_dirs(result_root: str) -> List[str]:
    if not os.path.isdir(result_root):
        raise FileNotFoundError(f"结果文件夹不存在: {result_root}")
    dataset_dirs = [
        name for name in os.listdir(result_root)
        if os.path.isdir(os.path.join(result_root, name))
    ]
    return sorted(dataset_dirs)


def load_summary(json_path: str) -> Tuple[float, float] | None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return None

    avg_hv = summary.get("avg_hv")
    avg_time = summary.get("avg_time")
    if avg_hv is None or avg_time is None:
        return None
    return float(avg_hv), float(avg_time)


def collect_algorithm_results(dataset_dir: str) -> Dict[str, Tuple[float, float]]:
    results = {}
    for name in sorted(os.listdir(dataset_dir)):
        if not name.lower().endswith(".json"):
            continue
        json_path = os.path.join(dataset_dir, name)
        summary = load_summary(json_path)
        if summary is None:
            continue
        algorithm_name = os.path.splitext(name)[0]
        results[algorithm_name] = summary
    return results


def select_fastest(results: Dict[str, Tuple[float, float]]) -> Tuple[str, Tuple[float, float]] | None:
    if not results:
        return None
    algorithm = min(results.items(), key=lambda item: (item[1][1], -item[1][0], item[0]))
    return algorithm[0], algorithm[1]


def select_best_performance(results: Dict[str, Tuple[float, float]]) -> Tuple[str, Tuple[float, float]] | None:
    if not results:
        return None
    algorithm = max(results.items(), key=lambda item: (item[1][0], -item[1][1], item[0]))
    return algorithm[0], algorithm[1]


def format_result_line(name: str, result: Tuple[float, float]) -> str:
    score, run_time = result
    return f"{name}: [{score:.6f}, {run_time:.6f}]"


def summarize_dataset(dataset_name: str, result_root: str, baseline_root: str) -> str:
    dataset_result_dir = os.path.join(result_root, dataset_name)
    dataset_baseline_dir = os.path.join(baseline_root, dataset_name)

    result_algorithms = collect_algorithm_results(dataset_result_dir)
    baseline_algorithms = collect_algorithm_results(dataset_baseline_dir) if os.path.isdir(dataset_baseline_dir) else {}

    fastest = select_fastest(result_algorithms)
    best_perf = select_best_performance(result_algorithms)

    lines = [f"{dataset_name}："]

    if fastest is None:
        lines.append("最快算法：无有效结果")
    else:
        lines.append(f"最快算法：{format_result_line(fastest[0], fastest[1])}")

    if best_perf is None:
        lines.append("最高效算法：无有效结果")
    else:
        lines.append(f"最高效算法：{format_result_line(best_perf[0], best_perf[1])}")

    lines.append("六种baseline算法：")
    for baseline_name in BASELINE_ORDER:
        if baseline_name in baseline_algorithms:
            lines.append(f"  {format_result_line(baseline_name, baseline_algorithms[baseline_name])}")
        else:
            lines.append(f"  {baseline_name}: 无结果")

    return "\n".join(lines)


def summarize_result_folder(result_root: str, baseline_root: str) -> str:
    dataset_names = list_dataset_dirs(result_root)
    sections = [summarize_dataset(dataset_name, result_root, baseline_root) for dataset_name in dataset_names]
    return "\n\n".join(sections)


def write_text(path: str, content: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="汇总 test_result 文件夹中每个数据集的最快算法、最高效算法和六种 baseline 算法结果。")
    parser.add_argument("--result_root", type=str, required=True, help="待分析的结果文件夹")
    parser.add_argument("--baseline_root", type=str, default="./test_result/baseline", help="baseline 结果文件夹")
    parser.add_argument("--save", type=str, default=None, help="可选，保存输出文本路径")
    args = parser.parse_args()

    content = summarize_result_folder(args.result_root, args.baseline_root)
    print(content)
    if args.save:
        write_text(args.save, content)
        print(f"\n已保存到: {args.save}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("=" * 60)
        print("右键运行模式：请直接在脚本中修改默认路径")
        print("=" * 60)

        result_root = "./test_result/exp_raw"
        baseline_root = "./test_result/baseline"
        save_path = "./test_result/exp_raw_summary.txt"

        content = summarize_result_folder(result_root, baseline_root)
        print(content)
        write_text(save_path, content)
        print(f"\n已保存到: {save_path}")
    else:
        main()

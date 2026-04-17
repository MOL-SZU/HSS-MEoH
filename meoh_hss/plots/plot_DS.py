from __future__ import annotations

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paths import IMAGES_DIR, LOGS_DIR, TEST_RESULT_DIR, TRAIN_RESULT_DIR

import argparse
import glob
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from codebleu.syntax_match import calc_syntax_match

BENCHMARK_ALGORITHMS = {
    "GL_HSS",
    "GAHSS",
    "GHSS",
    "GSI_LS",
    "SPESS",
    "TPOSS",
}


def find_elitist_files(log_dir: str) -> List[Tuple[int, str]]:
    elitist_dir = os.path.join(log_dir, "elitist")
    pattern = os.path.join(elitist_dir, "elitist_*.json")
    files = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        try:
            generation = int(name.replace("elitist_", "").replace(".json", ""))
        except ValueError:
            continue
        files.append((generation, path))
    return sorted(files, key=lambda item: item[0])


def load_elitist_records(file_path: str, *, exclude_benchmarks: bool = False) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        algorithm = item.get("algorithm", "")
        if exclude_benchmarks and algorithm in BENCHMARK_ALGORITHMS:
            continue
        score = item.get("score")
        function_code = item.get("function", "")
        if score is None or len(score) < 2 or not function_code:
            continue
        score_array = np.array(score[:2], dtype=float)
        if np.isinf(score_array).any():
            continue
        records.append(
            {
                "score": score_array,
                "function": function_code,
            }
        )
    return records


def compute_ds_values(records: List[dict]) -> np.ndarray:
    n = len(records)
    if n == 0:
        return np.array([], dtype=float)

    dominated_counts = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if (records[i]["score"] >= records[j]["score"]).all():
                dominated_counts[i, j] = -calc_syntax_match(
                    [records[i]["function"]],
                    records[j]["function"],
                    "python",
                )
            elif (records[j]["score"] >= records[i]["score"]).all():
                dominated_counts[j, i] = -calc_syntax_match(
                    [records[j]["function"]],
                    records[i]["function"],
                    "python",
                )
    return dominated_counts.sum(0)


def build_ds_series(log_dir: str) -> Tuple[List[int], List[float]]:
    generations = []
    ds_sums = []
    for generation, path in find_elitist_files(log_dir):
        records = load_elitist_records(path, exclude_benchmarks=True)
        ds_values = compute_ds_values(records)
        generations.append(generation)
        ds_sums.append(float(np.sum(ds_values)) if len(ds_values) else 0.0)
    return generations, ds_sums


def plot_ds_curves(
    log_dir_a: str,
    log_dir_b: str,
    label_a: str,
    label_b: str,
    save_path: str | None,
    show_plot: bool,
):
    generations_a, ds_a = build_ds_series(log_dir_a)
    generations_b, ds_b = build_ds_series(log_dir_b)

    plt.figure(figsize=(10, 6))
    plt.plot(generations_a, ds_a, color="#1f77b4", linewidth=2.2, label=label_a)
    plt.plot(generations_b, ds_b, color="#d62728", linewidth=2.2, label=label_b)
    plt.xlabel("Generation", fontsize=12, fontweight="bold")
    plt.ylabel("DS Sum", fontsize=12, fontweight="bold")
    plt.title("Dominance-Dissimilarity Sum Curve", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"已保存图像到: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="绘制两个 MEoH 结果文件夹的 DS 总和曲线。")
    parser.add_argument("--log_dir_a", type=str, required=True, help="第一个 MEoH 结果文件夹")
    parser.add_argument("--log_dir_b", type=str, required=True, help="第二个 MEoH 结果文件夹")
    parser.add_argument("--label_a", type=str, default="Result A", help="第一个结果集标签")
    parser.add_argument("--label_b", type=str, default="Result B", help="第二个结果集标签")
    parser.add_argument("--save", type=str, default=None, help="保存图片路径")
    parser.add_argument("--no-show", action="store_true", help="不显示图片")
    args = parser.parse_args()

    plot_ds_curves(
        log_dir_a=args.log_dir_a,
        log_dir_b=args.log_dir_b,
        label_a=args.label_a,
        label_b=args.label_b,
        save_path=args.save,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("=" * 60)
        print("右键运行模式：请直接在脚本中修改默认路径")
        print("=" * 60)

        log_dir_a = str(LOGS_DIR / "meoh" / "exp_baseline")
        log_dir_b = str(LOGS_DIR / "meoh" / "exp_raw")
        label_a = "Baseline"
        label_b = "exp_raw"
        save_path = str(IMAGES_DIR / "ds_curve.png")

        plot_ds_curves(
            log_dir_a=log_dir_a,
            log_dir_b=log_dir_b,
            label_a=label_a,
            label_b=label_b,
            save_path=save_path,
            show_plot=True,
        )
    else:
        main()

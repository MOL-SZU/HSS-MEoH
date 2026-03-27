from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt

from plot_config import (
    DEFAULT_PLOT_STYLE,
    PlotStyleConfig,
    add_plot_style_arguments,
    build_plot_style_config,
)
from plot_presets import DEFAULT_COMPARISON_PRESETS

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


def count_filtered_elitist(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for item in data:
        algorithm = item.get("algorithm", "")
        if algorithm in BENCHMARK_ALGORITHMS:
            continue
        score = item.get("score")
        if score is None or len(score) < 2:
            continue
        count += 1
    return count


def build_number_series(log_dir: str) -> Tuple[List[int], List[int]]:
    generations = []
    counts = []
    for generation, path in find_elitist_files(log_dir):
        generations.append(generation)
        counts.append(count_filtered_elitist(path))
    return generations, counts


def plot_number_curves(
    log_dir_a: str,
    log_dir_b: str,
    label_a: str,
    label_b: str,
    save_path: str | None,
    show_plot: bool,
    style_config: PlotStyleConfig = DEFAULT_PLOT_STYLE,
):
    generations_a, counts_a = build_number_series(log_dir_a)
    generations_b, counts_b = build_number_series(log_dir_b)

    plt.figure(figsize=style_config.figsize)
    plt.plot(generations_a, counts_a, color="#1f77b4", linewidth=2.2, label=label_a)
    plt.plot(generations_b, counts_b, color="#d62728", linewidth=2.2, label=label_b)
    plt.xlabel("Generation", fontsize=style_config.label_fontsize, fontweight="bold")
    plt.ylabel("Filtered Elitist Count", fontsize=style_config.label_fontsize, fontweight="bold")
    plt.title("Elitist Count Curve", fontsize=style_config.title_fontsize, fontweight="bold")
    plt.xticks(fontsize=style_config.tick_fontsize)
    plt.yticks(fontsize=style_config.tick_fontsize)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=style_config.legend_fontsize)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot elitist-count curves for two MEoH runs.")
    parser.add_argument("--log_dir_a", type=str, required=True, help="First MEoH log directory")
    parser.add_argument("--log_dir_b", type=str, required=True, help="Second MEoH log directory")
    parser.add_argument("--label_a", type=str, default="Result A", help="Legend label for the first run")
    parser.add_argument("--label_b", type=str, default="Result B", help="Legend label for the second run")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    add_plot_style_arguments(parser)
    args = parser.parse_args()
    style_config = build_plot_style_config(args)

    plot_number_curves(
        log_dir_a=args.log_dir_a,
        log_dir_b=args.log_dir_b,
        label_a=args.label_a,
        label_b=args.label_b,
        save_path=args.save,
        show_plot=not args.no_show,
        style_config=style_config,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("=" * 60)
        print("Interactive Mode - Generate all preset number plots")
        print("=" * 60)

        for preset in DEFAULT_COMPARISON_PRESETS:
            plot_number_curves(
                log_dir_a=preset.log_dir,
                log_dir_b=preset.compare_log_dir,
                label_a=preset.log_label,
                label_b=preset.compare_label,
                save_path=f"image/number_{preset.suffix}.png",
                show_plot=False,
                style_config=DEFAULT_PLOT_STYLE,
            )
    else:
        main()

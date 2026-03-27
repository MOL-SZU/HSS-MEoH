from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygmo as pg

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


def load_scores(file_path: str, *, exclude_benchmarks: bool = False) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = []
    for item in data:
        algorithm = item.get("algorithm", "")
        if exclude_benchmarks and algorithm in BENCHMARK_ALGORITHMS:
            continue
        score = item.get("score")
        if score is None or len(score) < 2:
            continue
        score_array = np.array(score[:2], dtype=float)
        if np.isinf(score_array).any():
            continue
        scores.append(score_array)

    if not scores:
        return np.empty((0, 2), dtype=float)
    return np.array(scores, dtype=float)


def load_generation_scores(log_dir: str, *, exclude_benchmarks: bool = False) -> Dict[int, np.ndarray]:
    return {
        generation: load_scores(path, exclude_benchmarks=exclude_benchmarks)
        for generation, path in find_elitist_files(log_dir)
    }


def compute_minimization_bounds(log_dir_a: str, log_dir_b: str) -> Tuple[np.ndarray, np.ndarray]:
    final_files_a = find_elitist_files(log_dir_a)
    final_files_b = find_elitist_files(log_dir_b)
    if not final_files_a or not final_files_b:
        raise FileNotFoundError("Missing elitist_*.json files in one of the log directories.")

    final_scores_a = load_scores(final_files_a[-1][1], exclude_benchmarks=True)
    final_scores_b = load_scores(final_files_b[-1][1], exclude_benchmarks=True)
    final_scores = np.vstack([final_scores_a, final_scores_b])
    if len(final_scores) == 0:
        raise ValueError("No valid non-benchmark elitist scores were found.")

    minimization_scores = -final_scores
    mins = np.min(minimization_scores, axis=0)
    maxs = np.max(minimization_scores, axis=0)
    return mins, maxs


def normalize_minimization_scores(scores: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return np.empty((0, 2), dtype=float)

    minimization_scores = -scores
    spans = np.where(np.abs(maxs - mins) < 1e-12, 1.0, maxs - mins)
    normalized = (minimization_scores - mins) / spans
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def compute_hv(points: np.ndarray, reference_point: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    hv = pg.hypervolume(points)
    return float(hv.compute(reference_point))


def build_hv_series(log_dir: str, mins: np.ndarray, maxs: np.ndarray) -> Tuple[List[int], List[float]]:
    generation_scores = load_generation_scores(log_dir, exclude_benchmarks=True)
    generations = sorted(generation_scores.keys())
    reference_point = np.array([1.1, 1.1], dtype=float)
    hv_values = []

    for generation in generations:
        normalized = normalize_minimization_scores(generation_scores[generation], mins, maxs)
        hv_values.append(compute_hv(normalized, reference_point))

    return generations, hv_values


def plot_hv_curves(
    log_dir_a: str,
    log_dir_b: str,
    label_a: str,
    label_b: str,
    save_path: str | None,
    show_plot: bool,
    style_config: PlotStyleConfig = DEFAULT_PLOT_STYLE,
):
    mins, maxs = compute_minimization_bounds(log_dir_a, log_dir_b)
    generations_a, hv_a = build_hv_series(log_dir_a, mins, maxs)
    generations_b, hv_b = build_hv_series(log_dir_b, mins, maxs)

    if generations_a and hv_a:
        print(f"{label_a} final generation {generations_a[-1]} HV = {hv_a[-1]:.6f}")
    if generations_b and hv_b:
        print(f"{label_b} final generation {generations_b[-1]} HV = {hv_b[-1]:.6f}")

    plt.figure(figsize=style_config.figsize)
    plt.plot(generations_a, hv_a, color="#1f77b4", linewidth=2.2, label=label_a)
    plt.plot(generations_b, hv_b, color="#d62728", linewidth=2.2, label=label_b)
    plt.xlabel("Generation", fontsize=style_config.label_fontsize, fontweight="bold")
    plt.ylabel("HV", fontsize=style_config.label_fontsize, fontweight="bold")
    plt.title("Hypervolume Curve", fontsize=style_config.title_fontsize, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="Plot HV curves for two MEoH runs.")
    parser.add_argument("--log_dir_a", type=str, required=True, help="First MEoH log directory")
    parser.add_argument("--log_dir_b", type=str, required=True, help="Second MEoH log directory")
    parser.add_argument("--label_a", type=str, default="Result A", help="Legend label for the first run")
    parser.add_argument("--label_b", type=str, default="Result B", help="Legend label for the second run")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    add_plot_style_arguments(parser)
    args = parser.parse_args()
    style_config = build_plot_style_config(args)

    plot_hv_curves(
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
        print("Interactive Mode - Generate all preset HV plots")
        print("=" * 60)

        for preset in DEFAULT_COMPARISON_PRESETS:
            plot_hv_curves(
                log_dir_a=preset.log_dir,
                log_dir_b=preset.compare_log_dir,
                label_a=preset.log_label,
                label_b=preset.compare_label,
                save_path=f"image/hv_{preset.suffix}.png",
                show_plot=False,
                style_config=DEFAULT_PLOT_STYLE,
            )
    else:
        main()

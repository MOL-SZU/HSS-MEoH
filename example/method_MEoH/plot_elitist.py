"""
Plot Pareto-front comparisons for MEoH experiments and benchmark baselines.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from plot_config import (
    DEFAULT_PLOT_STYLE,
    PlotStyleConfig,
    add_plot_style_arguments,
    build_plot_style_config,
)
from plot_presets import DEFAULT_COMPARISON_PRESETS

STYLE_MAP = {
    "MEoH": {"color": "#1f77b4", "marker": "o"},
    "MEoH-HS": {"color": "#ff7f0e", "marker": "s"},
    "HS-MEoH": {"color": "#ff7f0e", "marker": "s"},
    "GAHSS": {"color": "#2ca02c", "marker": "^"},
    "GHSS": {"color": "#d62728", "marker": "D"},
    "GL_HSS": {"color": "#9467bd", "marker": "v"},
    "GSI_LS": {"color": "#8c564b", "marker": "p"},
}

BENCHMARK_ALGORITHMS = {
    "GAHSS",
    "GHSS",
    "GL_HSS",
    "GSI_LS",
    "TPOSS",
    "SPESS",
}


def load_elitist_from_file(file_path: str) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    objectives = []
    for item in data:
        if "score" in item and item["score"] is not None:
            score = item["score"]
            if isinstance(score, list) and len(score) == 2:
                objectives.append(score)

    return np.array(objectives)


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    if objectives is None or len(objectives) == 0:
        return np.empty((0, 2))

    scores = np.asarray(objectives, dtype=float)
    is_dominated = np.zeros(len(scores), dtype=bool)

    for i, score_i in enumerate(scores):
        if is_dominated[i]:
            continue
        for j, score_j in enumerate(scores):
            if i == j:
                continue
            if np.all(score_j >= score_i) and np.any(score_j > score_i):
                is_dominated[i] = True
                break

    return scores[~is_dominated]


def find_latest_elitist_file(log_dir: str) -> Optional[str]:
    elitist_dir = os.path.join(log_dir, "elitist")
    if not os.path.exists(elitist_dir):
        return None

    pattern = os.path.join(elitist_dir, "elitist_*.json")
    files = glob.glob(pattern)
    if not files:
        return None

    def get_generation(filepath: str) -> int:
        filename = os.path.basename(filepath)
        try:
            return int(filename.replace("elitist_", "").replace(".json", ""))
        except ValueError:
            return -1

    files.sort(key=get_generation, reverse=True)
    return files[0]


def load_algorithm_results(file_path: str, format_type: str = "auto") -> np.ndarray:
    if format_type == "auto":
        if file_path.endswith(".json"):
            format_type = "json"
        elif file_path.endswith(".csv"):
            format_type = "csv"
        elif file_path.endswith(".npy"):
            format_type = "npy"
        else:
            format_type = "json"

    if format_type == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "summary" in data:
            summary = data["summary"]
            avg_hv = summary.get("avg_hv")
            avg_time = summary.get("avg_time")
            if avg_hv is None or avg_time is None:
                raise ValueError(f"Summary in {file_path} missing 'avg_hv' or 'avg_time'")
            return np.array([[avg_hv, -avg_time]])

        if isinstance(data, list):
            objectives = []
            for item in data:
                if isinstance(item, dict):
                    if "score" in item:
                        score = item["score"]
                    elif "objectives" in item:
                        score = item["objectives"]
                    elif "fitness" in item:
                        score = item["fitness"]
                    else:
                        values = list(item.values())
                        if len(values) >= 2:
                            score = [values[0], values[1]]
                        else:
                            continue
                elif isinstance(item, list) and len(item) >= 2:
                    score = item[:2]
                else:
                    continue

                if score is not None and len(score) >= 2:
                    objectives.append([score[0], score[1]])

            return np.array(objectives)

        raise ValueError(f"Unsupported JSON format in {file_path}")

    if format_type == "csv":
        data = np.loadtxt(file_path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        return data[:, :2]

    if format_type == "npy":
        data = np.load(file_path)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        return data[:, :2]

    raise ValueError(f"Unsupported format: {format_type}")


def get_plot_style(alg_name: str, idx: int) -> Dict[str, str]:
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "X", "P"]

    normalized_name = alg_name.strip()
    if normalized_name in STYLE_MAP:
        return STYLE_MAP[normalized_name]

    normalized_upper = normalized_name.upper()
    for style_name, style in STYLE_MAP.items():
        if style_name.upper() == normalized_upper:
            return style

    return {
        "color": colors[idx % len(colors)],
        "marker": markers[idx % len(markers)],
    }


def resolve_style_key(preferred_key: Optional[str], idx: int) -> str:
    if preferred_key:
        return preferred_key
    return f"__series_{idx}__"


def plot_pareto_fronts(
    results_dict: Dict[str, np.ndarray],
    style_key_dict: Optional[Dict[str, str]] = None,
    title: str = "Pareto Front Comparison",
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    style_config: PlotStyleConfig = DEFAULT_PLOT_STYLE,
    all_point_size: float = 80.0,
    front_point_size: float = 130.0,
    inset_all_point_size: float = 60.0,
    inset_front_point_size: float = 95.0,
    inset_enabled: bool = False,
    inset_zoom_region: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    inset_axes_rect: Tuple[float, float, float, float] = (0.52, 0.18, 0.34, 0.34),
):
    fig, ax = plt.subplots(figsize=style_config.figsize)
    inset_ax = None
    if inset_enabled:
        left, bottom, width, height = inset_axes_rect
        inset_ax = ax.inset_axes([left, bottom, width, height])

    for idx, (alg_name, objectives) in enumerate(results_dict.items()):
        if len(objectives) == 0:
            continue

        style_key = alg_name if style_key_dict is None else style_key_dict.get(alg_name, alg_name)
        style = get_plot_style(style_key, idx)
        color = style["color"]
        marker = style["marker"]

        all_points = np.asarray(objectives, dtype=float)
        sorted_indices = np.argsort(all_points[:, 0])
        sorted_obj = all_points[sorted_indices]
        front = pareto_front(all_points)

        ax.scatter(
            sorted_obj[:, 0],
            sorted_obj[:, 1],
            color=color,
            marker=marker,
            s=all_point_size,
            alpha=0.35,
            edgecolors="black",
            linewidths=0.4,
        )

        if inset_ax is not None:
            inset_ax.scatter(
                sorted_obj[:, 0],
                sorted_obj[:, 1],
                color=color,
                marker=marker,
                s=inset_all_point_size,
                alpha=0.35,
                edgecolors="black",
                linewidths=0.3,
            )

        if len(front) > 0:
            front = front[np.argsort(front[:, 0])]
            ax.plot(front[:, 0], front[:, 1], color=color, linewidth=2.0, alpha=0.9, zorder=2)
            ax.scatter(
                front[:, 0],
                front[:, 1],
                label=alg_name,
                color=color,
                marker=marker,
                s=front_point_size,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.8,
            )

            if inset_ax is not None:
                inset_ax.plot(front[:, 0], front[:, 1], color=color, linewidth=1.6, alpha=0.9, zorder=2)
                inset_ax.scatter(
                    front[:, 0],
                    front[:, 1],
                    color=color,
                    marker=marker,
                    s=inset_front_point_size,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.6,
                )

    ax.set_xlabel(xlabel, fontsize=style_config.label_fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=style_config.label_fontsize, fontweight="bold")
    ax.set_title(title, fontsize=style_config.title_fontsize, fontweight="bold")
    ax.tick_params(axis="both", labelsize=style_config.tick_fontsize)
    ax.legend(loc="best", fontsize=style_config.legend_fontsize, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    if inset_ax is not None:
        x1, x2, y1, y2 = inset_zoom_region
        inset_ax.set_xlim(x1, x2)
        inset_ax.set_ylim(y1, y2)
        inset_ax.grid(True, alpha=0.25, linestyle="--")
        inset_ax.tick_params(axis="both", labelsize=style_config.inset_tick_fontsize)
        mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.45", lw=1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def build_results_for_preset(
    log_dir: str,
    compare_log_dir: str,
    log_label: str,
    compare_label: str,
    log_style_key: str,
    compare_style_key: str,
    other_algs: list[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    results_dict: Dict[str, np.ndarray] = {}
    style_key_dict: Dict[str, str] = {}

    for idx, (current_label, current_log_dir, current_style_key) in enumerate(
        [
            (log_label, log_dir, log_style_key),
            (compare_label, compare_log_dir, compare_style_key),
        ]
    ):
        elitist_file = find_latest_elitist_file(current_log_dir)
        if elitist_file:
            print(f"Loading {current_label} elitist from: {elitist_file}")
            elitist_results = load_elitist_from_file(elitist_file)
            results_dict[current_label] = elitist_results
            style_key_dict[current_label] = resolve_style_key(current_style_key, idx)
            print(f"  Loaded {len(elitist_results)} solutions")
        else:
            print(f"Warning: No elitist file found in {current_log_dir}")

    for alg_spec in other_algs:
        if ":" not in alg_spec:
            continue
        alg_name, file_path = alg_spec.split(":", 1)
        if not os.path.exists(file_path):
            continue
        try:
            print(f"Loading {alg_name} from: {file_path}")
            alg_results = load_algorithm_results(file_path)
            results_dict[alg_name] = alg_results
            style_key_dict[alg_name] = alg_name
            print(f"  Loaded {len(alg_results)} solutions")
        except Exception as exc:
            print(f"Error loading {alg_name} from {file_path}: {exc}")

    return results_dict, style_key_dict


def main():
    parser = argparse.ArgumentParser(description="Plot elitist results and compare with other algorithms")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing MEoH log files")
    parser.add_argument("--compare_log_dir", type=str, default=None, help="Second log directory for comparison")
    parser.add_argument("--log_label", type=str, default="ours", help="Legend label for --log_dir")
    parser.add_argument("--compare_label", type=str, default="MEoH", help="Legend label for --compare_log_dir")
    parser.add_argument("--log_style_key", type=str, default="MEoH", help="Style key for --log_dir")
    parser.add_argument("--compare_style_key", type=str, default="MEoH-HS", help="Style key for --compare_log_dir")
    parser.add_argument(
        "--other_algs",
        type=str,
        nargs="*",
        default=[],
        help="Other algorithm result files in name:path format",
    )
    parser.add_argument("--title", type=str, default="Pareto Front Comparison", help="Plot title")
    parser.add_argument("--xlabel", type=str, default="Objective 1 (Hypervolume)", help="X-axis label")
    parser.add_argument("--ylabel", type=str, default="Objective 2 (-Running Time)", help="Y-axis label")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    parser.add_argument("--inset", action="store_true", help="Enable inset zoom view")
    parser.add_argument("--inset_region", type=float, nargs=4, default=[0.0, 1.0, 0.0, 1.0])
    parser.add_argument("--inset_rect", type=float, nargs=4, default=[0.52, 0.18, 0.34, 0.34])
    parser.add_argument("--all_point_size", type=float, default=80.0)
    parser.add_argument("--front_point_size", type=float, default=130.0)
    parser.add_argument("--inset_all_point_size", type=float, default=60.0)
    parser.add_argument("--inset_front_point_size", type=float, default=95.0)
    add_plot_style_arguments(parser)
    args = parser.parse_args()
    style_config = build_plot_style_config(args)

    results_dict, style_key_dict = build_results_for_preset(
        log_dir=args.log_dir,
        compare_log_dir=args.compare_log_dir,
        log_label=args.log_label,
        compare_label=args.compare_label,
        log_style_key=args.log_style_key,
        compare_style_key=args.compare_style_key,
        other_algs=args.other_algs,
    )

    if not results_dict:
        print("Error: No results to plot!")
        return

    plot_pareto_fronts(
        results_dict=results_dict,
        style_key_dict=style_key_dict,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        save_path=args.save,
        show_plot=not args.no_show,
        style_config=style_config,
        all_point_size=args.all_point_size,
        front_point_size=args.front_point_size,
        inset_all_point_size=args.inset_all_point_size,
        inset_front_point_size=args.inset_front_point_size,
        inset_enabled=args.inset,
        inset_zoom_region=tuple(args.inset_region),
        inset_axes_rect=tuple(args.inset_rect),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("=" * 60)
        print("Interactive Mode - Generate all preset Pareto plots")
        print("=" * 60)

        all_point_size = 80
        front_point_size = 130
        inset_all_point_size = 60
        inset_front_point_size = 95
        inset_enabled = True
        inset_zoom_region = (6.2, 6.4, -0.15, -0.005)
        inset_axes_rect = (0.5, 0.5, 0.4, 0.4)

        other_algs = [
            "GAHSS:./train_result/GAHSS.json",
            "GHSS:./train_result/GHSS.json",
            "GL_HSS:./train_result/GL_HSS.json",
            "GSI_LS:./train_result/GSI_LS.json",
            "TPOSS:./train_result/TPOSS.json",
            "SPESS:./train_result/SPESS.json",
        ]

        for preset in DEFAULT_COMPARISON_PRESETS:
            results_dict, style_key_dict = build_results_for_preset(
                log_dir=preset.log_dir,
                compare_log_dir=preset.compare_log_dir,
                log_label=preset.log_label,
                compare_label=preset.compare_label,
                log_style_key=preset.log_style_key,
                compare_style_key=preset.compare_style_key,
                other_algs=other_algs,
            )

            if results_dict:
                plot_pareto_fronts(
                    results_dict=results_dict,
                    style_key_dict=style_key_dict,
                    xlabel="Hypervolume",
                    ylabel="-Running Time",
                    save_path=preset.pareto_save_path,
                    show_plot=False,
                    style_config=DEFAULT_PLOT_STYLE,
                    all_point_size=all_point_size,
                    front_point_size=front_point_size,
                    inset_all_point_size=inset_all_point_size,
                    inset_front_point_size=inset_front_point_size,
                    inset_enabled=inset_enabled,
                    inset_zoom_region=inset_zoom_region,
                    inset_axes_rect=inset_axes_rect,
                )
            else:
                print("No results to plot. Please check the file paths.")
    else:
        main()

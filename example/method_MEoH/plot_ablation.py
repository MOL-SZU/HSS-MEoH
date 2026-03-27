"""
两次实验的 test 结果帕累托曲线对比：传入两个结果根路径，在每个数据集上绘制 HV vs -Time，
只区分两个结果集（不再区分岛屿），使用鲜明颜色；同时包含基线算法
（GHSS、GAHSS、GSI_LS、DPP、SPESS、TPOSS）。
图片保存到 ./image/ablation_{算法1}_{算法2}/，数量等于测试集（结果子文件夹）数量。
"""

from __future__ import annotations

import os
import glob
import json
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# 基线算法名称（含 SPESS、TPOSS）
BASELINE_NAMES = ["GHSS", "GAHSS", "GSI_LS", "GL_HSS", "SPESS", "TPOSS"]


def _load_points_from_result_dir(
    result_dir: str,
    exp_name: str,
) -> List[Tuple[float, float, str]]:
    """
    从单个结果文件夹加载所有 code_*.json（不包含基线，基线由 baseline 目录统一加载）。
    返回 [(hv, -time, label), ...]。label 为结果集名称 exp_name。
    """
    out: List[Tuple[float, float, str]] = []
    for path in glob.glob(os.path.join(result_dir, "code_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("summary") if isinstance(data, dict) else None
        if not s or "avg_hv" not in s or "avg_time" not in s:
            continue
        hv = float(s["avg_hv"])
        t = float(s["avg_time"])
        if not (np.isfinite(hv) and np.isfinite(t)):
            print(f"[WARN] Skip non-finite point in {path}: avg_hv={hv}, avg_time={t}")
            continue
        out.append((hv, -t, exp_name))
    return out


def _load_baseline_points_from_dir(baseline_dir: str) -> List[Tuple[float, float, str]]:
    """
    从 baseline 目录加载六个基线的 JSON，返回 [(hv, -time, name), ...]。
    """
    out: List[Tuple[float, float, str]] = []
    for name in BASELINE_NAMES:
        path = os.path.join(baseline_dir, f"{name}.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("summary") if isinstance(data, dict) else None
        if not s or "avg_hv" not in s or "avg_time" not in s:
            continue
        hv = float(s["avg_hv"])
        t = float(s["avg_time"])
        if not (np.isfinite(hv) and np.isfinite(t)):
            print(f"[WARN] Skip non-finite baseline point in {path}: avg_hv={hv}, avg_time={t}")
            continue
        out.append((hv, -t, name))
    return out


def _normalize_points(points: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    finite_points = [p for p in points if np.isfinite(p[0]) and np.isfinite(p[1])]
    if not finite_points:
        return []
    if len(finite_points) != len(points):
        print(f"[WARN] Filtered {len(points) - len(finite_points)} non-finite points before normalization.")

    xs = [p[0] for p in finite_points]
    ys = [p[1] for p in finite_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y

    normalized: List[Tuple[float, float, str]] = []
    for x, y, label in finite_points:
        nx = 0.0 if span_x == 0 else (x - min_x) / span_x
        ny = 0.0 if span_y == 0 else (y - min_y) / span_y
        normalized.append((nx, ny, label))
    return normalized


def _pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    在最大化目标下提取帕累托前沿。
    """
    front: List[Tuple[float, float]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def _normalize_xy_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    finite_points = [p for p in points if np.isfinite(p[0]) and np.isfinite(p[1])]
    if not finite_points:
        return []
    xs = [p[0] for p in finite_points]
    ys = [p[1] for p in finite_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    return [
        (
            0.0 if span_x == 0 else (x - min_x) / span_x,
            0.0 if span_y == 0 else (y - min_y) / span_y,
        )
        for x, y in finite_points
    ]


def _compute_normalized_hv(
    pts1: List[Tuple[float, float, str]],
    pts2: List[Tuple[float, float, str]],
) -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float]]]:
    """
    对两个对比集合分别提取帕累托前沿，再用两者前沿的并集做归一化。
    参考点为归一化后两个目标最小值的 0.9 倍。
    """
    pf1 = _pareto_front([(p[0], p[1]) for p in pts1])
    pf2 = _pareto_front([(p[0], p[1]) for p in pts2])
    combined_pf = pf1 + pf2
    if not combined_pf:
        return None, None, None

    normalized_combined = _normalize_xy_points(combined_pf)
    split = len(pf1)
    normalized_pf1 = normalized_combined[:split]
    normalized_pf2 = normalized_combined[split:]

    try:
        import pygmo as pg
        neg_pf1 = [(-x, -y) for x, y in normalized_pf1]
        neg_pf2 = [(-x, -y) for x, y in normalized_pf2]
        neg_combined = neg_pf1 + neg_pf2
        if not neg_combined:
            return None, None, None

        xs = [p[0] for p in neg_combined]
        ys = [p[1] for p in neg_combined]
        print(xs, ys)
        reference_point = (max(xs) * 1.1, max(ys) * 1.1)

        hv1 = pg.hypervolume(np.array(neg_pf1, dtype=float)).compute(reference_point) if neg_pf1 else None
        hv2 = pg.hypervolume(np.array(neg_pf2, dtype=float)).compute(reference_point) if neg_pf2 else None
        return hv1, hv2, reference_point
    except Exception as exc:
        print(f"[WARN] HV compute failed: {exc}")
        return None, None, None


def plot_ablation_pareto(
    path1: str,
    path2: str,
    output_dir: Optional[str] = None,
    baseline_root: Optional[str] = "./test_result/baseline",
    use_normalized_plot: bool = False,
    show_hv_in_plot: bool = False,
) -> None:
    """
    绘制两次实验在不同数据集上的帕累托曲线对比图，并包含基线算法。
    两个结果集用鲜明颜色区分，不再区分同一结果集内的岛屿；基线从 baseline_root 按数据集子文件夹加载，颜色与图例与之前一致。

    参数：
        path1: 第一次实验的 test 结果根路径，如 ./test_result/island_search
        path2: 第二次实验的 test 结果根路径，如 ./test_result/island_search_MSEA
        output_dir: 图片保存根目录，默认 ./image/ablation_{name1}_{name2}
        baseline_root: 六个基线结果根路径，其下按数据集名建子文件夹，每子文件夹含 GHSS.json 等；None 表示不画基线
    """
    path1 = os.path.normpath(path1)
    path2 = os.path.normpath(path2)
    name1 = os.path.basename(path1)
    name2 = os.path.basename(path2)
    if output_dir is None:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(_script_dir, "image", f"ablation_{name1}_{name2}")
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if baseline_root is not None:
        baseline_root = os.path.normpath(baseline_root)

    sub1 = {d for d in os.listdir(path1) if os.path.isdir(os.path.join(path1, d))}
    sub2 = {d for d in os.listdir(path2) if os.path.isdir(os.path.join(path2, d))}
    common = sorted(sub1 & sub2)
    if not common:
        print(f"[WARN] 两个路径下无共同子文件夹。path1={path1}, path2={path2}")
        return

    # 两个结果集：鲜明颜色（蓝、橙红）
    color_result1 = "#0066CC"   # 鲜明蓝
    color_result2 = "#E63946"   # 鲜明红
    # 基线：各算法不同颜色与标记（与之前一致）
    color_baseline = {
        "GHSS": "#2ca02c",
        "GAHSS": "#9467bd",
        "GSI_LS": "#8c564b",
        "GL_HSS": "#e377c2",
        "SPESS": "#17becf",
        "TPOSS": "#bcbd22",
    }
    marker_baseline = {
        "GHSS": "D",
        "GAHSS": "v",
        "GSI_LS": "p",
        "GL_HSS": "*",
        "SPESS": "s",
        "TPOSS": "^",
    }

    for dataset in common:
        dir1 = os.path.join(path1, dataset)
        dir2 = os.path.join(path2, dataset)
        pts1 = _load_points_from_result_dir(dir1, name1)
        pts2 = _load_points_from_result_dir(dir2, name2)
        all_pts: List[Tuple[float, float, str]] = list(pts1)
        for t in pts2:
            all_pts.append(t)
        # 从 baseline 目录加载六个基线：先尝试 baseline/<dataset>/，否则用 baseline/ 根目录（同一组基线用于所有数据集）
        if baseline_root:
            baseline_dir = os.path.join(baseline_root, dataset)
            if not os.path.isdir(baseline_dir):
                baseline_dir = baseline_root
            if os.path.isdir(baseline_dir):
                baseline_pts = _load_baseline_points_from_dir(baseline_dir)
                all_pts.extend(baseline_pts)

        if not all_pts:
            print(f"[WARN] 无数据，跳过: {dataset}")
            continue

        hv1, hv2, hv_ref = _compute_normalized_hv(pts1, pts2)
        print(f"[INFO] {dataset} HV (normalized PF) | {name1}: {hv1}, {name2}: {hv2}, ref={hv_ref}")
        plot_pts = _normalize_points(all_pts) if use_normalized_plot else all_pts
        fig, ax = plt.subplots(figsize=(10, 8))
        plotted = set()
        for hv, nt, label in plot_pts:
            if label in plotted:
                continue
            plotted.add(label)
            group = [(p[0], p[1]) for p in plot_pts if p[2] == label]
            xs, ys = [p[0] for p in group], [p[1] for p in group]
            if label == name1:
                c = color_result1
                marker = "o"
            elif label == name2:
                c = color_result2
                marker = "o"
            else:
                c = color_baseline.get(label, "#333")
                marker = marker_baseline.get(label, "x")
            ax.scatter(
                xs, ys,
                label=label,
                color=c,
                marker=marker,
                s=70,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.5,
            )

        if use_normalized_plot:
            ax.set_xlabel("Normalized Objective 1 (Hypervolume)", fontsize=12)
            ax.set_ylabel("Normalized Objective 2 (-Average Time)", fontsize=12)
            ax.set_title(f"Pareto comparison (normalized): {dataset}", fontsize=14)
            # ax.set_xlim(-0.05, 1.05)
            # ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
            ax.set_yticks(np.linspace(0.0, 1.0, 6))
        else:
            ax.set_xlabel("Objective 1 (Hypervolume)", fontsize=12)
            ax.set_ylabel("Objective 2 (-Average Time)", fontsize=12)
            ax.set_title(f"Pareto comparison: {dataset}", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        if show_hv_in_plot:
            hv_text = f"HV({name1})={hv1}\nHV({name2})={hv2}\nref={hv_ref}"
            ax.text(
                0.02,
                0.98,
                hv_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#888"),
            )
        plt.tight_layout()
        suffix = "_normalized" if use_normalized_plot else ""
        save_path = os.path.join(output_dir, f"{dataset}{suffix}.png")
        print(f"[DEBUG] {dataset} axis limits before save: x={ax.get_xlim()}, y={ax.get_ylim()}")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] 已保存: {save_path}")

    print(f"[INFO] 共绘制 {len(common)} 张图，保存至: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="两次实验帕累托对比（不区分岛屿）+ 基线（从 baseline 目录加载）。")
    parser.add_argument("path1", type=str, nargs="?", default="./test_result/LLM4AD_initial_dominate_150", help="第一次实验结果根路径")
    parser.add_argument("path2", type=str, nargs="?", default="./test_result/LLM4AD_initial_uniform_150", help="第二次实验结果根路径")
    parser.add_argument("--output_dir", type=str, default=None, help="图片保存目录")
    parser.add_argument("--baseline_root", type=str, default="./test_result/baseline_wo_GA", help="六个基线结果根路径，其下按数据集名建子文件夹")
    parser.add_argument("--use_normalized_plot", action="store_true", default=True,  help="Use normalized data for plotting, including baseline points.")
    parser.add_argument("--show_hv_in_plot", action="store_true", default=True, help="Show normalized-PF HV values in each figure.")
    args = parser.parse_args()

    plot_ablation_pareto(
        path1=args.path1,
        path2=args.path2,
        output_dir=args.output_dir,
        baseline_root=args.baseline_root,
        use_normalized_plot=args.use_normalized_plot,
        show_hv_in_plot=args.show_hv_in_plot,
    )

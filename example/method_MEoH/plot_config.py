from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PlotStyleConfig:
    figsize: Tuple[float, float] = (10.0, 10.0)
    label_fontsize: int = 24
    title_fontsize: int = 28
    tick_fontsize: int = 24
    legend_fontsize: int = 24
    inset_tick_fontsize: int = 24


DEFAULT_PLOT_STYLE = PlotStyleConfig()


def add_plot_style_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=list(DEFAULT_PLOT_STYLE.figsize),
        help="Shared figure size: width height",
    )
    parser.add_argument(
        "--label_fontsize",
        type=int,
        default=DEFAULT_PLOT_STYLE.label_fontsize,
        help="Font size for axis labels",
    )
    parser.add_argument(
        "--title_fontsize",
        type=int,
        default=DEFAULT_PLOT_STYLE.title_fontsize,
        help="Font size for titles",
    )
    parser.add_argument(
        "--tick_fontsize",
        type=int,
        default=DEFAULT_PLOT_STYLE.tick_fontsize,
        help="Font size for axis ticks",
    )
    parser.add_argument(
        "--legend_fontsize",
        type=int,
        default=DEFAULT_PLOT_STYLE.legend_fontsize,
        help="Font size for legends",
    )
    parser.add_argument(
        "--inset_tick_fontsize",
        type=int,
        default=DEFAULT_PLOT_STYLE.inset_tick_fontsize,
        help="Font size for inset ticks",
    )


def build_plot_style_config(args: argparse.Namespace) -> PlotStyleConfig:
    return PlotStyleConfig(
        figsize=tuple(args.figsize),
        label_fontsize=args.label_fontsize,
        title_fontsize=args.title_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        inset_tick_fontsize=args.inset_tick_fontsize,
    )

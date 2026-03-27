from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonPreset:
    log_dir: str
    compare_log_dir: str
    log_label: str
    compare_label: str
    log_style_key: str = "MEoH"
    compare_style_key: str = "MEoH-HS"
    suffix: str = ""
    pareto_save_path: str = ""


DEFAULT_COMPARISON_PRESETS = [
    ComparisonPreset(
        log_dir="logs/meoh/exp_ours",
        compare_log_dir="logs/meoh/exp_raw",
        log_label="ours",
        compare_label="MEoH",
        suffix="ours_raw",
        pareto_save_path="image/pareto_ours_raw.png",
    ),
    ComparisonPreset(
        log_dir="logs/meoh/exp_ours",
        compare_log_dir="logs/meoh/exp_FR",
        log_label="ours",
        compare_label="w/o RSM",
        suffix="wo_FR",
        pareto_save_path="image/pareto_wo_FR.png",
    ),
    ComparisonPreset(
        log_dir="logs/meoh/exp_ours",
        compare_log_dir="logs/meoh/exp_initial",
        log_label="ours",
        compare_label="w/o DWS-Init",
        suffix="wo_initial",
        pareto_save_path="image/pareto_wo_initial.png",
    ),
    ComparisonPreset(
        log_dir="logs/meoh/exp_ours",
        compare_log_dir="logs/meoh/exp_power_law",
        log_label="ours",
        compare_label="w/o NWPS",
        suffix="wo_weighted",
        pareto_save_path="image/pareto_wo_weighted.png",
    ),
]

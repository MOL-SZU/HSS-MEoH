from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from paths import IMAGES_DIR, LOGS_DIR


@dataclass(frozen=True)
class ComparisonPreset:
    log_dir: str
    compare_log_dir: str
    log_label: str
    compare_label: str
    log_style_key: str = 'MEoH'
    compare_style_key: str = 'MEoH-HS'
    suffix: str = ''
    pareto_save_path: str = ''


DEFAULT_COMPARISON_PRESETS = [
    ComparisonPreset(str(LOGS_DIR / 'meoh' / 'exp_ours'), str(LOGS_DIR / 'meoh' / 'exp_raw'), 'ours', 'MEoH', suffix='ours_raw', pareto_save_path=str(IMAGES_DIR / 'pareto_ours_raw.png')),
    ComparisonPreset(str(LOGS_DIR / 'meoh' / 'exp_ours'), str(LOGS_DIR / 'meoh' / 'exp_FR'), 'ours', 'w/o RSM', suffix='wo_FR', pareto_save_path=str(IMAGES_DIR / 'pareto_wo_FR.png')),
    ComparisonPreset(str(LOGS_DIR / 'meoh' / 'exp_ours'), str(LOGS_DIR / 'meoh' / 'exp_initial'), 'ours', 'w/o DWS-Init', suffix='wo_initial', pareto_save_path=str(IMAGES_DIR / 'pareto_wo_initial.png')),
    ComparisonPreset(str(LOGS_DIR / 'meoh' / 'exp_ours'), str(LOGS_DIR / 'meoh' / 'exp_power_law'), 'ours', 'w/o NWPS', suffix='wo_weighted', pareto_save_path=str(IMAGES_DIR / 'pareto_wo_weighted.png')),
]

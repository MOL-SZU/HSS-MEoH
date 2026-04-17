from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = WORKSPACE_DIR.parent
for path in (WORKSPACE_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paths import LOGS_DIR, RESULT_CODE_DIR, ensure_result_dirs


def find_latest_elitist_file(log_dir: str | os.PathLike[str]) -> Optional[str]:
    elitist_dir = os.path.join(log_dir, 'elitist')
    if not os.path.isdir(elitist_dir):
        return None
    files = glob.glob(os.path.join(elitist_dir, 'elitist_*.json'))
    if not files:
        return None
    files.sort(key=lambda path: int(Path(path).stem.split('_')[-1]), reverse=True)
    return files[0]


def extract_elitist_code(log_dir: str, output_root: str = str(RESULT_CODE_DIR), encoding: str = 'utf-8') -> None:
    ensure_result_dirs()
    latest_elitist = find_latest_elitist_file(log_dir)
    if latest_elitist is None:
        print(f"[WARN] No elitist_*.json found under '{log_dir}'.")
        return
    with open(latest_elitist, 'r', encoding=encoding) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f'[ERROR] Invalid elitist file format: {latest_elitist}')
        return
    exp_name = os.path.basename(os.path.normpath(log_dir))
    out_dir = os.path.join(output_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        code = item.get('function')
        if not code:
            continue
        out_path = os.path.join(out_dir, f'code_{idx:03d}.py')
        with open(out_path, 'w', encoding=encoding) as fw:
            fw.write('import numpy as np\n\n')
            fw.write(str(code))
        count += 1
    print(f'[INFO] Extracted {count} functions into: {out_dir}')


if __name__ == '__main__':
    extract_elitist_code(str(LOGS_DIR / 'meoh' / 'exp_raw'), output_root=str(RESULT_CODE_DIR))

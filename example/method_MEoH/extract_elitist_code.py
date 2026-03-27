"""
从指定结果日志文件夹中提取最新 elitist 文件里的所有函数代码，
并保存到 result_code 的对应结果子文件夹中。

用法示例（在项目根目录下 / 或在 PyCharm 中直接运行本文件）：
    1) 修改文件底部的 log_dir 变量，例如：
       log_dir = 'logs/meoh/GSI_LS_gpt_200'
    2) 直接运行本文件

将会把最新的 elitist_*.json 中的所有 function 字段保存到：
    result_code/GSI_LS_gpt_200/code_000.py, code_001.py, ...
"""

import os
import json
import glob
from typing import Optional


def find_latest_elitist_file(log_dir: str) -> Optional[str]:
    """
    在指定 log_dir 下的 elitist 子目录中找到最新的 elitist_*.json 文件。
    """
    elitist_dir = os.path.join(log_dir, "elitist")
    if not os.path.isdir(elitist_dir):
        return None

    pattern = os.path.join(elitist_dir, "elitist_*.json")
    files = glob.glob(pattern)
    if not files:
        return None

    def get_generation(path: str) -> int:
        name = os.path.basename(path)
        try:
            return int(name.replace("elitist_", "").replace(".json", ""))
        except Exception:
            return -1

    files.sort(key=get_generation, reverse=True)
    return files[0]


def extract_elitist_code(
    log_dir: str,
    output_root: str = "result_code",
    encoding: str = "utf-8",
) -> None:
    """
    从 log_dir 中最新的 elitist_*.json 里提取所有函数代码，
    写入到 output_root/exp_name/ 目录下的多个 .py 文件中。

    - exp_name 取自 log_dir 的最后一级目录名
    - 文件命名规则：code_{index:03d}.py
    """
    latest_elitist = find_latest_elitist_file(log_dir)
    if latest_elitist is None:
        print(f"[WARN] 未在 '{log_dir}' 下找到 elitist_*.json")
        return

    print(f"[INFO] 使用 elitist 文件: {latest_elitist}")

    with open(latest_elitist, "r", encoding=encoding) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[ERROR] elitist 文件格式异常（应为 list）：{latest_elitist}")
        return

    exp_name = os.path.basename(os.path.normpath(log_dir))
    out_dir = os.path.join(output_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        code = item.get("function")
        if not code:
            continue

        filename = f"code_{idx:03d}.py"
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding=encoding) as fw:
            fw.write('import numpy as np\n\n')
            fw.write(str(code))

        count += 1

    print(f"[INFO] 共提取并保存 {count} 个函数到: {out_dir}")


if __name__ == "__main__":
    # 在这里修改为你想要处理的日志目录（相对于项目根目录 or 绝对路径）
    # 例如：log_dir = 'logs/meoh/GSI_LS_gpt_200'
    log_dir = 'logs/meoh/exp_raw'

    # 代码输出根目录（相对于当前工作目录）
    output_root = './result_code'

    extract_elitist_code(log_dir, output_root=output_root)



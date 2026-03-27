"""
批量评测 HSS 系列算法（DPP、GHSS、GAHSS、GSI_LS）：
- 对 data/train_data 目录下的所有 .mat 文件运行
- 计算每个文件的超体积 HV 与时间
- 对每个算法取平均结果
- 将结果写入 train_result/{algo}.json
"""

import os
import json
import time
import glob
from typing import Dict, List

import numpy as np
from mat2array import load_mat_to_numpy
import pygmo as pg

# 导入算法接口函数
from HV_cal.GL_HSS import HSS as GL_HSS
from HV_cal.GHSS import HSS as GHSS
from HV_cal.GAHSS import HSS as GAHSS
from HV_cal.GSI_LS import HSS as GSI_LS
from HV_cal.TPOSS import HSS as TPOSS
from HV_cal.SPESS import HSS as SPESS



def hv_cal(selected_objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """计算超体积"""
    hv = pg.hypervolume(selected_objectives)
    return hv.compute(reference_point)


def eval_on_file(row_data: np.ndarray, algo_fn, k: int, ref: float, num_runs: int = 3) -> Dict[str, float]:
    """
    单文件评测：返回 hv 和 time（这里负责计时，algo_fn 只返回 subset）
    由于算法具有随机性，运行num_runs次并取平均值
    
    参数:
        row_data: 输入数据
        algo_fn: 算法函数
        k: 选取点数
        ref: 参考点放大系数
        num_runs: 运行次数，默认3次
    """
    reference_point = np.max(row_data, axis=0) * ref
    
    hv_list = []
    time_list = []
    
    # 运行num_runs次，取平均值
    for run_idx in range(num_runs):
        start = time.time()
        subset = algo_fn(row_data, k, reference_point)
        t = time.time() - start
        hv = hv_cal(subset, reference_point)
        hv_list.append(float(hv))
        time_list.append(float(t))
    
    # 计算平均值
    avg_hv = float(np.mean(hv_list))
    avg_time = float(np.mean(time_list))
    
    return {
        "hv": avg_hv,
        "time": avg_time,
        "runs": {
            "hv_list": hv_list,
            "time_list": time_list,
            "num_runs": num_runs
        }
    }


def run_benchmark(
    data_folder: str = "../data/train_data",
    output_folder: str = "../example/MeoH_HSS/train_result",
    k: int = 8,
    ref: float = 1.1,
):
    os.makedirs(output_folder, exist_ok=True)

    # 收集数据文件
    mat_files = sorted(glob.glob(os.path.join(data_folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"未找到数据文件：{data_folder}/*.mat")

    # 算法列表
    algos = {
        "GL_HSS": GL_HSS,
        "GHSS": GHSS,
        "GAHSS": GAHSS,
        "GSI_LS": GSI_LS,
        "TPOSS": TPOSS,
        "SPESS": SPESS,
    }

    for algo_name, algo_fn in algos.items():
        details: List[Dict[str, float]] = []
        hv_list, time_list = [], []

        for fpath in mat_files:
            try:
                row_data = load_mat_to_numpy(fpath, "points")
                if row_data is None:
                    continue
                # 与示例保持一致，最多取前 1000 行
                row_data = row_data[:1000, :]
                res = eval_on_file(row_data, algo_fn, k=k, ref=ref)
                res["file"] = os.path.basename(fpath)
                details.append(res)
                hv_list.append(res["hv"])
                time_list.append(res["time"])
            except Exception as e:
                # 失败时记录错误并跳过
                details.append({"file": os.path.basename(fpath), "error": str(e)})
                continue

        if hv_list:
            avg_hv = float(np.mean(hv_list))
            avg_time = float(np.mean(time_list))
        else:
            avg_hv = float("-inf")
            avg_time = float("inf")

        result = {
            "summary": {
                "avg_hv": avg_hv,
                "avg_time": avg_time,
                "num_files": len(hv_list),
            },
            "details": details,
        }

        out_path = os.path.join(output_folder, f"{algo_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[{algo_name}] 写入结果: {out_path}")


if __name__ == "__main__":
    data_folder: str = "../data/train_data"
    output_folder: str = "../example/method_MEoH/train_result"
    k: int = 8
    ref: float = 1.1
    run_benchmark(data_folder, output_folder, k, ref)


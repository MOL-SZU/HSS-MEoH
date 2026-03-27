"""
在指定的 result_code 子文件夹中，批量测试所有函数（code_xxx.py）
在 test_data 下的9个测试文件夹中所有数据集上的平均表现，并保存为 JSON：

- 输入：result_code/<exp_name>/code_000.py, code_001.py, ...
- 测试数据：test_data/<test_folder>/*.mat （变量名默认 'points'）
- k参数：20, 50, 100
- 输出：./test_result/<exp_name>/<test_folder>_<k>/code_000.json, code_001.json, ...（每个函数对应一个 JSON 文件）
- 总共会生成27个结果文件夹（9个测试集 × 3个k值），全部放在以输入目录命名的子文件夹下

使用方式（在 PyCharm 里直接运行本文件）：
    1. 修改文件底部的 code_folder 和 test_data_root 变量
    2. 运行本文件
"""

import os
import json
import glob
import time
import importlib.util
from typing import Callable, Dict, List, Optional

import numpy as np
import pygmo as pg
from tqdm import tqdm
from mat2array import load_mat_to_numpy


def hv_cal(selected_objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """计算超体积"""
    hv = pg.hypervolume(selected_objectives)
    return hv.compute(reference_point)


def load_function_from_file(file_path: str) -> Optional[Callable]:
    """
    从 code_XXX.py 文件中加载候选函数。
    优先寻找名为 'HSS' 的函数，否则返回第一个普通 callable。
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    # 优先用 HSS
    if hasattr(module, "HSS") and callable(getattr(module, "HSS")):
        return getattr(module, "HSS")

    # 否则取第一个非私有 callable
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj):
            return obj
    return None


def evaluate_one_function(
    func: Callable,
    k: int,
    ref: float,
    test_data_dir: str,
    mat_key: str = "points",
    desc: str = None,
    max_points: Optional[int] = None,
    num_runs: int = 10,
) -> Dict:
    """
    在 test_data_dir 中所有 mat 文件上测试一个函数，返回平均结果与明细。
    函数签名假定为：func(points, k, reference_point) -> np.ndarray

    参数:
        func: 要测试的函数
        k: 选取点数
        ref: 参考点放大系数
        test_data_dir: 测试数据目录
        mat_key: mat文件中的变量名
        desc: tqdm进度条描述信息
        max_points: 若给定，则每个 mat 只取前 max_points 行（例如 200 表示只取前 200 个点）
        num_runs: 每个文件上运行次数，取平均，默认 3
    """
    mat_files = sorted(glob.glob(os.path.join(test_data_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"未在 '{test_data_dir}' 中找到 mat 文件")

    details: List[Dict] = []
    hv_list: List[float] = []
    time_list: List[float] = []

    # 使用tqdm显示处理mat文件的进度
    progress_desc = desc if desc else "处理中"
    with tqdm(total=len(mat_files), desc=progress_desc, unit="file", leave=False) as pbar:
        for fpath in mat_files:
            fname = os.path.basename(fpath)
            pbar.set_postfix({"file": fname})

            try:
                points = load_mat_to_numpy(fpath, mat_key)
                if points is None:
                    details.append({"file": fname, "error": "load_mat_to_numpy 返回 None"})
                    pbar.update(1)
                    continue

                # 确保是二维
                points = np.asarray(points)
                if points.ndim != 2:
                    details.append({"file": fname, "error": f"points 维度={points.ndim} 非 2"})
                    pbar.update(1)
                    continue

                if max_points is not None and points.shape[0] > max_points:
                    points = points[:max_points]

                reference_point = np.max(points, axis=0) * ref

                # 由于算法具有随机性，运行 num_runs 次并取平均值
                hv_runs = []
                time_runs = []
                
                for run_idx in range(num_runs):
                    start = time.time()
                    subset = func(points, k, reference_point)
                    elapsed = time.time() - start

                    subset = np.asarray(subset)
                    if subset.ndim != 2:
                        details.append({"file": fname, "error": f"subset 维度={subset.ndim} 非 2"})
                        break  # 跳出循环，不继续运行

                    hv = hv_cal(subset, reference_point)
                    hv_runs.append(float(hv))
                    time_runs.append(float(elapsed))
                
                # 如果所有运行都成功，计算平均值
                if len(hv_runs) == num_runs:
                    avg_hv = float(np.mean(hv_runs))
                    avg_time = float(np.mean(time_runs))
                    
                    details.append(
                        {
                            "file": fname,
                            "hv": avg_hv,
                            "time": avg_time,
                            "runs": {
                                "hv_list": hv_runs,
                                "time_list": time_runs,
                                "num_runs": num_runs
                            }
                        }
                    )
                    hv_list.append(avg_hv)
                    time_list.append(avg_time)
            except Exception as e:
                details.append({"file": fname, "error": str(e)})
            finally:
                pbar.update(1)

    if hv_list:
        avg_hv = float(np.mean(hv_list))
        avg_time = float(np.mean(time_list))
    else:
        avg_hv = float("-inf")
        avg_time = float("inf")

    return {
        "summary": {
            "avg_hv": avg_hv,
            "avg_time": avg_time,
            "num_files": len(hv_list),
        },
        "details": details,
    }


def evaluate_result_code_folder_with_config(
    code_folder: str,
    test_data_root: str,
    output_root: str,
    test_folders: List[str],
    k_list: List[int],
    ref: float = 1.1,
    max_points: Optional[int] = None,
    num_runs: int = 10,
) -> None:
    """
    对指定 code_folder 中的 code_*.py 在选定的测试文件夹和 k 上评测，支持 max_points、num_runs。
    用于“汇总 elitist 后”的同一套接口：可只测部分 test_folders、部分 k、限制点数与重复次数。
    """
    if not os.path.isdir(code_folder):
        raise FileNotFoundError(f"code_folder 不存在: {code_folder}")
    if not os.path.isdir(test_data_root):
        raise FileNotFoundError(f"test_data_root 不存在: {test_data_root}")

    input_folder_name = os.path.basename(code_folder)
    output_root_with_input = os.path.join(output_root, input_folder_name)
    os.makedirs(output_root_with_input, exist_ok=True)

    py_files = sorted(glob.glob(os.path.join(code_folder, "code_*.py")))
    if not py_files:
        print(f"[WARN] 在 '{code_folder}' 下未找到 code_*.py")
        return

    for test_folder in test_folders:
        test_data_dir = os.path.join(test_data_root, test_folder)
        if not os.path.isdir(test_data_dir):
            print(f"[WARN] 测试数据目录不存在: {test_data_dir}，跳过。")
            continue
        for k in k_list:
            out_dir = os.path.join(output_root_with_input, f"{test_folder}_{k}")
            os.makedirs(out_dir, exist_ok=True)
            for fpath in py_files:
                fname = os.path.basename(fpath)
                json_path = os.path.join(out_dir, 'elitist_', os.path.splitext(fname)[0] + ".json")
                if os.path.exists(json_path):
                    continue
                func = load_function_from_file(fpath)
                if func is None:
                    continue
                try:
                    res = evaluate_one_function(
                        func=func,
                        k=k,
                        ref=ref,
                        test_data_dir=test_data_dir,
                        desc=f"{fname} (k={k})",
                        max_points=max_points,
                        num_runs=num_runs,
                    )
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[ERROR] 评测 '{fname}' 时出错: {e}")


def evaluate_result_code_folder(
    code_folder: str,
    test_data_root: str = "../../data/test_data",
    output_root: str = "./test_result",
    k_list: List[int] = [5, 10, 15],
    ref: float = 1.1,
) -> None:
    """
    对指定 result_code 子文件夹中的所有 code_XXX.py 进行评测，
    在test_data_root下的所有测试文件夹中测试，对每个k值分别测试。
    结果保存为：./test_result/<input_folder_name>/<test_folder>_<k>/code_XXX.json

    参数:
        code_folder  : 包含 code_XXX.py 的文件夹，例如 'result_code/GSI_LS_gpt_200'
        test_data_root: 测试数据根目录，例如 '../../data/test_data'，该目录下应该有多个子文件夹
        output_root  : 输出根目录，默认 './test_result'
        k_list       : k值列表，默认 [20, 50, 100]
        ref          : 参考点放大系数
    """
    if not os.path.isdir(code_folder):
        raise FileNotFoundError(f"code_folder 不存在: {code_folder}")

    if not os.path.isdir(test_data_root):
        raise FileNotFoundError(f"test_data_root 不存在: {test_data_root}")

    # 从输入目录中提取文件夹名字作为输出目录名
    input_folder_name = os.path.basename(code_folder)
    output_root_with_input = os.path.join(output_root, input_folder_name)
    os.makedirs(output_root_with_input, exist_ok=True)

    # 获取所有测试文件夹
    test_folders = [d for d in os.listdir(test_data_root)
                    if os.path.isdir(os.path.join(test_data_root, d))]
    test_folders = sorted(test_folders)

    if not test_folders:
        raise FileNotFoundError(f"在 '{test_data_root}' 下未找到任何测试文件夹")

    print(f"[INFO] 找到 {len(test_folders)} 个测试文件夹: {test_folders}")

    # 获取所有代码文件
    py_files = sorted(glob.glob(os.path.join(code_folder, "code_*.py")))
    if not py_files:
        print(f"[WARN] 在 '{code_folder}' 下未找到 code_*.py")
        return

    print(f"[INFO] 找到 {len(py_files)} 个代码文件")
    print(f"[INFO] 输出目录: {output_root_with_input}")
    print(f"[INFO] 将进行 {len(test_folders)} × {len(k_list)} = {len(test_folders) * len(k_list)} 组测试")

    total_success = 0
    total_skip = 0
    total_existing = 0

    # 遍历每个测试文件夹和每个k值
    for test_folder in test_folders:
        test_data_dir = os.path.join(test_data_root, test_folder)
        
        for k in k_list:
            # 创建输出目录：{input_folder_name}/{test_folder}_{k}
            output_folder_name = f"{test_folder}_{k}"
            out_dir = os.path.join(output_root_with_input, output_folder_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n[INFO] ===== 开始测试: {test_folder}, k={k} =====")
            print(f"[INFO] 测试数据目录: {test_data_dir}")
            print(f"[INFO] 输出目录: {out_dir}")

            success_count = 0
            skip_count = 0
            existing_count = 0

            # 遍历所有算法文件
            for fpath in py_files:
                fname = os.path.basename(fpath)
                
                # 检查JSON文件是否已存在
                json_name = os.path.splitext(fname)[0] + ".json"  # code_000.py -> code_000.json
                json_path = os.path.join(out_dir, json_name)
                
                if os.path.exists(json_path):
                    # 文件已存在，跳过计算
                    print(f"[INFO] {fname} 的结果已存在，跳过计算")
                    existing_count += 1
                    continue
                
                print(f"[INFO] 开始计算 {fname} (k={k}, test_folder={test_folder})")
                
                func = load_function_from_file(fpath)
                if func is None:
                    print(f"[WARN] 在 '{fname}' 中未找到可用函数，跳过。")
                    skip_count += 1
                    continue

                try:
                    # 调用evaluate_one_function，内部会显示处理mat文件的进度条
                    res = evaluate_one_function(
                        func=func,
                        k=k,
                        ref=ref,
                        test_data_dir=test_data_dir,
                        desc=f"{fname} (k={k})",
                    )

                    # 保存JSON文件
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False, indent=2)
                    
                    print(f"[INFO] {fname} 计算完成，结果已保存至: {json_path}")
                    success_count += 1
                except Exception as e:
                    print(f"[ERROR] 评测 '{fname}' 时出错: {e}")
                    skip_count += 1
                    continue

            print(f"[INFO] {test_folder} (k={k}) 完成: 成功 {success_count} 个，跳过 {skip_count} 个，已存在 {existing_count} 个")
            total_success += success_count
            total_skip += skip_count
            total_existing += existing_count

    print(f"\n[INFO] ===== 所有测试完成 =====")
    print(f"[INFO] 总计: 成功 {total_success} 个，跳过 {total_skip} 个，已存在 {total_existing} 个")
    print(f"[INFO] 所有结果保存在: {output_root_with_input}")
    print(f"[INFO] 共生成 {len(test_folders) * len(k_list)} 个结果文件夹")


if __name__ == "__main__":
    # 在这里修改要评测的 result_code 子文件夹
    # 例如：code_folder = 'result_code/GSI_LS_gpt_200'
    code_folder = "./result_code/exp_raw"

    # 测试数据根目录（该目录下应该有9个测试文件夹）
    test_data_root = "../../data/test_data"

    # 输出根目录
    output_root = "./test_result"

    # k值列表
    k_list = [5,10,15]

    evaluate_result_code_folder(
        code_folder=code_folder,
        test_data_root=test_data_root,
        output_root=output_root,
        k_list=k_list,
        ref=1.1,
    )



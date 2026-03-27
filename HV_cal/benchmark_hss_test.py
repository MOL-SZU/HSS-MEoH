"""
批量评测 HSS 系列算法（DPP、GHSS、GAHSS、GSI_LS）在测试数据上：
- 对 data/test_data 目录下的9个测试文件夹分别运行
- 对每个测试文件夹使用 k=[20, 50, 100] 分别测试
- 计算每个文件的超体积 HV 与时间
- 对每个算法取平均结果
- 将结果写入 test_result/<test_folder>_<k>/{algo}.json
- 总共生成27个结果文件夹（9个测试集 × 3个k值）

新功能：
- 支持在指定的文件夹路径下的每个子文件夹中运行四个算法的测试
- 从子文件夹名称中提取test_folder和k值（格式：{test_folder}_{k}）
"""

import os
import json
import time
import glob
from typing import Dict, List

import numpy as np
from mat2array import load_mat_to_numpy
import pygmo as pg
from tqdm import tqdm

# 导入算法接口函数
from HV_cal.DPP import HSS as DPP
from HV_cal.GHSS import HSS as GHSS
from HV_cal.GAHSS import HSS as GAHSS
from HV_cal.GSI_LS import HSS as GSI_LS


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


def run_benchmark_on_folder(
    data_folder: str,
    algo_name: str,
    algo_fn,
    k: int,
    ref: float = 1.1,
) -> Dict:
    """
    在指定数据文件夹上运行一个算法，返回结果。
    
    参数:
        data_folder: 数据文件夹路径
        algo_name: 算法名称
        algo_fn: 算法函数
        k: 选取点数
        ref: 参考点放大系数
    
    返回:
        包含summary和details的字典
    """
    # 收集数据文件
    mat_files = sorted(glob.glob(os.path.join(data_folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"未找到数据文件：{data_folder}/*.mat")

    details: List[Dict[str, float]] = []
    hv_list, time_list = [], []

    # 使用tqdm显示进度
    with tqdm(total=len(mat_files), desc=f"{algo_name} (k={k})", unit="file", leave=False) as pbar:
        for fpath in mat_files:
            fname = os.path.basename(fpath)
            pbar.set_postfix({"file": fname})
            
            try:
                row_data = load_mat_to_numpy(fpath, "points")
                if row_data is None:
                    pbar.update(1)
                    continue
                
                res = eval_on_file(row_data, algo_fn, k=k, ref=ref)
                res["file"] = fname
                details.append(res)
                hv_list.append(res["hv"])
                time_list.append(res["time"])
            except Exception as e:
                # 失败时记录错误并跳过
                details.append({"file": fname, "error": str(e)})
            finally:
                pbar.update(1)

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
    
    return result


def run_benchmark(
    test_data_root: str = "../data/test_data",
    output_root: str = "../example/MEoH_HSS/test_result",
    k_list: List[int] = [20, 50, 100],
    ref: float = 1.1,
):
    """
    在test_data_root下的所有测试文件夹中运行benchmark算法。
    
    参数:
        test_data_root: 测试数据根目录，该目录下应该有多个测试文件夹
        output_root: 输出根目录，默认 '../example/MEoH_HSS/test_result'
        k_list: k值列表，默认 [20, 50, 100]
        ref: 参考点放大系数
    """
    if not os.path.isdir(test_data_root):
        raise FileNotFoundError(f"test_data_root 不存在: {test_data_root}")

    # 获取所有测试文件夹
    test_folders = [d for d in os.listdir(test_data_root) 
                    if os.path.isdir(os.path.join(test_data_root, d))]
    test_folders = sorted(test_folders)
    
    if not test_folders:
        raise FileNotFoundError(f"在 '{test_data_root}' 下未找到任何测试文件夹")
    
    print(f"[INFO] 找到 {len(test_folders)} 个测试文件夹: {test_folders}")
    print(f"[INFO] 将进行 {len(test_folders)} × {len(k_list)} = {len(test_folders) * len(k_list)} 组测试")

    # 算法列表
    algos = {
        "DPP": DPP,
        "GHSS": GHSS,
        "GAHSS": GAHSS,
        # "GSI_LS": GSI_LS,
    }

    total_combinations = len(test_folders) * len(k_list) * len(algos)
    current_combination = 0

    # 遍历每个测试文件夹和每个k值
    for test_folder in test_folders:
        test_data_dir = os.path.join(test_data_root, test_folder)
        
        for k in k_list:
            # 创建输出目录：{test_folder}_{k}
            output_folder_name = f"{test_folder}_{k}"
            out_dir = os.path.join(output_root, output_folder_name)
            os.makedirs(out_dir, exist_ok=True)
            
            print(f"\n[INFO] ===== 开始测试: {test_folder}, k={k} =====")
            print(f"[INFO] 测试数据目录: {test_data_dir}")
            print(f"[INFO] 输出目录: {out_dir}")

            # 对每个算法运行测试
            for algo_name, algo_fn in algos.items():
                current_combination += 1
                print(f"[INFO] [{current_combination}/{total_combinations}] 运行算法: {algo_name}")
                
                # 检查结果文件是否已存在
                out_path = os.path.join(out_dir, f"{algo_name}.json")
                if os.path.exists(out_path):
                    print(f"[INFO] {algo_name} 的结果已存在，跳过计算")
                    continue

                try:
                    result = run_benchmark_on_folder(
                        data_folder=test_data_dir,
                        algo_name=algo_name,
                        algo_fn=algo_fn,
                        k=k,
                        ref=ref,
                    )

                    # 保存结果
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"[INFO] {algo_name} 结果已保存至: {out_path}")
                except Exception as e:
                    print(f"[ERROR] 运行 {algo_name} 时出错: {e}")
                    continue

            print(f"[INFO] {test_folder} (k={k}) 完成")

    print(f"\n[INFO] ===== 所有测试完成 =====")
    print(f"[INFO] 所有结果保存在: {output_root}")
    print(f"[INFO] 共生成 {len(test_folders) * len(k_list)} 个结果文件夹")


def run_benchmark_on_result_folder(
    result_folder_path: str,
    test_data_root: str = "../data/test_data",
    ref: float = 1.1,
):
    """
    在指定的结果文件夹路径下的每个子文件夹中运行四个算法的测试。
    
    参数:
        result_folder_path: 指定的结果文件夹路径，例如 '../example/MEoH_HSS/test_result/mul_gpt_150'
        test_data_root: 测试数据根目录，默认 '../data/test_data'
        ref: 参考点放大系数
    
    说明:
        - 在result_folder_path下找到所有子文件夹（例如 points_1000_3_100_20, points_2000_3_100_20 等）
        - 从子文件夹名称中提取test_folder和k值（格式：{test_folder}_{k}）
        - 对每个子文件夹，运行四个算法的测试，将结果保存到对应的子文件夹下
    """
    if not os.path.isdir(result_folder_path):
        raise FileNotFoundError(f"结果文件夹路径不存在: {result_folder_path}")
    
    if not os.path.isdir(test_data_root):
        raise FileNotFoundError(f"测试数据根目录不存在: {test_data_root}")
    
    # 获取所有子文件夹
    sub_folders = [d for d in os.listdir(result_folder_path)
                   if os.path.isdir(os.path.join(result_folder_path, d))]
    sub_folders = sorted(sub_folders)
    
    if not sub_folders:
        raise FileNotFoundError(f"在 '{result_folder_path}' 下未找到任何子文件夹")
    
    print(f"[INFO] 找到 {len(sub_folders)} 个子文件夹: {sub_folders}")
    
    # 算法列表
    algos = {
        "DPP": DPP,
        "GHSS": GHSS,
        "GAHSS": GAHSS,
        "GSI_LS": GSI_LS,
    }
    
    total_combinations = len(sub_folders) * len(algos)
    current_combination = 0
    
    # 遍历每个子文件夹
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(result_folder_path, sub_folder)
        
        # 从子文件夹名称中提取test_folder和k值
        # 格式：{test_folder}_{k}，例如 points_1000_3_100_20
        # 需要找到最后一个下划线分隔的k值
        parts = sub_folder.rsplit('_', 1)
        if len(parts) != 2:
            print(f"[WARN] 无法从 '{sub_folder}' 中提取test_folder和k值，跳过")
            continue
        
        test_folder = parts[0]
        try:
            k = int(parts[1])
        except ValueError:
            print(f"[WARN] 无法从 '{sub_folder}' 中提取k值，跳过")
            continue
        
        # 构建测试数据目录
        test_data_dir = os.path.join(test_data_root, test_folder)
        if not os.path.isdir(test_data_dir):
            print(f"[WARN] 测试数据目录不存在: {test_data_dir}，跳过")
            continue
        
        print(f"\n[INFO] ===== 开始测试: {sub_folder} (test_folder={test_folder}, k={k}) =====")
        print(f"[INFO] 测试数据目录: {test_data_dir}")
        print(f"[INFO] 输出目录: {sub_folder_path}")
        
        # 对每个算法运行测试
        for algo_name, algo_fn in algos.items():
            current_combination += 1
            print(f"[INFO] [{current_combination}/{total_combinations}] 运行算法: {algo_name}")
            
            # 检查结果文件是否已存在
            out_path = os.path.join(sub_folder_path, f"{algo_name}.json")
            if os.path.exists(out_path):
                print(f"[INFO] {algo_name} 的结果已存在，跳过计算")
                continue
            
            try:
                result = run_benchmark_on_folder(
                    data_folder=test_data_dir,
                    algo_name=algo_name,
                    algo_fn=algo_fn,
                    k=k,
                    ref=ref,
                )
                
                # 保存结果
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"[INFO] {algo_name} 结果已保存至: {out_path}")
            except Exception as e:
                print(f"[ERROR] 运行 {algo_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[INFO] {sub_folder} 完成")
    
    print(f"\n[INFO] ===== 所有测试完成 =====")
    print(f"[INFO] 所有结果保存在: {result_folder_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="批量评测 HSS 系列算法（DPP、GHSS、GAHSS、GSI_LS）在测试数据上"
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default=None,
        help="指定的结果文件夹路径，例如 '../example/MEoH_HSS/test_result/mul_gpt_150'。如果不指定，则使用默认的run_benchmark函数。"
    )
    parser.add_argument(
        "--test_data_root",
        type=str,
        default="../data/test_data",
        help="测试数据根目录，默认 '../data/test_data'"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="../example/MEoH_HSS/test_result/GSI_LS_gpt_200",
        help="输出根目录（仅在未指定result_folder时使用），默认 '../example/MEoH_HSS/test_result'"
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[20, 50],
        help="k值列表，默认 [20, 50, 100]"
    )
    parser.add_argument(
        "--ref",
        type=float,
        default=1.1,
        help="参考点放大系数，默认 1.1"
    )
    
    args = parser.parse_args()
    
    if args.result_folder:
        # 使用新功能：在指定的结果文件夹路径下的每个子文件夹中运行测试
        run_benchmark_on_result_folder(
            result_folder_path=args.result_folder,
            test_data_root=args.test_data_root,
            ref=args.ref,
        )
    else:
        # 使用原有功能：在test_data_root下的所有测试文件夹中运行测试
        run_benchmark(
            test_data_root=args.test_data_root,
            output_root=args.output_root,
            k_list=args.k_list,
            ref=args.ref,
        )


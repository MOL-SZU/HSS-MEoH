import numpy as np
import time
from mat2array import load_mat_to_numpy
from typing import List


def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    HSS (Hypervolume-based Subset Selection): 贪心选择超体积贡献最大的 k 个点

    参数:
        points: np.ndarray
            所有候选点，形状为 (N, D)，其中 N 是点数，D 是目标维度
        k: int
            需要选择的点数
        reference_point: np.ndarray, optional
            参考点，形状为 (D,)。如果为 None，则使用 points 的最大值 * 1.1

    返回:
        subset: np.ndarray
            选出的子集，形状为 (k, D)
    """
    # 如果未提供参考点，自动计算
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1

    # 内部辅助函数：计算超体积
    def HV_cal(objectives_list: List[np.ndarray]) -> float:
        """计算给定解集列表的超体积"""
        import pygmo as pg
        if not objectives_list:
            return 0.0
        objectives_array = np.array(objectives_list)
        hv = pg.hypervolume(objectives_array)
        return hv.compute(reference_point)

    # 内部辅助函数：计算单个点的超体积贡献
    def hypervolume_contribution(point: np.ndarray,
                                 selected_subset: List[np.ndarray]) -> float:
        """计算点相对于当前子集的超体积贡献"""
        if len(selected_subset) == 0:
            return HV_cal([point])
        else:
            hv_before = HV_cal(selected_subset)
            hv_after = HV_cal(selected_subset + [point])
            return hv_after - hv_before

    # 主算法：贪心选择超体积贡献最大的点
    n, d = points.shape

    # 存储选中的索引
    selected_indices: List[int] = []
    # 存储选中的点（用于快速计算超体积贡献）
    selected_points: List[np.ndarray] = []

    # 逐次选择k个点
    while len(selected_indices) < k:
        max_contrib = -np.inf
        max_idx = None

        # 遍历所有未选中的点
        for idx in range(n):
            if idx in selected_indices:
                continue

            point = points[idx]
            # 计算该点的超体积贡献
            contrib = hypervolume_contribution(point, selected_points)

            # 更新最大贡献
            if contrib > max_contrib:
                max_contrib = contrib
                max_idx = idx

        # 如果找不到有效点（理论上不应该发生）
        if max_idx is None:
            # 如果没有点可选但还需要选择，就从剩余点中随机选择
            remaining_indices = [i for i in range(n) if i not in selected_indices]
            if remaining_indices:
                max_idx = remaining_indices[0]
            else:
                break

        # 添加选中的点
        selected_indices.append(max_idx)
        selected_points.append(points[max_idx])

    # 将选中的点转换为numpy数组
    subset = np.array(selected_points)

    return subset

def HV_cal(selected_objectives, reference_point):
    import pygmo as pg
    hv = pg.hypervolume(selected_objectives)
    return hv.compute(reference_point)

if __name__ == "__main__":
    # 测试示例
    row_data = load_mat_to_numpy('../data/my_data.mat','points')[:200, :]

    reference_point = np.max(row_data, axis=0) * 1.1

    reference_point = np.array([1.1, 1.1, 1.1])
    start = time.time()
    subset1 = HSS(row_data, 8, reference_point)
    time1 = time.time() - start
    score = HV_cal(list(subset1), reference_point)

    print("HV:", score, "time:", time1)
    print("reference_point:", reference_point)
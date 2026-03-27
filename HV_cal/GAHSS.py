import numpy as np
import time
from mat2array import load_mat_to_numpy


def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    Select k subsets from the points so that the hypervolume formed with the reference point is maximized.

    Args:
        points: Numpy array of shape (n, m) where each row is a point in the objective space
        k: Number of subsets to select
        reference_point: Numpy array of shape (m,) representing the reference point in the objective space

    Returns:
        Numpy array of shape (k, m) where each row is a selected point from the points

    Important: Set "all" random seeds to 2025, including the packages (such as scipy sub-packages) involving random seeds.
    """
    # 设置随机种子以确保可重复性
    np.random.seed(2025)

    # 获取点集的大小和维度
    solNum, M = points.shape

    # 参数设置
    num_vec = 100

    # 生成均匀权重向量（单位超球面上的向量）
    # 生成多元正态分布随机数
    mu = np.zeros(M)
    sigma = np.eye(M)
    R = np.random.multivariate_normal(mu, sigma, num_vec)

    # 归一化处理得到单位向量
    W = np.abs(R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True)))

    # 确保参考点有效
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1

    # 初始化张量
    tensor = np.zeros((solNum, num_vec))

    # 计算初始张量
    for i in range(solNum):
        s = points[i, :]
        # 计算最小归一化距离
        temp1 = np.min(np.abs(s - reference_point) / W, axis=1)
        tensor[i, :] = temp1

    # 初始化最小张量
    mintensor = tensor.copy()

    # 存储选择的点
    selVal = np.zeros((k, M))
    selected_indices = []  # 存储选择的索引

    # 迭代选择过程
    for num in range(k):
        # 更新最小张量（保持历史最小值）
        mintensor = np.minimum(mintensor, tensor)

        # 计算超体积贡献（对每个点）
        r2hvc = np.sum(mintensor, axis=1)

        # 选择贡献最大的解
        bestindex = np.argmax(r2hvc)

        # 确保不重复选择同一个点
        if bestindex in selected_indices:
            # 如果已经选择过，选择下一个最好的点
            r2hvc[bestindex] = -np.inf
            bestindex = np.argmax(r2hvc)

        selected_indices.append(bestindex)

        # 更新张量
        for i in range(solNum):
            s = points[i, :]
            # 计算新选择的点与其他点之间的最大归一化距离
            temp1 = np.max((points[bestindex, :] - s) / W, axis=1)
            tensor[i, :] = temp1

        # 存储选择的点
        selVal[num, :] = points[bestindex, :]

    return selVal


def HV_cal(selected_objectives, reference_point):
    import pygmo as pg
    hv = pg.hypervolume(selected_objectives)
    return hv.compute(reference_point)



if __name__ == "__main__":
    # 创建测试数据
    row_data = load_mat_to_numpy('../data/my_data.mat', 'points')[:200, :]

    reference_point = np.max(row_data, axis=0) * 1.1
    start = time.time()
    subset1 = HSS(row_data, 20, reference_point)
    time1 = time.time() - start
    score = HV_cal(subset1, reference_point)

    print("GAHSS HV:", score)
    print("GAHSS time:", time1)
import numpy as np
from scipy.linalg import cholesky, LinAlgError
import logging
from mat2array import load_mat_to_numpy
import time
from sklearn.preprocessing import MinMaxScaler


def HSS(points, k: int, reference_point) -> np.ndarray:
    n, m = points.shape
    reference_point = reference_point if reference_point is not None else np.max(points, axis=0) * 1.1

    # 快速计算收敛性和多样性
    v = np.prod(np.maximum(reference_point - points, 1e-10), axis=1)
    F_prime = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-10)
    F_similarity = F_prime @ F_prime.T

    # 初始化
    selected = []
    remaining = np.ones(n, dtype=bool)
    c_vecs = [np.array([]) for _ in range(n)]
    d_sq = F_similarity.diagonal().copy()

    for t in range(k):
        # 快速得分计算
        scores = np.full(n, -np.inf)
        valid = remaining & (d_sq > 1e-10)
        scores[valid] = 0.5 * v[valid] + (1 - 0.5) * np.log(d_sq[valid])

        # 选择最佳解
        best_idx = np.argmax(scores[remaining])
        actual_idx = np.where(remaining)[0][best_idx]
        selected.append(actual_idx)
        remaining[actual_idx] = False

        if t == k - 1:
            break

        # 快速更新
        d_j = np.sqrt(d_sq[actual_idx])
        remaining_idx = np.where(remaining)[0]
        F_ji = F_similarity[actual_idx, remaining_idx]

        if t == 0:
            e_vals = F_ji / d_j
            for idx, e_val in zip(remaining_idx, e_vals):
                c_vecs[idx] = np.array([e_val])
                d_sq[idx] -= e_val ** 2
        else:
            c_j = c_vecs[actual_idx]
            dot_prods = np.array(
                [np.dot(c_j[:min(len(c_j), len(c_vecs[i]))], c_vecs[i][:min(len(c_j), len(c_vecs[i]))])
                 for i in remaining_idx])
            e_vals = (F_ji - dot_prods) / d_j
            for idx, e_val in zip(remaining_idx, e_vals):
                c_vecs[idx] = np.append(c_vecs[idx], e_val)
                d_sq[idx] -= e_val ** 2

    return points[selected]

def HV_cal(selected_points, reference_point):
    import pygmo as pg
    hv = pg.hypervolume(selected_points)
    return hv.compute(reference_point)


if __name__ == "__main__":
    # row_data = load_mat_to_numpy('../data/raw_data/data_set_concave_triangular_M10_100000.mat', 'data_set')[:500, :]
    row_data = load_mat_to_numpy('../data/my_data.mat', 'points')[:200, :]

    reference_point = np.max(row_data, axis=0) * 1.1
    start = time.time()
    subset = HSS(row_data, 20)
    elapsed_time = time.time() - start
    HV_sel = HV_cal(subset, reference_point)

    print(f"DPP 子集 HV: {HV_sel}")
    print(f"DPP 算法执行时间: {elapsed_time:.6f} 秒")
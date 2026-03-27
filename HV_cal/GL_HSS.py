import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.special import comb
from typing import Tuple, List, Optional
import pygmo as pg


def HSS(points: np.ndarray, k: int, reference_point: Optional[np.ndarray] = None) -> np.ndarray:
    """
    基于 GLHSS_NAGO 和 TGAHSS 的 Python 实现
    参数:
        points: 输入数据点 (n x m)
        k: 选择的子集大小
        reference_point: 参考点 (1 x m)，如果为None则使用各维度的最大值

    返回:
        selected_set: 选择的子集 (k x m)
    """

    # 设置默认参数（硬编码在函数内部）
    num_vec1 = 1  # TGAHSS 第一阶段权重向量数
    num_sol2 = 200  # TGAHSS 第二阶段候选解数
    seed = 1  # 随机种子

    # 如果未提供参考点，则使用各维度的最大值
    if reference_point is None:
        ref = np.max(points, axis=0)
    else:
        ref = np.array(reference_point).flatten()

    # 数据维度
    n, m = points.shape

    # ====================== 内部函数定义开始 ======================

    def uniform_vector(N: int, M: int) -> Tuple[np.ndarray, int]:
        """生成均匀分布的权重向量"""
        if M == 1:
            return np.ones((N, 1)), N

        # 计算H1
        H1 = 1
        while comb(H1 + M - 1, M - 1) <= N:
            H1 += 1
        H1 -= 1

        # 生成权重向量
        W_list = []
        if M == 2:
            # 2维情况：直接生成
            for i in range(H1 + 1):
                vec = [i, H1 - i]
                prob = np.array(vec) / H1
                if np.all(prob > 0):
                    W_list.append(prob)
        else:
            # 高维情况：使用递归
            def generate_weights_recursive(H: int, M: int, current_vec: List[int] = None) -> List[np.ndarray]:
                """递归生成权重向量"""
                if current_vec is None:
                    current_vec = []

                if M == 1:
                    # 最后一维
                    complete_vec = current_vec + [H]
                    prob = np.array(complete_vec) / H
                    if np.all(prob > 0):
                        return [prob]
                    else:
                        return []

                weights = []
                for i in range(H + 1):
                    new_vec = current_vec + [i]
                    sub_weights = generate_weights_recursive(H - i, M - 1, new_vec)
                    weights.extend(sub_weights)

                return weights

            W_list = generate_weights_recursive(H1, M)

        if not W_list:
            # 如果没有生成任何向量，使用随机向量
            W = np.random.dirichlet(np.ones(M), N)
        else:
            W = np.array(W_list)
            actual_N = W.shape[0]

            if actual_N < N:
                # 补充随机向量
                extra = np.random.dirichlet(np.ones(M), N - actual_N)
                W = np.vstack([W, extra])

        # 归一化（添加维度检查）
        if W.ndim == 1:
            W = W.reshape(1, -1)

        # 确保有多个维度才能计算范数
        if W.shape[1] > 1:
            norms = np.linalg.norm(W, axis=1, keepdims=True)
            mask = norms.flatten() > 1e-10
            if np.any(mask):
                W[mask] = W[mask] / norms[mask]

        return W, W.shape[0]

    def HV_cal(selected_objectives, reference_point):
        """计算超体积"""
        hv = pg.hypervolume(selected_objectives)
        return hv.compute(reference_point)

    def stk_dominatedhv(data: np.ndarray, ref: np.ndarray) -> float:
        """计算被支配的超体积"""
        if len(data) == 0:
            return 0.0

        n, m = data.shape
        dominated = np.zeros(n, dtype=bool)

        # 识别被支配的点
        for i in range(n):
            for j in range(n):
                if i != j and np.all(data[j, :] <= data[i, :]) and np.any(data[j, :] < data[i, :]):
                    dominated[i] = True
                    break

        non_dominated = data[~dominated, :]

        if len(non_dominated) == 0:
            return 0.0

        # 计算超体积
        return HV_cal(non_dominated, ref)

    def HVCE(s: np.ndarray, data: np.ndarray, r: np.ndarray) -> float:
        """计算超体积贡献"""
        if len(data) == 0:
            dataP = s.reshape(1, -1)
        else:
            dataP = np.maximum(data, s)

        hvc = np.prod(r - s) - stk_dominatedhv(dataP, r)
        return hvc

    def HVC_gradient(dataSel: np.ndarray, j: int, M: int, refSel: np.ndarray) -> np.ndarray:
        """计算HVC的梯度方向"""
        s = dataSel[j, :]
        diff = refSel - s
        diff = np.maximum(diff, 1e-10)  # 防止除零
        g = np.prod(diff) / diff
        return g.reshape(1, -1)

    def association_ini(data: np.ndarray, dataSel: np.ndarray) -> np.ndarray:
        """初始化关联矩阵"""
        n = data.shape[0]
        A = np.zeros((n, 2))

        # 计算距离
        dist = cdist(data, dataSel)
        A[:, 0] = np.min(dist, axis=1)
        A[:, 1] = np.argmin(dist, axis=1)

        return A

    def association_upd(data: np.ndarray, dataSel: np.ndarray, i: int, newS: np.ndarray, A: np.ndarray) -> np.ndarray:
        """更新关联矩阵"""
        n = data.shape[0]

        # 找到与i关联的点
        indRel = np.where(A[:, 1] == i)[0]
        indOth = np.where(A[:, 1] != i)[0]

        # 计算到新解的距离
        if len(indOth) > 0:
            distToNew = cdist(data[indOth, :], newS.reshape(1, -1)).flatten()

            # 找到关联发生变化的点
            indChangeOth = indOth[distToNew < A[indOth, 0]]
            indRel = np.concatenate([indRel, indChangeOth])

        # 更新数据选择
        dataSel_new = dataSel.copy()
        dataSel_new[i, :] = newS

        # 重新计算关联
        if len(indRel) > 0:
            distToAll = cdist(data[indRel, :], dataSel_new)
            A[indRel, 0] = np.min(distToAll, axis=1)
            A[indRel, 1] = np.argmin(distToAll, axis=1)

        return A

    def TGAHSS(objVal: np.ndarray, selNum: int, num_vec1: int, num_sol2: int, seed: int, ref: np.ndarray) -> np.ndarray:
        """TGAHSS 算法实现"""
        np.random.seed(seed)
        solNum, M = objVal.shape
        num_vec2 = 100

        # 生成权重向量
        if num_vec1 > 0:
            W1 = np.zeros((num_vec1, M))
            W1[0, :] = np.sqrt(1 / M) * np.ones(M)
            if num_vec1 > 1:
                W1_vecs, _ = uniform_vector(num_vec1 - 1, M)
                if len(W1_vecs) > 0:
                    W1[1:1 + len(W1_vecs), :] = W1_vecs

        W2, _ = uniform_vector(num_vec2, M)

        # 初始化张量
        if num_vec1 > 0:
            tensor1 = np.full((solNum, num_vec1), np.inf)
        tensor2 = np.full((solNum, num_vec2), np.inf)

        count = np.zeros(solNum, dtype=int)
        numDist = 0

        # 计算到参考点的距离
        if num_vec1 > 0:
            if num_vec1 < 10:
                for i in range(num_vec1):
                    temp1 = np.min(np.abs(objVal - ref) / W1[i, :], axis=1)
                    tensor1[:, i] = temp1
            else:
                for i in range(solNum):
                    temp1 = np.min(np.abs(objVal[i, :] - ref) / W1, axis=1)
                    tensor1[i, :] = temp1
            numDist += num_vec1 * solNum

        if num_sol2 > 1:
            for i in range(solNum):
                temp2 = np.min(np.abs(objVal[i, :] - ref) / W2, axis=1)
                tensor2[i, :] = temp2
            numDist += num_vec2 * solNum

        # 初始化最小张量
        if num_vec1 > 0:
            mintensor1 = tensor1.copy()
        mintensor2 = tensor2.copy()

        selVal = np.zeros((selNum, M))
        selInd = np.zeros(selNum, dtype=int)
        canInd = np.ones(solNum, dtype=bool)

        # 主选择循环
        for num in range(selNum):
            # 第一阶段：选择候选解
            if num_vec1 > 0:
                mintensor1 = np.minimum(mintensor1, tensor1)
                r2hvc1 = np.sum(mintensor1, axis=1)
                candidateInd = np.argsort(r2hvc1)[-num_sol2:]
            else:
                unselectedInd = np.where(canInd)[0]
                perm = np.random.permutation(len(unselectedInd))
                candidateInd = unselectedInd[perm[:num_sol2]]

            # 第二阶段：精确评估候选解
            if num_sol2 > 1:
                for i in range(num_sol2):
                    s = objVal[candidateInd[i], :]
                    if num > 0:
                        # 计算距离
                        start_idx = count[candidateInd[i]]
                        temp3 = np.zeros((num - start_idx, num_vec2))

                        for j_idx, j in enumerate(range(start_idx, num)):
                            sn = selVal[j, :]
                            temp3[j_idx, :] = np.max((sn - s) / W2, axis=1)
                            numDist += num_vec2

                        # 更新最小张量
                        if temp3.shape[0] > 0:
                            mintensor2[candidateInd[i], :] = np.minimum(
                                mintensor2[candidateInd[i], :],
                                np.min(temp3, axis=0)
                            )

                # 更新计数
                count[candidateInd] = num

                # 计算近似HVC并选择最佳解
                r2hvc2 = np.sum(mintensor2[candidateInd, :], axis=1)
                maxInd = np.argmax(r2hvc2)
                bestindex = candidateInd[maxInd]
            else:
                bestindex = candidateInd[0] if len(candidateInd) > 0 else 0

            # 更新张量
            if num_vec1 > 0:
                if num_vec1 < 10:
                    for i in range(num_vec1):
                        temp1 = np.max((objVal[bestindex, :] - objVal) / W1[i, :], axis=1)
                        tensor1[:, i] = temp1
                else:
                    for i in range(solNum):
                        temp1 = np.max((objVal[bestindex, :] - objVal[i, :]) / W1, axis=1)
                        tensor1[i, :] = temp1
                numDist += num_vec1 * solNum

            # 保存选择
            selVal[num, :] = objVal[bestindex, :]
            selInd[num] = bestindex
            canInd[bestindex] = False

        return selVal

    def GLHSS_NAGO(data: np.ndarray, dataSel: np.ndarray, selNum: int, refSel: np.ndarray) -> np.ndarray:
        """GLHSS_NAGO 算法实现"""
        start_time = time.time()

        # 初始化关联
        A = association_ini(data, dataSel)

        M = data.shape[1]
        dataSelSet = np.zeros((selNum, M, 2))
        dataSelSet[:, :, 0] = dataSel.copy()

        gen = 1
        count = 0
        current_sel = dataSel.copy()

        while True:
            isC = np.zeros(selNum, dtype=bool)

            for j in range(selNum):
                # 计算当前解的超体积贡献
                other_indices = [i for i in range(selNum) if i != j]
                hvC = HVCE(current_sel[j, :], current_sel[other_indices, :], refSel)

                while True:
                    # 计算梯度方向
                    g = HVC_gradient(current_sel, j, M, refSel)

                    # 找到与当前解关联的候选解
                    indd = np.where(A[:, 1] == j)[0]

                    if len(indd) == 0:
                        break

                    # 计算角度
                    diff = data[indd, :] - current_sel[j, :]
                    norm_diff = np.linalg.norm(diff, axis=1, keepdims=True)
                    norm_g = np.linalg.norm(g)

                    # 防止除以零
                    mask = (norm_diff.flatten() > 1e-10) & (norm_g > 1e-10)
                    if np.sum(mask) == 0:
                        break

                    # 计算余弦相似度
                    cos_angle = np.zeros(len(indd))
                    cos_angle[mask] = np.sum(diff[mask] * g, axis=1) / (norm_diff[mask].flatten() * norm_g)

                    # 选择角度最大的解
                    inda = np.argmax(cos_angle)

                    # 计算新解的超体积贡献
                    hvN = HVCE(data[indd[inda], :], current_sel[other_indices, :], refSel)

                    if hvN > hvC:
                        count = 0
                        isC[j] = True

                        # 更新关联
                        A = association_upd(data, current_sel, j, data[indd[inda], :], A)

                        # 更新当前解
                        current_sel[j, :] = data[indd[inda], :]
                        hvC = hvN
                    else:
                        break

                    count += 1
                    if count >= selNum:
                        break

            # 保存当前代的结果
            if gen < 2:  # 只保存最终结果
                dataSelSet = np.zeros((selNum, M, 2))
                dataSelSet[:, :, 0] = dataSel.copy()
                dataSelSet[:, :, 1] = current_sel.copy()
            gen += 1

            # 终止条件
            if np.sum(isC) == 0 or count >= selNum:
                break

        # 返回最终结果
        return current_sel

    # ====================== 内部函数定义结束 ======================

    # 1. 使用 TGAHSS 初始化选择子集
    print("Step 1: Running TGAHSS for initialization...")
    subset_init = TGAHSS(points, k, num_vec1, num_sol2, seed, ref)

    # 2. 使用 GLHSS_NAGO 优化子集
    print("Step 2: Running GLHSS_NAGO for optimization...")
    final_subset = GLHSS_NAGO(points, subset_init, k, ref)

    return final_subset

def HV_cal(selected_objectives, reference_point):
    import pygmo as pg
    hv = pg.hypervolume(selected_objectives)
    return hv.compute(reference_point)


# 测试函数
def test_HSS():
    """测试函数"""
    from mat2array import load_mat_to_numpy

    points = load_mat_to_numpy('../data/my_data.mat', 'points')[:200, :]

    # 设置参考点（各维度的最大值）
    ref = np.max(points, axis=0) * 1.1

    ref = np.array([1.1, 1.1, 1.1])

    # 运行算法
    k = 8
    # print(f"Testing HSS with {n_points} points, {m} objectives, selecting {k} points")

    result = HSS(points, k, ref)

    print(f"Selected {len(result)} points:")
    print(result)

    # 计算超体积
    hv = HV_cal(result, ref)
    print(f"Hypervolume: {hv}")

    return result


if __name__ == "__main__":
    # 运行测试
    test_HSS()
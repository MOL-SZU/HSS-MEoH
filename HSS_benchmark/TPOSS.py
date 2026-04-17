"""
TPOSS (Targeted Pareto Optimal Subset Selection) for HSS.
Single entry: HSS(points, k, reference_point).
"""
import numpy as np
import pygmo as pg
from mat2array import load_mat_to_numpy
import time


def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    使用 TPOSS 从点集中选择 k 个子集，使与参考点形成的超体积最大。

    Args:
        points: 形状 (n, m)，每行为目标空间中的一个点
        k: 要选择的子集大小
        reference_point: 形状 (m,)，参考点

    Returns:
        形状 (k, m)，选中的点
    """
    n, m = points.shape
    if reference_point is None:
        reference_point = np.max(points, axis=0) * 1.1

    # 算法参数（在函数内部定义）
    epsilon = 1
    delta = 5
    T = int(round(2 * n * 8 * 8 * np.e))

    def calfitness(mask: np.ndarray) -> float:
        """计算选中子集的超体积。"""
        data = points[mask, :]
        if data.size == 0:
            return 0.0
        hv = pg.hypervolume(data)
        return hv.compute(reference_point)

    def targeted_mutation(offspring: np.ndarray, k_val: int, num: int = 1) -> np.ndarray:
        """Targeted mutation: 以与 |k-b| 相关的概率翻转位。"""
        offspring = offspring.astype(bool)
        b = int(np.sum(offspring))
        a = n - b
        rate0 = (abs(k_val - b) + k_val - b + num) / (2 * a) if a > 0 else 0.0
        rate1 = (abs(k_val - b) - k_val + b + num) / (2 * b) if b > 0 else 0.0
        rate = np.where(offspring, rate1, rate0)
        exchange = np.random.rand(n) < rate
        offspring = offspring.copy()
        offspring[exchange] = ~offspring[exchange]
        return offspring

    def env_selection(offspring: np.ndarray, population: np.ndarray, fitness: np.ndarray,
                     pop_size: int, lb: int, ub: int):
        """环境选择：仅接受 lb <= size <= ub 的后代，并更新种群。"""
        offspring = offspring.astype(bool)
        off_size = int(np.sum(offspring))
        if not (lb <= off_size <= ub):
            return population, pop_size, fitness

        index = np.where(fitness[:, 1] == off_size)[0]
        if len(index) == 0:
            off_fit1 = calfitness(offspring)
            population = np.vstack([population, offspring.astype(np.float64)])
            fitness = np.vstack([fitness, [off_fit1, off_size]])
            pop_size += 1
            return population, pop_size, fitness

        subpop = population[index, :]
        if np.any(np.all(subpop == offspring, axis=1)):
            return population, pop_size, fitness

        off_fit1 = calfitness(offspring)
        if len(index) < delta:
            population = np.vstack([population, offspring.astype(np.float64)])
            fitness = np.vstack([fitness, [off_fit1, off_size]])
            pop_size += 1
            return population, pop_size, fitness

        min_val = np.min(fitness[index, 0])
        if min_val < off_fit1:
            which = np.argmin(fitness[index, 0])
            population[index[which], :] = offspring.astype(np.float64)
            fitness[index[which], 0] = off_fit1
        return population, pop_size, fitness

    # 初始化：一个随机大小为 k 的子集
    population = np.zeros((1, n))
    idx = np.random.permutation(n)[:k]
    population[0, idx] = 1
    pop_size = 1
    fitness = np.array([[calfitness(population[0].astype(bool)), k]])

    lb, ub = k - epsilon, k + epsilon
    p = 0
    while p < T:
        s0 = population[np.random.randint(pop_size), :]
        offspring0 = targeted_mutation(s0, k, 1)
        p += 1
        population, pop_size, fitness = env_selection(
            offspring0, population, fitness, pop_size, lb, ub)

    # 在 size <= k 的解中取 fitness(1) 最大的
    temp = fitness[:, 1] <= k
    if not np.any(temp):
        seq = np.array([0])
    else:
        j = np.max(fitness[temp, 1])
        seq = np.where(fitness[:, 1] == j)[0]
    d = np.argmax(fitness[seq, 0])
    selected_mask = population[seq[d], :].astype(bool)
    result_points = points[selected_mask]
    # 保证返回恰好 k 个点
    if len(result_points) > k:
        result_points = result_points[:k]
    elif len(result_points) < k:
        unselected = np.where(~selected_mask)[0]
        need = k - len(result_points)
        extra_idx = np.random.choice(unselected, size=min(need, len(unselected)), replace=False)
        result_points = np.vstack([result_points, points[extra_idx]])
    return result_points

def HV_cal(selected_objectives, reference_point):
    import pygmo as pg
    hv = pg.hypervolume(selected_objectives)
    return hv.compute(reference_point)

if __name__ == "__main__":
    row_data = load_mat_to_numpy('../data/my_data.mat', 'points')[:200, :]
    reference_point = np.max(row_data, axis=0) * 1.1

    reference_point = np.array([1.1, 1.1, 1.1])
    start = time.time()
    subset1 = HSS(row_data, 8, reference_point)
    t = time.time() - start

    score = HV_cal(subset1, reference_point)
    print(f"TPOSS time: {t} 秒")
    print(f"TPOSS HV: {score}")
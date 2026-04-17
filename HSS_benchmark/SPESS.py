"""
SPESS (Sparse Pareto Evolutionary Subset Selection) for HSS.
Single entry: HSS(points, k, reference_point).
"""
import numpy as np
import pygmo as pg
from mat2array import load_mat_to_numpy
import time


def HSS(points, k: int, reference_point) -> np.ndarray:
    """
    使用 SPESS 从点集中选择 k 个子集，使与参考点形成的超体积最大。

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

    T = int(round(2 * n * 8 * 8 * np.e))
    lb, ub = 0, k + 1

    def calfitness(mask: np.ndarray) -> float:
        """计算选中子集的超体积。"""
        data = points[mask, :]
        if data.size == 0:
            return 0.0
        hv = pg.hypervolume(data)
        return hv.compute(reference_point)

    def sparseSS_mutation(offspring: np.ndarray) -> np.ndarray:
        """Sparse SS 变异：以 1/(2*sum) 或 1/(2*(n-sum)) 概率翻转。"""
        offspring = offspring.astype(bool)
        s = int(np.sum(offspring))
        rate1 = 1.0 / (2 * s) if s > 0 else 0.0
        rate0 = 1.0 / (2 * (n - s)) if (n - s) > 0 else 0.0
        rate = np.where(offspring, rate1, rate0)
        exchange = np.random.rand(n) < rate
        offspring = offspring.copy()
        offspring[exchange] = ~offspring[exchange]
        return offspring

    def sparseSS_crossover(s1: np.ndarray, s2: np.ndarray, k_val: int):
        """Sparse SS 交叉：在差异位上按给定概率交换。"""
        s1 = s1.astype(bool)
        s2 = s2.astype(bool)
        diff = s1 != s2
        xx = n  # MATLAB length(diff)
        x1 = int(np.sum(s1[diff]))
        if x1 == 0 or x1 == xx:
            return s1.copy(), s2.copy()
        rate10 = xx / 2.0 / 2.0 / x1
        rate01 = xx / 2.0 / 2.0 / (xx - x1)
        rate = np.zeros(n)
        rate[diff & s1] = rate10
        rate[diff & ~s1] = rate01
        exchange = np.random.rand(n) < rate
        swap = diff & exchange
        s1_new = s1.copy()
        s2_new = s2.copy()
        s1_new[swap] = s2[swap]
        s2_new[swap] = s1[swap]
        return s1_new, s2_new

    def evaluation_k(offspring: np.ndarray, population: np.ndarray, fitness: np.ndarray,
                     pop_size: int):
        """评估后代并更新种群（Pareto 支配）。"""
        offspring = offspring.astype(bool)
        off_size = int(np.sum(offspring))
        if not (lb < off_size < ub):
            return population, pop_size, fitness

        off_fit1 = calfitness(offspring)
        off_fit = np.array([off_fit1, off_size])

        # 是否存在解支配 offspring
        dominated = np.any(
            (fitness[:pop_size, 0] > off_fit1) & (fitness[:pop_size, 1] <= off_size)
        ) or np.any(
            (fitness[:pop_size, 0] >= off_fit1) & (fitness[:pop_size, 1] < off_size)
        )
        if dominated:
            return population, pop_size, fitness

        # 删除被 offspring 支配的解：f1 <= off_f1 且 f2 >= off_f2
        delete_index = (fitness[:pop_size, 0] <= off_fit1) & (fitness[:pop_size, 1] >= off_size)
        ndelete = np.where(~delete_index)[0]
        population = np.vstack([population[ndelete], offspring.astype(np.float64)])
        fitness = np.vstack([fitness[ndelete], off_fit])
        pop_size = len(ndelete) + 1
        return population, pop_size, fitness

    # 初始化：单解全 0（或随机一个大小为 k 的？）MATLAB 是 zeros(1,n)，然后 evaluation_k 只接受 0<size<k+1，所以第一个后代必须是 1..k 之间。所以初始 population 需要一个合法解。看 MATLAB  again: population=zeros(1,n); popSize=1; fitness=zeros(1,2); 所以初始是空集（全0），fitness=[0,0]。但 evaluation_k 要求 lb < off_size < ub 即 0 < size < k+1，所以 size 在 1..k。那初始 fitness(2)=0 不会被更新因为不会进 evaluation_k。所以第一轮随机变异 offspring 若 size 在 (0, k+1) 即 1..k 才会加入。所以初始可以保持 zeros(1,n), fitness=[-inf, 0] 或 [0, 0]。MATLAB 里 fitness=zeros(1,2) 即 [0,0]，所以初始解 size=0。OK。
    population = np.zeros((1, n))
    pop_size = 1
    fitness = np.array([[0.0, 0.0]])

    # Phase 1: 随机变异直到首次出现 k 个 size<=k 且 popSize==k+1 且 max(fitness(:,2))==k
    p = 0
    pp = 0
    while p < T:
        parent_idx = np.random.randint(pop_size)
        parent = population[parent_idx, :]
        r = (np.random.rand(n) < 1.0 / n).astype(np.float64)
        offspring = np.abs(parent - r)

        if p % (8 * n) == 0 and p > 0:
            temp = fitness[:, 1] <= k
            if np.any(temp):
                j = np.max(fitness[temp, 1])
                seq = np.where(fitness[:, 1] == j)[0]
                _ = fitness[seq]

        p += 1
        population, pop_size, fitness = evaluation_k(offspring, population, fitness, pop_size)

        if np.max(fitness[:, 1]) == k and pop_size == k + 1:
            pp = p
            p = T
            population = population[1:, :]
            fitness = fitness[1:, :]
            pop_size -= 1

    # Phase 2: 交叉 + 变异
    while pp < T:
        s1 = population[np.random.randint(pop_size), :].astype(bool)
        s2 = population[np.random.randint(pop_size), :].astype(bool)
        offspring1, offspring2 = sparseSS_crossover(s1, s2, k)
        offspring1 = sparseSS_mutation(offspring1)
        offspring2 = sparseSS_mutation(offspring2)

        if pp % (8 * n) == 0 and pp > 0:
            temp = fitness[:, 1] <= k
            if np.any(temp):
                j = np.max(fitness[temp, 1])
                seq = np.where(fitness[:, 1] == j)[0]
                _ = fitness[seq]

        pp += 1
        population, pop_size, fitness = evaluation_k(offspring1, population, fitness, pop_size)

        if pp % (8 * n) == 0 and pp > 0:
            temp = fitness[:, 1] <= k
            if np.any(temp):
                j = np.max(fitness[temp, 1])
                seq = np.where(fitness[:, 1] == j)[0]
                _ = fitness[seq]

        pp += 1
        population, pop_size, fitness = evaluation_k(offspring2, population, fitness, pop_size)

    # 选择：fitness(2) 最大的解中取 fitness(1) 最大的
    j = np.max(fitness[:, 1])
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
    print(f"SPESS time: {t} 秒")
    print(f"SPESS HV: {score}")
import inspect
import textwrap
import numpy as np

from evaluation import HSS_Evaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.method.meoh.resume import resume_meoh
from llm4ad.base import TextFunctionProgramConverter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# 六个基准算法，作为种子加入初始种群
from HV_cal.GL_HSS import HSS as GL_HSS
from HV_cal.GAHSS import HSS as GAHSS_HSS
from HV_cal.GHSS import HSS as GHSS_HSS
from HV_cal.GSI_LS import HSS as GSI_LS_HSS
from HV_cal.SPESS import HSS as SPESS_HSS
from HV_cal.TPOSS import HSS as TPOSS_HSS


def main():
    llm = HttpsApi(
        host='www.dmxapi.cn',
        key='sk-Bf6yC0nmwmn5sBSN9QHlshTIqUEkGRsWvyi5s1Enlgt6qU55',
        model='DeepSeek-V3.2',
        timeout=240,
    )

    # 预先写好要跑的实验，按顺序自动执行
    experiment_configs = [
        # {
        #     'name': 'exp_raw',
        #     'resume_mode': False,
        #     'resume_log_dir': './logs/meoh/exp_raw',
        #     'log_dir': 'logs/meoh/exp_raw',
        #     'use_flash_reflection': False,
        #     'use_initialization': False,
        #     'use_harmony_search': False,
        #     'use_pt_operators': False,
        #     'parent_selection_strategy': 'dominated',
        # },
        # {
        #     'name': 'exp_ours',
        #     'resume_mode': False,
        #     'resume_log_dir': './logs/meoh/exp_ours',
        #     'log_dir': 'logs/meoh/exp_ours',
        #     'use_flash_reflection': True,
        #     'use_initialization': True,
        #     'use_harmony_search': False,
        #     'use_pt_operators': True,
        #     'parent_selection_strategy': 'dominated',
        # },
        {
            'name': 'exp_FR',
            'resume_mode': False,
            'resume_log_dir': './logs/meoh/LLM4AD_initial_dominate_FR_150',
            'log_dir': 'logs/meoh/exp_FR',
            'use_flash_reflection': False,
            'use_initialization': True,
            'use_harmony_search': False,
            'use_pt_operators': True,
            'parent_selection_strategy': 'dominated',
        },
        # {
        #     'name': 'exp_initial',
        #     'resume_mode': False,
        #     'resume_log_dir': './logs/meoh/LLM4AD_initial_dominate_FR_150',
        #     'log_dir': 'logs/meoh/exp_initial',
        #     'use_flash_reflection': True,
        #     'use_initialization': False,
        #     'use_harmony_search': False,
        #     'use_pt_operators': True,
        #     'parent_selection_strategy': 'dominated',
        # },
        # {
        #     'name': 'exp_weighted',
        #     'resume_mode': False,
        #     'resume_log_dir': './logs/meoh/LLM4AD_initial_dominate_FR_150',
        #     'log_dir': 'logs/meoh/exp_weighted',
        #     'use_flash_reflection': True,
        #     'use_initialization': True,
        #     'use_harmony_search': False,
        #     'use_pt_operators': True,
        #     'parent_selection_strategy': 'weighted',
        # },
        # {
        #     'name': 'exp_uniform',
        #     'resume_mode': False,
        #     'resume_log_dir': './logs/meoh/LLM4AD_initial_dominate_FR_150',
        #     'log_dir': 'logs/meoh/exp_uniform',
        #     'use_flash_reflection': True,
        #     'use_initialization': True,
        #     'use_harmony_search': False,
        #     'use_pt_operators': True,
        #     'parent_selection_strategy': 'uniform',
        # },
    ]

    for idx, config in enumerate(experiment_configs, start=1):
        print("=" * 80)
        print(f"[INFO] 开始实验 {idx}/{len(experiment_configs)}: {config['name']}")
        print(
            f"[INFO] 参数: resume_mode={config['resume_mode']}, "
            f"use_initialization={config['use_initialization']}, "
            f"use_harmony_search={config['use_harmony_search']}, "
            f"use_pt_operators={config['use_pt_operators']}, "
            f"parent_selection_strategy={config['parent_selection_strategy']}"
        )

        task = HSS_Evaluation()
        method = MEoH(
            llm=llm,
            profiler=MEoHProfiler(
                log_dir=config['resume_log_dir'] if config['resume_mode'] else config['log_dir'],
                num_objs=2,
                log_style='simple',
                create_random_path=not config['resume_mode'],
            ),
            evaluation=task,
            max_sample_nums=150,
            max_generations=150,
            pop_size=10,
            num_samplers=4,
            num_evaluators=4,
            num_objs=2,
            use_flash_reflection=config['use_flash_reflection'],
            debug_mode=False,
            parent_selection_strategy=config['parent_selection_strategy'],
            resume_mode=config['resume_mode'],
            use_harmony_search=config['use_harmony_search'],
            use_p1_operator=config['use_pt_operators'],
            use_t1_operator=config['use_pt_operators'],
            # resume_direction='efficient'
        )

        if config['resume_mode']:
            print(f"[INFO] 从日志目录恢复实验: {config['resume_log_dir']}")
            resume_meoh(method, config['resume_log_dir'])
        else:
            if config['use_initialization']:
                print("[INFO] 使用基准算法进行初始化。")
                add_seed_algorithms(method)
            else:
                print("[INFO] 不使用基准算法初始化，直接开始演化。")

        method.run()
        print(f"[INFO] 实验完成: {config['name']}")
        print("=" * 80)


def add_seed_algorithms(method: MEoH):
    seed_algos = {
        'GL_HSS': GL_HSS,
        'GAHSS': GAHSS_HSS,
        'GHSS': GHSS_HSS,
        'GSI_LS': GSI_LS_HSS,
        'SPESS': SPESS_HSS,
        'TPOSS': TPOSS_HSS,
    }

    template_program = method._template_program
    added_funcs = []

    for name, algo_func in seed_algos.items():
        try:
            algo_src = inspect.getsource(algo_func)
            algo_src = textwrap.dedent(algo_src)

            func = TextFunctionProgramConverter.text_to_function(algo_src)
            if func is None:
                print(f"[WARN] 无法将 {name} 转换为 Function，跳过。")
                continue

            func.algorithm = name

            program = TextFunctionProgramConverter.function_to_program(func, template_program)
            if program is None:
                print(f"[WARN] 无法将 {name} 转换为 Program，跳过。")
                continue

            score, eval_time = method._evaluator.evaluate_program_record_time(program)
            func.score = score
            func.evaluate_time = eval_time
            func.entire_code = str(program)

            if func.score is not None and not np.isinf(np.array(func.score)).any():
                added_funcs.append(func)

            if method._profiler is not None:
                method._profiler.register_function(func, program=str(program))

            print(f"[INFO] 种子算法 {name} 已评估，score={score}")

        except Exception as e:
            print(f"[ERROR] 处理种子算法 {name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    if len(added_funcs) > 0:
        _inject_seeds_into_population(method._population, added_funcs)
        _update_elitist_with_seed_algorithms(method._population, added_funcs)
        print(f"[INFO] 已将 {len(added_funcs)} 个基准算法加入初始种群，elitist 数量: {len(method._population.elitist)}")
        for func in method._population.elitist:
            print(f"[INFO]   - {getattr(func, 'algorithm', '?')}: score={func.score}")
    else:
        print("[WARN] 没有有效的种子算法可以加入初始种群。")


def _inject_seeds_into_population(population, seed_funcs):
    """将种子函数加入 _next_gen_pop，使它们参与第一轮 survival 进入初始种群。"""
    try:
        population._lock.acquire()
        valid = [func for func in seed_funcs if func.score is not None and not np.isinf(np.array(func.score)).any()]
        for func in valid:
            population._next_gen_pop.append(func)
    finally:
        if population._lock.locked():
            population._lock.release()


def _update_elitist_with_seed_algorithms(population, seed_funcs):
    """根据种子算法之间的非支配关系更新 elitist。"""
    try:
        population._lock.acquire()

        valid_seed_funcs = []
        for func in seed_funcs:
            if func.score is None:
                continue
            score_array = np.array(func.score)
            if np.isinf(score_array).any():
                continue
            valid_seed_funcs.append(func)

        if len(valid_seed_funcs) == 0:
            return

        unique_funcs = []
        seen_scores = set()
        for func in valid_seed_funcs:
            score_tuple = tuple(np.array(func.score).tolist())
            if score_tuple not in seen_scores:
                unique_funcs.append(func)
                seen_scores.add(score_tuple)

        if len(unique_funcs) == 0:
            return

        objs = [np.array(func.score) for func in unique_funcs]
        objs_array = -np.array(objs)
        nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)

        population._elitist = [unique_funcs[i] for i in nondom_idx.tolist()]

    except Exception as e:
        print(f"[ERROR] 更新 elitist 时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if population._lock.locked():
            population._lock.release()


if __name__ == '__main__':
    main()

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = WORKSPACE_DIR.parent
for path in (WORKSPACE_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.evaluation import HSS_Evaluation
from llm4ad.base import TextFunctionProgramConverter
from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.method.meoh.resume import resume_meoh
from llm4ad.tools.llm.llm_api_https import HttpsApi
from paths import LOGS_DIR, ensure_result_dirs
from HSS_benchmark.GL_HSS import HSS as GL_HSS
from HSS_benchmark.GAHSS import HSS as GAHSS_HSS
from HSS_benchmark.GHSS import HSS as GHSS_HSS
from HSS_benchmark.GSI_LS import HSS as GSI_LS_HSS
from HSS_benchmark.SPESS import HSS as SPESS_HSS
from HSS_benchmark.TPOSS import HSS as TPOSS_HSS


def main():
    ensure_result_dirs()
    llm = HttpsApi(host='www.dmxapi.cn', key='sk-Bf6yC0nmwmn5sBSN9QHlshTIqUEkGRsWvyi5s1Enlgt6qU55', model='DeepSeek-V3.2', timeout=240)
    meoh_log_root = LOGS_DIR / 'meoh'
    experiment_configs = [{
        'name': 'exp_FR',
        'resume_mode': False,
        'resume_log_dir': meoh_log_root / 'LLM4AD_initial_dominate_FR_150',
        'log_dir': meoh_log_root / 'exp_FR',
        'use_flash_reflection': False,
        'use_initialization': True,
        'use_harmony_search': False,
        'use_pt_operators': True,
        'parent_selection_strategy': 'dominated',
    }]

    for idx, config in enumerate(experiment_configs, start=1):
        print('=' * 80)
        print(f"[INFO] Start experiment {idx}/{len(experiment_configs)}: {config['name']}")
        method = MEoH(
            llm=llm,
            profiler=MEoHProfiler(log_dir=str(config['resume_log_dir'] if config['resume_mode'] else config['log_dir']), num_objs=2, log_style='simple', create_random_path=not config['resume_mode']),
            evaluation=HSS_Evaluation(),
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
        )
        if config['resume_mode']:
            resume_meoh(method, str(config['resume_log_dir']))
        elif config['use_initialization']:
            add_seed_algorithms(method)
        method.run()
        print(f"[INFO] Finished experiment: {config['name']}")
        print('=' * 80)


def add_seed_algorithms(method: MEoH):
    seed_algos = {'GL_HSS': GL_HSS, 'GAHSS': GAHSS_HSS, 'GHSS': GHSS_HSS, 'GSI_LS': GSI_LS_HSS, 'SPESS': SPESS_HSS, 'TPOSS': TPOSS_HSS}
    template_program = method._template_program
    added_funcs = []
    for name, algo_func in seed_algos.items():
        try:
            algo_src = textwrap.dedent(inspect.getsource(algo_func))
            func = TextFunctionProgramConverter.text_to_function(algo_src)
            if func is None:
                continue
            func.algorithm = name
            program = TextFunctionProgramConverter.function_to_program(func, template_program)
            if program is None:
                continue
            score, eval_time = method._evaluator.evaluate_program_record_time(program)
            func.score = score
            func.evaluate_time = eval_time
            func.entire_code = str(program)
            if func.score is not None and not np.isinf(np.array(func.score)).any():
                added_funcs.append(func)
            if method._profiler is not None:
                method._profiler.register_function(func, program=str(program))
        except Exception as exc:
            print(f'[ERROR] Failed to process seed {name}: {exc}')
    if added_funcs:
        _inject_seeds_into_population(method._population, added_funcs)
        _update_elitist_with_seed_algorithms(method._population, added_funcs)


def _inject_seeds_into_population(population, seed_funcs):
    try:
        population._lock.acquire()
        population._next_gen_pop.extend([func for func in seed_funcs if func.score is not None and not np.isinf(np.array(func.score)).any()])
    finally:
        if population._lock.locked():
            population._lock.release()


def _update_elitist_with_seed_algorithms(population, seed_funcs):
    try:
        population._lock.acquire()
        valid_seed_funcs = [func for func in seed_funcs if func.score is not None and not np.isinf(np.array(func.score)).any()]
        unique_funcs = []
        seen_scores = set()
        for func in valid_seed_funcs:
            score_tuple = tuple(np.array(func.score).tolist())
            if score_tuple not in seen_scores:
                unique_funcs.append(func)
                seen_scores.add(score_tuple)
        if unique_funcs:
            objs_array = -np.array([np.array(func.score) for func in unique_funcs])
            nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)
            population._elitist = [unique_funcs[i] for i in nondom_idx.tolist()]
    finally:
        if population._lock.locked():
            population._lock.release()


if __name__ == '__main__':
    main()

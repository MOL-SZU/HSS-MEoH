# Module Name: MEoH
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Shunyu Yao, Fei Liu, Xi Lin, Zhichao Lu, Zhenkun Wang, and Qingfu Zhang.
#       "Multi-objective evolution of heuristic using large language model."
#       In Proceedings of the AAAI Conference on Artificial Intelligence, 2025.
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import ast
import concurrent.futures
import copy
import re
import sys
import time
import traceback
from threading import Lock, Thread
import numpy as np

from .flash_reflection import MEoHFlashReflection
from .population import Population
from .profiler import MEoHProfiler
from .prompt import MEoHPrompt
from .sampler import MEoHSampler
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator, SampleTrimmer
)
from ...tools.profiler import ProfilerBase


class MEoH:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: int | None = 10,
                 max_sample_nums: int | None = 100,
                 pop_size: int = 20,
                 selection_num=2,
                 use_e2_operator: bool = True,
                 use_m1_operator: bool = True,
                 use_m2_operator: bool = True,
                 use_p1_operator: bool = True,
                 use_t1_operator: bool = True,
                 use_flash_reflection: bool = False,
                 use_harmony_search: bool = False,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 num_objs: int = 2,
                 *,
                 resume_mode: bool = False,
                 resume_direction: str | None = None,
                 initial_sample_num: int | None = None,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: str = 'thread',
                 parent_selection_strategy: str = 'dominated',
                 exploitation_alpha: float = 1.0,
                 parent_selection_lambda: float = 10.0,
                 **kwargs):
        """
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.meoh.MEoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums'.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations'.
            pop_size        : population size.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            use_p1_operator : if use p1 operator (performance unchanged, shorter time).
            use_t1_operator : if use t1 operator (time unchanged, better performance).
            use_flash_reflection : if use one flash-reflection update after each generation transition.
            use_harmony_search : if set to True, replace m2 prompt sampling with the HSEvo-style Harmony Search operator.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            resume_direction: in resume_mode, specify exploration direction: 'efficient' (focus on performance) or 'time' (focus on time reduction). If None, use both operators equally.
            debug_mode      : if set to True, we will print detailed information.
            parent_selection_strategy: 父代选择策略 'uniform' | 'power_law' | 'weighted' | 'dominated'(默认).
            exploitation_alpha: power_law 策略的指数，0 为均匀抽样。
            parent_selection_lambda: weighted 策略的 lambda，越大越倾向选优。
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._num_objs = num_objs
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator
        self._use_p1_operator = use_p1_operator
        self._use_t1_operator = use_t1_operator
        self._use_flash_reflection = use_flash_reflection
        self._use_harmony_search = use_harmony_search
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._resume_direction = resume_direction
        self._initial_sample_num = initial_sample_num
        self._debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval
        self._parent_selection_strategy = parent_selection_strategy
        self._exploitation_alpha = exploitation_alpha
        self._parent_selection_lambda = parent_selection_lambda

        # 在resume模式下，根据方向调整算子使用
        if self._resume_mode and self._resume_direction is not None:
            if self._resume_direction == 'efficient':
                # 重点使用t1算子（提高性能）
                self._use_t1_operator = True
                self._use_p1_operator = False
            elif self._resume_direction == 'time':
                # 重点使用p1算子（缩短时间）
                self._use_p1_operator = True
                self._use_t1_operator = False

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # population, sampler, and evaluator
        self._population = Population(
            pop_size=self._pop_size,
            parent_selection_strategy=self._parent_selection_strategy,
            exploitation_alpha=self._exploitation_alpha,
            parent_selection_lambda=self._parent_selection_lambda,
        )
        llm.debug_mode = debug_mode
        self._sampler = MEoHSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: Necessary

        # statistics
        self._tot_sample_nums = 0 if initial_sample_num is None else initial_sample_num
        self._reflection_lock = Lock()
        self._last_reflection_generation = -1
        self._flash_reflection = (
            MEoHFlashReflection(llm, self._task_description_str)
            if self._use_flash_reflection else None
        )
        self._previous_reflection_baseline = None
        self._hs_hm_size = 5
        self._hs_hmcr = 0.7
        self._hs_par = 0.5
        self._hs_bandwidth = 0.2
        self._hs_max_iter = 5

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

    def _get_reflection_context(self, operator_name: str) -> str:
        # 根据不同的算子来使用不同的reflection提示。
        if self._flash_reflection is None:
            return ""
        return self._flash_reflection.get_context_for_operator(operator_name)

    def _handle_generation_transition(self, generation_before: int, previous_population_snapshot):
        if not self._use_flash_reflection:
            return
        if self._population.generation <= generation_before:
            return

        transition = self._population.consume_generation_transition()
        if transition is None:
            transition = {
                "generation": self._population.generation,
                "previous_population": list(previous_population_snapshot),
                "current_elitist": list(self._population.elitist),
            }

        with self._reflection_lock:
            generation = transition["generation"]
            if generation <= self._last_reflection_generation:
                return

            reflection_worked = self._did_reflection_improve(
                self._previous_reflection_baseline,
                transition.get("current_elitist", []),
            )
            self._flash_reflection.update(
                generation=generation,
                current_elitist=transition.get("current_elitist", []),
                previous_population=transition.get("previous_population", []),
                reflection_worked=reflection_worked,
            )
            memory = self._flash_reflection.memory
            # print(f"\n================= Flash Reflection Generation {generation} =================")
            # print("**Analysis:**")
            # print(memory.analysis or "None")
            # print("**Experience:**")
            # print(memory.experience or "None")
            # if memory.comprehensive_reflection:
            #     print("**Comprehensive Reflection:**")
            #     print(memory.comprehensive_reflection)
            # print("==========================================================================\n")
            if isinstance(self._profiler, MEoHProfiler):
                self._profiler.register_flash_reflection(generation, memory)
            self._previous_reflection_baseline = self._summarize_elitist_scores(
                transition.get("current_elitist", [])
            )
            self._last_reflection_generation = generation

    @staticmethod
    def _summarize_elitist_scores(funcs):
        best_perf = None
        best_time = None
        for func in funcs:
            if func is None or func.score is None:
                continue
            score = np.array(func.score)
            if np.isinf(score).any():
                continue
            perf = float(score[0])
            time_score = float(score[1]) if len(score) > 1 else float("-inf")
            best_perf = perf if best_perf is None else max(best_perf, perf)
            best_time = time_score if best_time is None else max(best_time, time_score)
        if best_perf is None or best_time is None:
            return None
        return best_perf, best_time

    @staticmethod
    def _did_reflection_improve(previous_baseline, current_elitist):
        if previous_baseline is None:
            return None
        current_baseline = MEoH._summarize_elitist_scores(current_elitist)
        if current_baseline is None:
            return False
        prev_perf, prev_time = previous_baseline
        curr_perf, curr_time = current_baseline
        return curr_perf > prev_perf or curr_time > prev_time

    def _evaluate_and_register_function(self, func, parent_indivs=None, thought=None, sample_time=None):
        if func is None:
            return

        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return

        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()

        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought if thought is not None else getattr(func, 'algorithm', '')
        func.sample_time = 0.0 if sample_time is None else sample_time
        if not func.algorithm:
            func.algorithm = '{Harmony Search tuned variant}'

        try:
            if self._profiler is not None:
                self._profiler.register_function(func, program=str(program))
                if isinstance(self._profiler, MEoHProfiler):
                    self._profiler.register_population(self._population)
                self._tot_sample_nums += 1
        except Exception:
            traceback.print_exc()

        generation_before = self._population.generation
        previous_population_snapshot = list(self._population.population)
        self._population.register_function(func)
        self._handle_generation_transition(generation_before, previous_population_snapshot)
        if parent_indivs:
            for f in parent_indivs:
                setattr(f, 'num_offspring', getattr(f, 'num_offspring', 0) + 1)

    def _sample_evaluate_register(self, prompt, parent_indivs=None):
        """Sample a function using the given prompt -> evaluate it by submitting to the process/thread pool ->
        add the function to the population and register it to the profiler.
        parent_indivs: 本次用于生成 prompt 的父代列表，成功注册子代后会给这些父代的 num_offspring 加 1。
        """
        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return

        self._evaluate_and_register_function(
            func,
            parent_indivs=parent_indivs,
            thought=thought,
            sample_time=sample_time,
        )


    @staticmethod
    def _parse_parameter_ranges(parameter_ranges_text):
        content = parameter_ranges_text.strip()
        if not content:
            return None
        if '=' in content:
            left, right = content.split('=', 1)
            if left.strip() == 'parameter_ranges':
                content = right.strip()
        try:
            parsed = ast.literal_eval(content)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None

        normalized = {}
        for key, value in parsed.items():
            if not isinstance(key, str):
                return None
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                return None
            try:
                normalized[key] = (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                return None
        return normalized or None

    @classmethod
    def _extract_harmony_candidate(cls, response):
        blocks = re.findall(r"```python\s*(.*?)```", response or "", flags=re.IGNORECASE | re.DOTALL)
        func_block = None
        parameter_ranges = None
        for block in blocks:
            candidate = block.strip()
            if func_block is None and 'def ' in candidate:
                func_block = candidate
            if parameter_ranges is None:
                parameter_ranges = cls._parse_parameter_ranges(candidate)
        return func_block, parameter_ranges

    def _initialize_harmony_memory(self, bounds):
        harmony_memory = np.zeros((self._hs_hm_size, len(bounds)))
        for i, (lower_bound, upper_bound) in enumerate(bounds):
            harmony_memory[:, i] = np.random.uniform(lower_bound, upper_bound, self._hs_hm_size)
        return harmony_memory

    def _create_new_harmony(self, harmony_memory, bounds):
        new_harmony = np.zeros((harmony_memory.shape[1],))
        for i in range(harmony_memory.shape[1]):
            if np.random.rand() < self._hs_hmcr:
                new_harmony[i] = harmony_memory[np.random.randint(0, harmony_memory.shape[0]), i]
                if np.random.rand() < self._hs_par:
                    adjustment = np.random.uniform(-1, 1) * (bounds[i][1] - bounds[i][0]) * self._hs_bandwidth
                    new_harmony[i] += adjustment
            else:
                new_harmony[i] = np.random.uniform(bounds[i][0], bounds[i][1])
            new_harmony[i] = np.clip(new_harmony[i], bounds[i][0], bounds[i][1])
        return new_harmony

    @staticmethod
    def _score_key(score):
        return tuple(float(x) for x in np.array(score).tolist())

    @staticmethod
    def _is_valid_score(score):
        if score is None:
            return False
        score_array = np.array(score, dtype=float)
        return not np.isinf(score_array).any()

    @staticmethod
    def _non_dominated_indices(scores):
        valid_scores = np.array(scores, dtype=float)
        if len(valid_scores) == 0:
            return []
        objs_array = -valid_scores
        return NonDominatedSorting().do(objs_array, only_non_dominated_front=True).tolist()

    @staticmethod
    def _crowding_distance(scores):
        scores = np.array(scores, dtype=float)
        if len(scores) == 0:
            return np.array([])
        if len(scores) <= 2:
            return np.full(len(scores), np.inf)

        num_points, num_objs = scores.shape
        distances = np.zeros(num_points, dtype=float)

        for obj_idx in range(num_objs):
            order = np.argsort(scores[:, obj_idx])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            obj_values = scores[order, obj_idx]
            span = obj_values[-1] - obj_values[0]
            if span <= 1e-12:
                continue
            for rank in range(1, num_points - 1):
                if np.isinf(distances[order[rank]]):
                    continue
                prev_value = obj_values[rank - 1]
                next_value = obj_values[rank + 1]
                distances[order[rank]] += (next_value - prev_value) / span

        return distances

    def _select_hs_replacement(self, population_hs, candidate):
        trial_population = population_hs + [candidate]
        trial_scores = [item['score'] for item in trial_population]
        front_indices = self._non_dominated_indices(trial_scores)
        new_index = len(trial_population) - 1

        if new_index not in front_indices:
            return None

        dominated_existing = [idx for idx in range(len(population_hs)) if idx not in front_indices]
        if dominated_existing:
            return dominated_existing[0]

        if len(population_hs) == 0:
            return None

        front_scores = np.array([trial_scores[idx] for idx in front_indices], dtype=float)
        crowding = self._crowding_distance(front_scores)
        front_pos = {idx: pos for pos, idx in enumerate(front_indices)}
        candidate_front_pos = front_pos[new_index]
        candidate_crowding = crowding[candidate_front_pos]

        replaceable = [idx for idx in front_indices if idx != new_index]
        if not replaceable:
            return None

        replace_index = min(
            replaceable,
            key=lambda idx: (crowding[front_pos[idx]], idx),
        )

        if candidate_crowding < crowding[front_pos[replace_index]]:
            return None

        return replace_index

    def _select_effective_hs_candidate(self, population_hs):
        if not population_hs:
            return None

        elitist_scores = [
            np.array(func.score, dtype=float)
            for func in self._population.elitist
            if func is not None and self._is_valid_score(func.score)
        ]
        candidate_scores = [np.array(item['score'], dtype=float) for item in population_hs]
        combined_scores = elitist_scores + candidate_scores
        front_indices = self._non_dominated_indices(combined_scores)

        elitist_count = len(elitist_scores)
        candidate_front_indices = [idx - elitist_count for idx in front_indices if idx >= elitist_count]
        if not candidate_front_indices:
            return None

        if len(candidate_front_indices) == 1:
            return population_hs[candidate_front_indices[0]]

        front_scores = np.array([combined_scores[idx] for idx in front_indices], dtype=float)
        crowding = self._crowding_distance(front_scores)
        front_pos = {idx: pos for pos, idx in enumerate(front_indices)}

        best_candidate_index = max(
            candidate_front_indices,
            key=lambda idx: (
                crowding[front_pos[idx + elitist_count]],
                -idx,
            ),
        )
        return population_hs[best_candidate_index]

    @staticmethod
    def _ensure_hs_state(func):
        if func is None:
            return
        if not hasattr(func, 'hs_tried'):
            setattr(func, 'hs_tried', False)
        if not hasattr(func, 'hs_try_count'):
            setattr(func, 'hs_try_count', 0)

    def _rank_hs_candidates(self, funcs, *, prioritize_untried):
        ranked = []
        seen = set()
        for func in funcs:
            if func is None:
                continue
            self._ensure_hs_state(func)
            signature = (str(func), tuple(np.array(func.score).tolist()) if func.score is not None else None)
            if signature in seen:
                continue
            seen.add(signature)
            if prioritize_untried and getattr(func, 'hs_tried', False):
                continue
            ranked.append(func)
        ranked.sort(
            key=lambda f: (
                getattr(f, 'hs_try_count', 0),
                -float(np.array(f.score)[0]) if f.score is not None else float('inf'),
                -float(np.array(f.score)[1]) if f.score is not None and len(np.array(f.score)) > 1 else float('inf'),
            )
        )
        return ranked

    def _select_harmony_candidate(self):
        elitist_untried = self._rank_hs_candidates(self._population.elitist, prioritize_untried=True)
        if elitist_untried:
            return elitist_untried[0]

        population_untried = self._rank_hs_candidates(self._population.population, prioritize_untried=True)
        if population_untried:
            return population_untried[0]

        elitist_ranked = self._rank_hs_candidates(self._population.elitist, prioritize_untried=False)
        if elitist_ranked:
            return elitist_ranked[0]

        population_ranked = self._rank_hs_candidates(self._population.population, prioritize_untried=False)
        if population_ranked:
            return population_ranked[0]
        return None

    def _materialize_harmony_function(self, func_block, parameter_names, harmony_values):
        code = copy.deepcopy(func_block)
        for name, value in zip(parameter_names, harmony_values):
            code = code.replace('{' + name + '}', repr(float(value)))
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        if function is None:
            return None
        program = SampleTrimmer.sample_to_program(code, self._template_program)
        if program is None:
            return None
        function.entire_code = str(program)
        return function

    def _evaluate_harmony_candidate(self, func_block, parameter_names, harmony_values, sample_time):
        func = self._materialize_harmony_function(func_block, parameter_names, harmony_values)
        if func is None:
            return None
        func.algorithm = '{Harmony Search tuned variant}'
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return None
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()
        if score is None or np.isinf(np.array(score)).any():
            return None
        func.score = score
        func.evaluate_time = eval_time
        func.sample_time = sample_time
        return {
            'func': func,
            'score': score,
            'vector': np.array(harmony_values, dtype=float),
        }

    def _sample_evaluate_register_harmony(self, indi):
        self._ensure_hs_state(indi)
        setattr(indi, 'hs_tried', True)
        setattr(indi, 'hs_try_count', getattr(indi, 'hs_try_count', 0) + 1)

        sample_start = time.time()
        prompt = MEoHPrompt.get_prompt_hs(indi)
        response = self._sampler.llm.draw_sample(prompt)
        sample_time = time.time() - sample_start

        func_block, parameter_ranges = self._extract_harmony_candidate(response)
        if not func_block or not parameter_ranges:
            return

        parameter_names = list(parameter_ranges.keys())
        bounds = [parameter_ranges[name] for name in parameter_names]
        harmony_memory = self._initialize_harmony_memory(bounds)
        population_hs = []
        for memory_index, harmony in enumerate(harmony_memory):
            candidate = self._evaluate_harmony_candidate(
                func_block,
                parameter_names,
                harmony,
                sample_time,
            )
            if candidate is not None:
                candidate['memory_index'] = memory_index
                population_hs.append(candidate)
        if not population_hs:
            return

        for _ in range(self._hs_max_iter):
            new_harmony = self._create_new_harmony(harmony_memory, bounds)
            candidate = self._evaluate_harmony_candidate(
                func_block,
                parameter_names,
                new_harmony,
                sample_time,
            )
            if candidate is None:
                continue
            replace_index = self._select_hs_replacement(population_hs, candidate)
            if replace_index is not None:
                candidate['memory_index'] = population_hs[replace_index]['memory_index']
                population_hs[replace_index] = candidate
                harmony_memory[candidate['memory_index']] = new_harmony

        best_candidate = self._select_effective_hs_candidate(population_hs)
        if best_candidate is None:
            return
        self._evaluate_and_register_function(
            best_candidate['func'],
            parent_indivs=[indi],
            thought=best_candidate['func'].algorithm,
            sample_time=sample_time,
        )

    def _continue_sample(self):
        """Check if it meets the max_sample_nums restrictions.
        """
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        if self._max_generations is None and self._max_sample_nums is not None:
            if self._tot_sample_nums < self._max_sample_nums:
                return True
            else:
                return False
        if self._max_generations is not None and self._max_sample_nums is None:
            if self._population.generation < self._max_generations:
                return True
            else:
                return False
        if self._max_generations is not None and self._max_sample_nums is not None:
            continue_until_reach_gen = False
            continue_until_reach_sample = False
            if self._population.generation < self._max_generations:
                continue_until_reach_gen = True
            if self._tot_sample_nums < self._max_sample_nums:
                continue_until_reach_sample = True
            return continue_until_reach_gen and continue_until_reach_sample

    def _thread_do_evolutionary_operator(self):
        while self._continue_sample():
            try:
                # get a new func using e1
                indivs = [self._population.selection() for _ in range(self._selection_num)]
                prompt = MEoHPrompt.get_prompt_e1(
                    self._task_description_str,
                    indivs,
                    self._function_to_evolve,
                    reflection_context=self._get_reflection_context("e1"),
                )

                if self._debug_mode:
                    print(prompt)
                    input()

                self._sample_evaluate_register(prompt, parent_indivs=indivs)
                if not self._continue_sample():
                    break

                # get a new func using e2
                if self._use_e2_operator:
                    indivs = [self._population.selection() for _ in range(self._selection_num)]
                    prompt = MEoHPrompt.get_prompt_e2(
                        self._task_description_str,
                        indivs,
                        self._function_to_evolve,
                        reflection_context=self._get_reflection_context("e2"),
                    )

                    if self._debug_mode:
                        print(prompt)
                        input()

                    self._sample_evaluate_register(prompt, parent_indivs=indivs)
                    if not self._continue_sample():
                        break

                # get a new func using m1
                if self._use_m1_operator:
                    indiv = self._population.selection()
                    prompt = MEoHPrompt.get_prompt_m1(
                        self._task_description_str,
                        indiv,
                        self._function_to_evolve,
                        reflection_context=self._get_reflection_context("m1"),
                    )

                    if self._debug_mode:
                        print(prompt)
                        input()

                    self._sample_evaluate_register(prompt, parent_indivs=[indiv])
                    if not self._continue_sample():
                        break

                # get a new func using m2
                if self._use_m2_operator:
                    if self._use_harmony_search:
                        indiv = self._select_harmony_candidate()
                        if indiv is None:
                            break
                        if self._debug_mode:
                            print(MEoHPrompt.get_prompt_hs(indiv))
                            input()
                        self._sample_evaluate_register_harmony(indiv)
                    else:
                        indiv = self._population.selection()
                        prompt = MEoHPrompt.get_prompt_m2(
                            self._task_description_str,
                            indiv,
                            self._function_to_evolve,
                            reflection_context=self._get_reflection_context("m2"),
                        )

                        if self._debug_mode:
                            print(prompt)
                            input()

                        self._sample_evaluate_register(prompt, parent_indivs=[indiv])
                    if not self._continue_sample():
                        break

                # get a new func using p1 (performance unchanged, shorter time)
                if self._use_p1_operator:
                    # 从elitist中选择算法
                    elitist_selected = self._population.select_elitist_by_time()
                    # 从population中选择一些算法
                    pop_selected = [self._population.selection() for _ in
                                    range(min(self._selection_num, len(self._population)))]
                    # 合并elitist和population的算法
                    indivs = pop_selected + elitist_selected
                    if len(indivs) > 0:
                        prompt = MEoHPrompt.get_prompt_p1(
                            self._task_description_str,
                            indivs,
                            self._function_to_evolve,
                            reflection_context=self._get_reflection_context("p1"),
                        )

                        if self._debug_mode:
                            print(prompt)
                            input()

                        self._sample_evaluate_register(prompt, parent_indivs=indivs)
                        if not self._continue_sample():
                            break

                # get a new func using t1 (time unchanged, better performance)
                if self._use_t1_operator:
                    # 从elitist中选择算法
                    elitist_selected = self._population.select_elitist_by_performance()
                    # 从population中选择一些算法
                    pop_selected = [self._population.selection() for _ in
                                    range(min(self._selection_num, len(self._population)))]
                    # 合并elitist和population的算法
                    indivs = pop_selected + elitist_selected
                    if len(indivs) > 0:
                        prompt = MEoHPrompt.get_prompt_t1(
                            self._task_description_str,
                            indivs,
                            self._function_to_evolve,
                            reflection_context=self._get_reflection_context("t1"),
                        )

                        if self._debug_mode:
                            print(prompt)
                            input()

                        self._sample_evaluate_register(prompt, parent_indivs=indivs)
                        if not self._continue_sample():
                            break
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def _thread_init_population(self):
        """Let a thread repeat {sample -> evaluate -> register to population}
        to initialize a population.
        """
        while self._population.generation == 0:
            if not self._continue_sample():
                break
            try:
                # get a new func using i1
                prompt = MEoHPrompt.get_prompt_i1(
                    self._task_description_str,
                    self._function_to_evolve,
                    reflection_context=self._get_reflection_context("i1"),
                )
                self._sample_evaluate_register(prompt)
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _init_population(self):
        # threads for sampling
        sampler_threads = [
            Thread(
                target=self._thread_init_population,
            ) for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def _do_sample(self):
        sampler_threads = [
            Thread(
                target=self._thread_do_evolutionary_operator,
            ) for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        if not self._resume_mode:
            # do init
            # 注意：不再重新创建Population对象，使用__init__中已创建的对象
            # 这样可以保留在__init__之后添加到elitist中的算法（如种子算法）
            self._init_population()
            while len([f for f in self._population if not np.isinf(np.array(f.score)).any()]) < self._selection_num:
                self._population._generation -= 1
                self._init_population()
        # do evolve
        self._do_sample()

        # finish
        if self._profiler is not None:
            self._profiler.finish()

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import HSS_Evaluation
from llm4ad.base import TextFunctionProgramConverter
from llm4ad.method.meoh import MEoH
from llm4ad.method.meoh.prompt import MEoHPrompt
from llm4ad.method.meoh.resume import _load_json_records, _record_to_function
from llm4ad.tools.llm.llm_api_https import HttpsApi
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# PyCharm direct-run defaults. Edit these values, then right-click this file and run it.
DEFAULTS = {
    "elitist_json": r"./logs/meoh/exp_baseline/elitist/elitist_15.json",
    "host": "www.dmxapi.cn",
    "key": "sk-Bf6yC0nmwmn5sBSN9QHlshTIqUEkGRsWvyi5s1Enlgt6qU55",
    "model": "DeepSeek-V3.2",
    "timeout": 480,
    "output_dir": r"./logs/meoh/hs_compare",
    "k": 8,
    "data_folder": r"../../data/train_data",
    "data_key": "points",
    "num_evaluators": 1,
    "hm_size": 5,
    "hmcr": 0.7,
    "par": 0.5,
    "bandwidth": 0.2,
    "max_iter": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one Harmony Search pass on every function in an elitist JSON file and compare Pareto fronts."
    )
    parser.add_argument("--elitist-json", default=DEFAULTS["elitist_json"], help="Path to elitist_*.json")
    parser.add_argument("--host", default=DEFAULTS["host"], help="HTTPS API host, e.g. www.dmxapi.cn")
    parser.add_argument("--key", default=DEFAULTS["key"], help="API key")
    parser.add_argument("--model", default=DEFAULTS["model"], help="Model name")
    parser.add_argument("--timeout", type=int, default=DEFAULTS["timeout"], help="LLM timeout in seconds")
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"], help="Directory for JSON outputs and plot")
    parser.add_argument("--k", type=int, default=DEFAULTS["k"], help="HSS evaluation parameter k")
    parser.add_argument("--data-folder", default=DEFAULTS["data_folder"], help="Dataset folder for HSS evaluation")
    parser.add_argument("--data-key", default=DEFAULTS["data_key"], help="MAT key for dataset loading")
    parser.add_argument("--num-evaluators", type=int, default=DEFAULTS["num_evaluators"], help="Concurrent evaluators for scoring")
    parser.add_argument("--hm-size", type=int, default=DEFAULTS["hm_size"], help="Harmony memory size")
    parser.add_argument("--hmcr", type=float, default=DEFAULTS["hmcr"], help="Harmony memory consideration rate")
    parser.add_argument("--par", type=float, default=DEFAULTS["par"], help="Pitch adjustment rate")
    parser.add_argument("--bandwidth", type=float, default=DEFAULTS["bandwidth"], help="Pitch adjustment bandwidth")
    parser.add_argument("--max-iter", type=int, default=DEFAULTS["max_iter"], help="Harmony Search iterations")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def score_key(score) -> tuple[float, ...]:
    return tuple(float(x) for x in np.array(score).tolist())


def dominates(lhs, rhs) -> bool:
    a = np.array(lhs, dtype=float)
    b = np.array(rhs, dtype=float)
    return np.all(a >= b) and np.any(a > b)


def is_valid_score(score) -> bool:
    if score is None:
        return False
    score_array = np.array(score, dtype=float)
    return not np.isinf(score_array).any()


def non_dominated_indices(scores):
    valid_scores = np.array(scores, dtype=float)
    if len(valid_scores) == 0:
        return []
    objs_array = -valid_scores
    return NonDominatedSorting().do(objs_array, only_non_dominated_front=True).tolist()


def crowding_distance(scores):
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


def select_hs_replacement(candidates, candidate):
    trial_population = candidates + [candidate]
    trial_scores = [item["score"] for item in trial_population]
    front_indices = non_dominated_indices(trial_scores)
    new_index = len(trial_population) - 1

    if new_index not in front_indices:
        return None

    dominated_existing = [idx for idx in range(len(candidates)) if idx not in front_indices]
    if dominated_existing:
        return dominated_existing[0]

    if len(candidates) == 0:
        return None

    front_scores = np.array([trial_scores[idx] for idx in front_indices], dtype=float)
    crowding = crowding_distance(front_scores)
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


def select_effective_hs_candidate(candidates, reference_funcs):
    if not candidates:
        return None, []

    reference_scores = [
        np.array(func.score, dtype=float)
        for func in reference_funcs
        if func is not None and is_valid_score(func.score)
    ]
    candidate_scores = [np.array(item["score"], dtype=float) for item in candidates]
    combined_scores = reference_scores + candidate_scores
    front_indices = non_dominated_indices(combined_scores)

    reference_count = len(reference_scores)
    candidate_front_indices = [idx - reference_count for idx in front_indices if idx >= reference_count]
    if not candidate_front_indices:
        return None, front_indices

    if len(candidate_front_indices) == 1:
        return candidates[candidate_front_indices[0]], front_indices

    front_scores = np.array([combined_scores[idx] for idx in front_indices], dtype=float)
    crowding = crowding_distance(front_scores)
    front_pos = {idx: pos for pos, idx in enumerate(front_indices)}

    best_candidate_index = max(
        candidate_front_indices,
        key=lambda idx: (
            crowding[front_pos[idx + reference_count]],
            -idx,
        ),
    )
    return candidates[best_candidate_index], front_indices


def load_elitist_functions(elitist_json: Path):
    records = _load_json_records(str(elitist_json))
    funcs = []
    for record in records:
        func = _record_to_function(record, num_objs=2)
        if func is not None and func.score is not None:
            funcs.append(func)
    return funcs


def infer_parameter_types(func_block: str, parameter_names) -> dict[str, str]:
    inferred = {name: "float" for name in parameter_names}
    try:
        tree = ast.parse(func_block)
        func_node = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if func_node is None:
            return inferred

        all_args = list(func_node.args.args) + list(func_node.args.kwonlyargs)
        for arg in all_args:
            if arg.arg not in inferred or arg.annotation is None:
                continue
            annotation = ast.unparse(arg.annotation).strip().lower()
            if annotation in {"int", "np.int64", "np.int32"}:
                inferred[arg.arg] = "int"
            elif annotation in {"float", "np.float64", "np.float32"}:
                inferred[arg.arg] = "float"
    except Exception:
        return inferred
    return inferred


def coerce_parameter_value(name: str, value: float, parameter_types: dict[str, str], bounds) -> int | float:
    lower, upper = bounds
    clipped = float(np.clip(float(value), lower, upper))
    if parameter_types.get(name, "float") == "int":
        coerced = int(round(clipped))
        coerced = max(int(np.ceil(lower)), min(int(np.floor(upper)), coerced))
        if any(token in name.lower() for token in ("seed", "size", "samples", "chunk", "divisor", "max", "min")):
            coerced = max(1, coerced)
        return coerced
    return clipped


def initialize_harmony_memory_typed(method: MEoH, parameter_names, bounds, parameter_types):
    harmony_memory = np.zeros((method._hs_hm_size, len(bounds)))
    for i, (name, (lower_bound, upper_bound)) in enumerate(zip(parameter_names, bounds)):
        if parameter_types.get(name, "float") == "int":
            low = int(np.ceil(lower_bound))
            high = int(np.floor(upper_bound))
            if any(token in name.lower() for token in ("seed", "size", "samples", "chunk", "divisor", "max", "min")):
                low = max(1, low)
            if high < low:
                high = low
            harmony_memory[:, i] = np.random.randint(low, high + 1, size=method._hs_hm_size)
        else:
            harmony_memory[:, i] = np.random.uniform(lower_bound, upper_bound, method._hs_hm_size)
    return harmony_memory


def create_new_harmony_typed(method: MEoH, harmony_memory, parameter_names, bounds, parameter_types):
    new_harmony = np.zeros((harmony_memory.shape[1],))
    for i, (name, (lower_bound, upper_bound)) in enumerate(zip(parameter_names, bounds)):
        if np.random.rand() < method._hs_hmcr:
            candidate = harmony_memory[np.random.randint(0, harmony_memory.shape[0]), i]
            if np.random.rand() < method._hs_par:
                adjustment = np.random.uniform(-1, 1) * (upper_bound - lower_bound) * method._hs_bandwidth
                candidate = candidate + adjustment
        else:
            if parameter_types.get(name, "float") == "int":
                low = int(np.ceil(lower_bound))
                high = int(np.floor(upper_bound))
                if any(token in name.lower() for token in ("seed", "size", "samples", "chunk", "divisor", "max", "min")):
                    low = max(1, low)
                if high < low:
                    high = low
                candidate = np.random.randint(low, high + 1)
            else:
                candidate = np.random.uniform(lower_bound, upper_bound)
        new_harmony[i] = coerce_parameter_value(name, candidate, parameter_types, (lower_bound, upper_bound))
    return new_harmony


def pareto_front(funcs):
    valid = [f for f in funcs if f.score is not None and not np.isinf(np.array(f.score)).any()]
    if not valid:
        return []
    front = []
    for i, func_i in enumerate(valid):
        dominated = False
        score_i = np.array(func_i.score, dtype=float)
        for j, func_j in enumerate(valid):
            if i == j:
                continue
            score_j = np.array(func_j.score, dtype=float)
            if np.all(score_j >= score_i) and np.any(score_j > score_i):
                dominated = True
                break
        if not dominated:
            front.append(func_i)
    return front


def run_single_hs(method: MEoH, func, reference_funcs):
    return run_single_hs_with_artifacts(method, func, reference_funcs, artifact_dir=None, func_index=None)


def evaluate_harmony_candidate_verbose(
    method: MEoH,
    func_block,
    parameter_names,
    harmony_values,
    parameter_types,
    bounds_map,
    sample_time=0.0,
):
    code = render_harmony_code(func_block, parameter_names, harmony_values, parameter_types, bounds_map)
    func, materialize_error = materialize_harmony_function(method, code)
    if func is None:
        detail = {
            "status": "materialize_failed",
            "harmony": [float(x) for x in harmony_values],
            "code": code,
        }
        if materialize_error:
            detail["error"] = materialize_error
        return None, detail

    program = method._template_program.__class__(method._template_program.preface, [func])
    try:
        score, eval_time = method._evaluation_executor.submit(
            method._evaluator.evaluate_program_record_time,
            program
        ).result()
    except Exception as exc:
        return None, {
            "status": "evaluation_exception",
            "harmony": [float(x) for x in harmony_values],
            "code": code,
            "error": repr(exc),
        }

    if score is None:
        return None, {"status": "score_none", "harmony": [float(x) for x in harmony_values], "code": code}

    score_array = np.array(score)
    if np.isinf(score_array).any():
        return None, {
            "status": "score_invalid",
            "harmony": [float(x) for x in harmony_values],
            "code": code,
            "score": score_array.tolist(),
        }

    func.score = score
    func.evaluate_time = eval_time
    func.sample_time = sample_time
    func.algorithm = '{Harmony Search tuned variant}'
    return {
        "func": func,
        "score": score,
        "vector": np.array(harmony_values, dtype=float),
    }, {
        "status": "ok",
        "harmony": [float(x) for x in harmony_values],
        "score": score_array.tolist(),
    }


def write_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)


def dump_candidate_code(artifact_dir: Path | None, func_index: int | None, candidate_idx: int, detail: dict):
    if artifact_dir is None:
        return
    if "code" not in detail:
        return
    prefix = f"func_{func_index:03d}" if func_index is not None else "func"
    suffix = detail.get("status", "candidate")
    write_text(
        artifact_dir / f"{prefix}_candidate_{candidate_idx:03d}_{suffix}.py",
        detail["code"],
    )


def render_harmony_code(func_block: str, parameter_names, harmony_values, parameter_types, bounds_map) -> str:
    code = str(func_block)
    if any(('{' + name + '}') in code for name in parameter_names):
        for name, value in zip(parameter_names, harmony_values):
            coerced = coerce_parameter_value(name, value, parameter_types, bounds_map[name])
            code = code.replace('{' + name + '}', repr(coerced))
        return code

    try:
        tree = ast.parse(code)
        func_node = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if func_node is None:
            return code

        defaults_map = {
            name: coerce_parameter_value(name, value, parameter_types, bounds_map[name])
            for name, value in zip(parameter_names, harmony_values)
        }
        positional_args = list(func_node.args.args)
        positional_defaults = list(func_node.args.defaults)
        positional_default_args = positional_args[len(positional_args) - len(positional_defaults):]
        for idx, arg_node in enumerate(positional_default_args):
            if arg_node.arg in defaults_map:
                func_node.args.defaults[idx] = ast.parse(repr(defaults_map[arg_node.arg]), mode='eval').body

        kwonlyargs = list(func_node.args.kwonlyargs)
        for idx, arg_node in enumerate(kwonlyargs):
            if func_node.args.kw_defaults[idx] is not None and arg_node.arg in defaults_map:
                func_node.args.kw_defaults[idx] = ast.parse(repr(defaults_map[arg_node.arg]), mode='eval').body

        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return code


def materialize_harmony_function(method: MEoH, code: str):
    try:
        func = TextFunctionProgramConverter.text_to_function(code)
        if func is None:
            func = method._materialize_harmony_function(code, [], [])
        if func is None:
            return None, "failed to parse function from rendered code"

        program = TextFunctionProgramConverter.function_to_program(func, method._template_program)
        if program is None:
            return None, "failed to convert function into template program"

        func.entire_code = str(program)
        return func, None
    except Exception as exc:
        return None, repr(exc)


def run_single_hs_with_artifacts(method: MEoH, func, reference_funcs, artifact_dir: Path | None, func_index: int | None):
    print(f"[HS] Start: algorithm={getattr(func, 'algorithm', '')}, score={np.array(func.score).tolist()}", flush=True)
    prompt = MEoHPrompt.get_prompt_hs(func)
    print("[HS] Prompt sent to LLM:", flush=True)
    print(prompt, flush=True)
    response = method._sampler.llm.draw_sample(prompt)
    print("[HS] LLM response:", flush=True)
    print(response, flush=True)

    if artifact_dir is not None:
        prefix = f"func_{func_index:03d}" if func_index is not None else "func"
        write_text(artifact_dir / f"{prefix}_prompt.txt", prompt)
        write_text(artifact_dir / f"{prefix}_response.txt", response)

    func_block, parameter_ranges = method._extract_harmony_candidate(response)
    if not func_block or not parameter_ranges:
        print("[HS] Parse failed: no parameterized function or parameter_ranges extracted.", flush=True)
        if artifact_dir is not None:
            prefix = f"func_{func_index:03d}" if func_index is not None else "func"
            write_json(
                artifact_dir / f"{prefix}_hs_debug.json",
                {"status": "parse_failed", "original_score": np.array(func.score).tolist(), "response": response},
            )
        return None, {"status": "parse_failed", "response": response}

    parameter_names = list(parameter_ranges.keys())
    bounds = [parameter_ranges[name] for name in parameter_names]
    bounds_map = {name: parameter_ranges[name] for name in parameter_names}
    parameter_types = infer_parameter_types(func_block, parameter_names)
    print(f"[HS] Parsed {len(parameter_names)} parameters: {parameter_names}", flush=True)
    print(f"[HS] Parameter types: {parameter_types}", flush=True)
    harmony_memory = initialize_harmony_memory_typed(method, parameter_names, bounds, parameter_types)
    candidates = []
    candidate_logs = []
    candidate_counter = 0
    for memory_index, harmony in enumerate(harmony_memory):
        candidate_counter += 1
        candidate, detail = evaluate_harmony_candidate_verbose(
            method,
            func_block,
            parameter_names,
            harmony,
            parameter_types,
            bounds_map,
            sample_time=0.0,
        )
        candidate_logs.append(detail)
        dump_candidate_code(artifact_dir, func_index, candidate_counter, detail)
        if candidate is not None:
            candidate["memory_index"] = memory_index
            candidates.append(candidate)
    if not candidates:
        print("[HS] No valid initial harmony candidate survived evaluation.", flush=True)
        print("[HS] Initial candidate diagnostics:", flush=True)
        for idx, log in enumerate(candidate_logs, start=1):
            print(f"  - candidate {idx}: {log['status']}", flush=True)
            if "score" in log:
                print(f"    score={log['score']}", flush=True)
            if "error" in log:
                print(f"    error={log['error']}", flush=True)
        if artifact_dir is not None:
            prefix = f"func_{func_index:03d}" if func_index is not None else "func"
            write_json(
                artifact_dir / f"{prefix}_hs_debug.json",
                {
                    "status": "no_valid_candidate",
                    "original_score": np.array(func.score).tolist(),
                    "parameter_ranges": parameter_ranges,
                    "candidate_logs": candidate_logs,
                },
            )
        return None, {
            "status": "no_valid_candidate",
            "response": response,
            "parameter_ranges": parameter_ranges,
            "candidate_logs": candidate_logs,
        }

    print(f"[HS] Initial valid candidates: {len(candidates)}", flush=True)
    for _ in range(method._hs_max_iter):
        new_harmony = create_new_harmony_typed(method, harmony_memory, parameter_names, bounds, parameter_types)
        candidate_counter += 1
        candidate, detail = evaluate_harmony_candidate_verbose(
            method,
            func_block,
            parameter_names,
            new_harmony,
            parameter_types,
            bounds_map,
            sample_time=0.0,
        )
        candidate_logs.append(detail)
        dump_candidate_code(artifact_dir, func_index, candidate_counter, detail)
        if candidate is None:
            continue
        replace_index = select_hs_replacement(candidates, candidate)
        if replace_index is not None:
            candidate["memory_index"] = candidates[replace_index]["memory_index"]
            candidates[replace_index] = candidate
            harmony_memory[candidate["memory_index"]] = new_harmony

    best, combined_front_indices = select_effective_hs_candidate(candidates, reference_funcs)
    if best is None:
        print("[HS] No tuned candidate enters the combined elitist Pareto front.", flush=True)
        if artifact_dir is not None:
            prefix = f"func_{func_index:03d}" if func_index is not None else "func"
            write_json(
                artifact_dir / f"{prefix}_hs_debug.json",
                {
                    "status": "not_effective_on_elitist",
                    "original_score": np.array(func.score).tolist(),
                    "parameter_ranges": parameter_ranges,
                    "candidate_logs": candidate_logs,
                    "reference_scores": [
                        np.array(ref_func.score).tolist()
                        for ref_func in reference_funcs
                        if ref_func is not None and is_valid_score(ref_func.score)
                    ],
                },
            )
        return None, {
            "status": "not_effective_on_elitist",
            "response": response,
            "parameter_ranges": parameter_ranges,
            "candidate_logs": candidate_logs,
            "combined_front_indices": combined_front_indices,
        }

    print(f"[HS] Effective tuned score on elitist front: {np.array(best['score']).tolist()}", flush=True)
    if artifact_dir is not None:
        prefix = f"func_{func_index:03d}" if func_index is not None else "func"
        write_json(
            artifact_dir / f"{prefix}_hs_debug.json",
            {
                "status": "ok",
                "original_score": np.array(func.score).tolist(),
                "best_score": np.array(best["score"]).tolist(),
                "parameter_ranges": parameter_ranges,
                "candidate_logs": candidate_logs,
                "combined_front_indices": combined_front_indices,
            },
        )
    return best["func"], {
        "status": "ok",
        "response": response,
        "parameter_ranges": parameter_ranges,
        "candidate_logs": candidate_logs,
        "combined_front_indices": combined_front_indices,
    }


def func_to_record(func, *, source, original_score=None, hs_status=None):
    return {
        "algorithm": getattr(func, "algorithm", ""),
        "function": str(func),
        "score": np.array(func.score).tolist() if func.score is not None else None,
        "source": source,
        "original_score": original_score,
        "hs_status": hs_status,
    }


def plot_fronts(original_funcs, improved_funcs, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skip plotting.")
        return False

    orig_all = np.array([np.array(f.score, dtype=float) for f in original_funcs], dtype=float)
    new_all = np.array([np.array(f.score, dtype=float) for f in improved_funcs], dtype=float)
    orig_pf = np.array([np.array(f.score, dtype=float) for f in pareto_front(original_funcs)], dtype=float)
    new_pf = np.array([np.array(f.score, dtype=float) for f in pareto_front(improved_funcs)], dtype=float)

    plt.figure(figsize=(8, 6))
    if len(orig_all):
        plt.scatter(orig_all[:, 0], orig_all[:, 1], label="Original elitist", alpha=0.55, color="#4C78A8")
    if len(new_all):
        plt.scatter(new_all[:, 0], new_all[:, 1], label="After HS", alpha=0.55, color="#F58518")
    if len(orig_pf):
        order = np.argsort(orig_pf[:, 0])
        plt.plot(orig_pf[order, 0], orig_pf[order, 1], color="#4C78A8", linewidth=2, label="Original PF")
    if len(new_pf):
        order = np.argsort(new_pf[:, 0])
        plt.plot(new_pf[order, 0], new_pf[order, 1], color="#F58518", linewidth=2, label="HS PF")

    plt.xlabel("Objective 1: Hypervolume")
    plt.ylabel("Objective 2: -Running Time")
    plt.title("Pareto Front Comparison Before and After Harmony Search")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def main():
    args = parse_args()
    elitist_json = Path(args.elitist_json).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve() if args.output_dir else elitist_json.parent / "hs_compare")
    artifact_dir = ensure_dir(output_dir / "artifacts")

    if not args.key or args.key == "YOUR_API_KEY":
        raise ValueError("Please edit DEFAULTS['key'] in hs_on_elitist_compare.py before running in PyCharm.")

    print(f"[INFO] Loading elitist file: {elitist_json}", flush=True)
    print(f"[INFO] Output directory: {output_dir}", flush=True)
    print(f"[INFO] Artifact directory: {artifact_dir}", flush=True)

    llm = HttpsApi(
        host=args.host,
        key=args.key,
        model=args.model,
        timeout=args.timeout,
    )
    print(f"[INFO] LLM ready: host={args.host}, model={args.model}", flush=True)
    evaluation = HSS_Evaluation(k=args.k, data_folder=args.data_folder, data_key=args.data_key)
    print(f"[INFO] Evaluation ready: k={args.k}, data_folder={args.data_folder}, data_key={args.data_key}", flush=True)
    method = MEoH(
        llm=llm,
        evaluation=evaluation,
        pop_size=10,
        max_generations=1,
        max_sample_nums=1,
        num_evaluators=args.num_evaluators,
        num_objs=2,
        use_harmony_search=True,
        use_flash_reflection=False,
        use_e2_operator=False,
        use_m1_operator=False,
        use_m2_operator=False,
        use_p1_operator=False,
        use_t1_operator=False,
        debug_mode=False,
    )
    method._hs_hm_size = args.hm_size
    method._hs_hmcr = args.hmcr
    method._hs_par = args.par
    method._hs_bandwidth = args.bandwidth
    method._hs_max_iter = args.max_iter
    print(
        f"[INFO] HS config: hm_size={args.hm_size}, hmcr={args.hmcr}, par={args.par}, "
        f"bandwidth={args.bandwidth}, max_iter={args.max_iter}",
        flush=True,
    )

    original_funcs = load_elitist_functions(elitist_json)
    if not original_funcs:
        raise RuntimeError(f"No valid functions found in {elitist_json}")
    print(f"[INFO] Loaded {len(original_funcs)} elitist functions.", flush=True)

    improved_funcs = []
    comparison = []

    for idx, func in enumerate(original_funcs):
        print(f"\n[INFO] Processing function {idx + 1}/{len(original_funcs)}", flush=True)
        improved_func, meta = run_single_hs_with_artifacts(
            method,
            func,
            original_funcs,
            artifact_dir=artifact_dir,
            func_index=idx,
        )
        if improved_func is None:
            improved_funcs.append(func)
            comparison.append(
                {
                    "index": idx,
                    "status": meta["status"],
                    "original_score": np.array(func.score).tolist(),
                    "new_score": None,
                    "improved": False,
                }
            )
            print("[INFO] HS failed or produced no effective candidate on elitist front. Keep original function.", flush=True)
            continue

        improved = True
        kept_func = improved_func
        improved_funcs.append(kept_func)
        comparison.append(
            {
                "index": idx,
                "status": meta["status"],
                "original_score": np.array(func.score).tolist(),
                "new_score": np.array(improved_func.score).tolist(),
                "improved": improved,
                "parameter_ranges": meta.get("parameter_ranges"),
            }
        )
        print(
            f"[INFO] Original score={np.array(func.score).tolist()}, "
            f"new score={np.array(improved_func.score).tolist()}, improved={improved}",
            flush=True,
        )

    original_pf = pareto_front(original_funcs)
    improved_pf = pareto_front(improved_funcs)
    print(f"[INFO] Original PF size: {len(original_pf)}", flush=True)
    print(f"[INFO] Post-HS PF size: {len(improved_pf)}", flush=True)

    with open(output_dir / "original_elitist.json", "w", encoding="utf-8") as f:
        json.dump([func_to_record(func, source="original") for func in original_funcs], f, indent=2)
    with open(output_dir / "post_hs_population.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                func_to_record(
                    func,
                    source="post_hs",
                    original_score=comparison[idx]["original_score"],
                    hs_status=comparison[idx]["status"],
                )
                for idx, func in enumerate(improved_funcs)
            ],
            f,
            indent=2,
        )
    with open(output_dir / "original_pf.json", "w", encoding="utf-8") as f:
        json.dump([func_to_record(func, source="original_pf") for func in original_pf], f, indent=2)
    with open(output_dir / "post_hs_pf.json", "w", encoding="utf-8") as f:
        json.dump([func_to_record(func, source="post_hs_pf") for func in improved_pf], f, indent=2)
    with open(output_dir / "hs_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    plotted = plot_fronts(original_funcs, improved_funcs, output_dir / "pareto_front_comparison.png")

    improved_count = sum(1 for item in comparison if item["improved"])
    print(f"Loaded {len(original_funcs)} elitist functions from: {elitist_json}")
    print(f"Functions improved by HS: {improved_count}/{len(original_funcs)}")
    print(f"Original PF size: {len(original_pf)}")
    print(f"Post-HS PF size: {len(improved_pf)}")
    print(f"Plot generated: {'yes' if plotted else 'no'}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

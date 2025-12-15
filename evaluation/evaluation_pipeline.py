import os
import json
from copy import deepcopy
from typing import List
from pathlib import Path
from argparse import Namespace
from utils.tasks import Task
from utils.paths import get_benchmark_dir, get_llm_gen_dir
from utils.helper import convert_path2str, run_symbolic_planner
from agents.agent_code_gen import postprocess_response
from evaluation.evaluation import run_evaluation
from agentic_genplan.create_tasks import create_tasks_split


def create_eval_tasks(benchmark_name: str,
                      domain_name: str,
                      eval_split_file: str,
                      eval_split_name: str,
                      input_version) -> List[Task]:
    benchmark_dir = get_benchmark_dir(benchmark_name=benchmark_name,
                                      domain_name=domain_name)
    domain_file_path = benchmark_dir / 'domain.pddl'
    seed_split_path = benchmark_dir / eval_split_file

    with open(seed_split_path, 'r', encoding='utf-8') as f:
        seed_split_maps = json.load(f)
    eval_instances = seed_split_maps[eval_split_name]

    if input_version == 'pddl':
        instance_dir = benchmark_dir / 'problems_pddl'
    else:
        instance_dir = benchmark_dir / 'problems_pddl_nl_objs'
    eval_tasks = create_tasks_split(instance_dir=instance_dir,
                                    instance_names=eval_instances,
                                    domain_file_path=domain_file_path)

    return eval_tasks


def create_val_tasks(benchmark_name: str,
                     domain_name: str,
                     val_split_file: str,
                     val_split_name: str,
                     input_version) -> List[Task]:
    benchmark_dir = get_benchmark_dir(benchmark_name=benchmark_name,
                                      domain_name=domain_name)
    domain_file_path = benchmark_dir / 'domain.pddl'
    seed_split_path = benchmark_dir / val_split_file

    with open(seed_split_path, 'r', encoding='utf-8') as f:
        seed_split_maps = json.load(f)
    seed_split_data = seed_split_maps[val_split_name]

    val_instances_names = []
    val_instances_names.extend(seed_split_data['prompt'])
    val_instances_names.extend(seed_split_data['validate'])

    if input_version == 'pddl':
        instance_dir = benchmark_dir / 'problems_pddl'
    else:
        instance_dir = benchmark_dir / 'problems_pddl_nl_objs'

    val_tasks = create_tasks_split(instance_dir=instance_dir,
                                   instance_names=val_instances_names,
                                   domain_file_path=domain_file_path)

    return val_tasks


def create_llm_generated_eval_tasks(
        output_dir,
        benchmark_name: str,
        domain_name: str
    ):
    benchmark_dir = get_benchmark_dir(benchmark_name=benchmark_name,
                                      domain_name=domain_name)
    domain_file_path = benchmark_dir / 'domain.pddl'

    llm_gen_dir = get_llm_gen_dir(output_dir=output_dir,
                                  benchmark_name=benchmark_name,
                                  domain_name=domain_name)

    instance_dir = llm_gen_dir / 'problems_pddl'
    instance_names = list(os.listdir(instance_dir))
    solvable_instances_names = deepcopy(instance_names)

    plan_dir = llm_gen_dir / 'plans_pddl'
    for instance in instance_names:
        plan_name = str(instance).replace('.pddl', '_plan.txt')
        plan_path = os.path.join(plan_dir, plan_name)
        problem_path = str(os.path.join(instance_dir, instance))
        if not os.path.exists(plan_path):
            task = Task(domain_file_path=domain_file_path,
                        problem_file_path=problem_path)
            run_symbolic_planner(
                task=task,
                plan_file=plan_path,
                error_if_not_plan=False,
                optimal=False
            )
        if not os.path.exists(plan_path):
            solvable_instances_names.remove(instance)

    eval_tasks = create_tasks_split(instance_dir=instance_dir,
                                    domain_file_path=domain_file_path,
                                    instance_names=solvable_instances_names)

    return eval_tasks


def run_evaluation_data_split(output_dir_exp,
                              result_dir_exp,
                              eval_tasks: List[Task],
                              flags: Namespace) -> dict:

    all_exp_files = list(os.listdir(output_dir_exp))

    result_file = None
    for file in os.listdir(result_dir_exp):
        if file.startswith('results_final_'):
            result_file = os.path.join(result_dir_exp, file)
            break
    assert result_file

    with open(result_file, 'r') as f:
        result_data = json.load(f)

    step_first_code = 0
    step_best_code = result_data['debug_metrics']['step_best_validator']
    step_last_code = result_data['debug_metrics']['used_debugging_attempts']

    # TODO: the number of code versions
    ordered_code_files = []
    for file in all_exp_files:
        agent_type = file.split('_')[1]
        if agent_type == 'CodeGen':
            file_id = int(file.split('_')[0])
            if 'response_code' in file:
                ordered_code_files.append((file_id, file))

    ordered_code_files.sort()

    assert len(ordered_code_files) == step_last_code + 1

    ordered_code_files_names = [ocf[1] for ocf in ordered_code_files]

    best_eval_results = None
    first_eval_results = None
    last_eval_results = None
    if flags.eval_best:
        best_code_file_name = ordered_code_files_names[step_best_code]
        best_eval_results = get_eval_results_code(
            output_dir_exp=output_dir_exp,
            code_file_name=best_code_file_name,
            eval_tasks=eval_tasks,
            flags=flags
        )

    if flags.eval_last:
        if step_best_code == step_last_code:
            last_eval_results = deepcopy(best_eval_results)
        else:
            last_code_file_name = ordered_code_files_names[step_last_code]
            last_eval_results = get_eval_results_code(
                output_dir_exp=output_dir_exp,
                code_file_name=last_code_file_name,
                eval_tasks=eval_tasks,
                flags=flags
            )

    if flags.eval_first:
        if step_first_code == step_best_code:
            first_eval_results = deepcopy(best_eval_results)
        elif step_first_code == step_last_code:
            first_eval_results = deepcopy(last_eval_results)
        else:
            first_code_file_name = ordered_code_files_names[0]
            first_eval_results = get_eval_results_code(
                output_dir_exp=output_dir_exp,
                code_file_name=first_code_file_name,
                eval_tasks=eval_tasks,
                flags=flags
            )

    return_dict = {
        "last_result": last_eval_results,
        "step_last": step_last_code,
        "best_result": best_eval_results,
        "step_best": step_best_code,
        "first_result": first_eval_results,
        "step_first": step_first_code
    }

    return return_dict


def get_eval_results_code(output_dir_exp,
                          code_file_name,
                          eval_tasks,
                          flags):

    code_file = os.path.join(output_dir_exp, code_file_name)

    with open(code_file, 'r') as f:
        content = f.read()

    python_code = postprocess_response(content)
    eval_metrics = run_evaluation(eval_tasks=eval_tasks,
                                  generalized_plan_code=python_code,
                                  flags=flags)

    return eval_metrics


def run_all_evaluations_table(benchmark_name: str,
                              domain_name: str,
                              output_file: str,
                              flags: Namespace,
                              overwrite: bool = False):

    eval_split_names = flags.eval_split_names
    eval_split_file = flags.split_file
    input_version = flags.input_version
    experiments_output_folder = flags.experiments_output_folder
    experiments_results_folder = flags.experiments_results_folder
    experiments_names = flags.experiments_names

    header_results = [
        'dataset',
        'domain',
        'experiment_main',
        'experiment_name',
        'seed',
        'output_folder',
        'eval_split',
        'accuracy_last',
        'step_last',
        'accuracy_best',
        'step_best',
        'accuracy_first',
        'step_first'
    ]
    data2save = [header_results]

    eval_tasks_per_split = dict()
    for eval_name in eval_split_names:

        if eval_name == 'llm_generated':
            tasks = create_llm_generated_eval_tasks(benchmark_name=benchmark_name,
                                                    domain_name=domain_name,
                                                    output_dir=experiments_output_folder)
        else:
            tasks = create_eval_tasks(benchmark_name=benchmark_name,
                                      domain_name=domain_name,
                                      eval_split_file=eval_split_file,
                                      eval_split_name=eval_name,
                                      input_version=input_version)
        eval_tasks_per_split[eval_name] = tasks

    output_dir = os.path.split(output_file)[0]
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    writing_version = 'w' if overwrite else 'a'
    header = data2save.pop(0)
    with open(output_file, writing_version) as f:
        if writing_version == 'w':
            header_line = '\t'.join(header)
            f.write(f'{header_line}\n')

    for exp_name in experiments_names:
        output_sub_dir = Path(experiments_output_folder) /exp_name / f'{benchmark_name}-{domain_name}'
        result_sub_dir = Path(experiments_results_folder) / exp_name / f'{benchmark_name}-{domain_name}'

        for time_stamped_folder in os.listdir(output_sub_dir):
            t_stamped_out_path = output_sub_dir / str(time_stamped_folder)
            t_stamped_res_path = result_sub_dir / str(time_stamped_folder)
            for seed_folder in os.listdir(t_stamped_out_path):
                seed_out_path = t_stamped_out_path / str(seed_folder)
                seed_res_path = t_stamped_res_path / str(seed_folder)

                for eval_name, eval_tasks in eval_tasks_per_split.items():

                    eval_dict = run_evaluation_data_split(output_dir_exp=seed_out_path,
                                                          result_dir_exp=seed_res_path,
                                                          eval_tasks=eval_tasks,
                                                          flags=flags)
                    if len(eval_dict.keys()) == 0:
                        continue

                    columns = [
                        benchmark_name,
                        domain_name,
                        experiments_output_folder,
                        exp_name,
                        seed_folder,
                        time_stamped_folder,
                        eval_name,
                        eval_dict['last_result']['accuracy'],
                        eval_dict['step_last'],
                        eval_dict['best_result']['accuracy'],
                        eval_dict['step_best'],
                        eval_dict['first_result']['accuracy'],
                        eval_dict['step_first']
                    ]
                    data2save.append(columns)

                    with open(output_file, 'a') as f:
                        row_vals_str = [str(v) for v in columns]
                        line = '\t'.join(row_vals_str)
                        f.write(f'{line}\n')


def run_all_evaluations_json(benchmark_name: str,
                             domain_name: str,
                             flags: Namespace,
                             output_file: str = ''):

    eval_split_names = flags.eval_split_names
    eval_split_file = flags.split_file
    input_version = flags.input_version
    experiments_output_folder = flags.experiments_output_folder
    experiments_results_folder = flags.experiments_results_folder
    experiments_names = flags.experiments_names

    eval_tasks_per_split = dict()
    for eval_name in eval_split_names:

        if eval_name == 'llm_generated':
            tasks = create_llm_generated_eval_tasks(benchmark_name=benchmark_name,
                                                    domain_name=domain_name,
                                                    output_dir=experiments_output_folder)
        else:
            tasks = create_eval_tasks(benchmark_name=benchmark_name,
                                      domain_name=domain_name,
                                      eval_split_file=eval_split_file,
                                      eval_split_name=eval_name,
                                      input_version=input_version)
        eval_tasks_per_split[eval_name] = tasks

    for exp_name in experiments_names:
        output_sub_dir = Path(experiments_output_folder) / exp_name / f'{benchmark_name}-{domain_name}'
        result_sub_dir = Path(experiments_results_folder) / exp_name / f'{benchmark_name}-{domain_name}'

        for time_stamped_folder in os.listdir(output_sub_dir):
            t_stamped_out_path = output_sub_dir / str(time_stamped_folder)
            t_stamped_res_path = result_sub_dir / str(time_stamped_folder)
            for seed_folder in os.listdir(t_stamped_out_path):
                seed_out_path = t_stamped_out_path / str(seed_folder)
                seed_res_path = t_stamped_res_path / str(seed_folder)

                results_all_eval_splits = dict()

                for eval_name, eval_tasks in eval_tasks_per_split.items():

                    eval_dict = run_evaluation_data_split(output_dir_exp=seed_out_path,
                                                          result_dir_exp=seed_res_path,
                                                          eval_tasks=eval_tasks,
                                                          flags=flags)

                    results_all_eval_splits[eval_name] = convert_path2str(eval_dict)

                if output_file == '':
                    output_file_path = os.path.join(seed_res_path, 'additional_eval_data.json')
                else:
                    output_file_path = os.path.join(seed_res_path, output_file)
                with open(output_file_path, 'w') as f:
                    json.dump(results_all_eval_splits, f, indent=2)


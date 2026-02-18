import os
import json
from copy import deepcopy
from typing import List
from pathlib import Path, PosixPath
from collections import defaultdict
from argparse import Namespace
from utils.tasks import Task
from utils.helper import convert_path2str
from utils.paths import get_benchmark_dir
from agents.agent_code_gen import postprocess_response
from evaluation.evaluation import run_evaluation
from agentic_genplan.create_tasks import create_tasks_split


def run_evaluation_data_split_all_codes(output_dir_exp,
                                        result_dir_exp,
                                        validation_tasks: List[Task],
                                        eval_tasks: List[Task],
                                        flags: Namespace) -> dict:
    all_exp_files = list(os.listdir(output_dir_exp))

    result_file = None
    print(result_dir_exp)
    error_occurred = False
    for file in os.listdir(output_dir_exp):
        if file.startswith('error'):
            error_occurred = True

    for file in os.listdir(result_dir_exp):
        if file.startswith('results_final_'):
            result_file = os.path.join(result_dir_exp, file)
            break
    assert result_file or error_occurred, f'File {result_file} missing'

    if result_file:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        n_debugs_per_code = result_data['config']['agents']['code_generation']['args']['max_debug_steps'] + 1

    else:
        result_dir_str = str(result_dir_exp)
        assert not isinstance(result_dir_str, PosixPath)
        if 'full_5_3' in result_dir_str:
            n_debugs_per_code = 2
        elif 'full_3_6' in result_dir_str:
            n_debugs_per_code = 5
        elif 'full_no' in result_dir_str:
            n_debugs_per_code = 5
        elif 'baseline' in result_dir_str:
            n_debugs_per_code = 5
        elif 'silver' in result_dir_str:
            n_debugs_per_code = 5
        else:
            raise ValueError

    n_codes_per_run = n_debugs_per_code + 1

    ordered_code_files = []
    for file in all_exp_files:
        agent_type = file.split('_')[1]
        if agent_type == 'CodeGen':
            file_id = int(file.split('_')[0])
            if 'response_code' in file:
                ordered_code_files.append((file_id, file))

    ordered_code_files.sort()
    # n_generated_codes = len(ordered_code_files)

    ordered_codes_per_run = []
    ordered_codes_per_run_file_names = []
    while len(ordered_code_files) > n_codes_per_run:
        next_code_run = ordered_code_files[:n_codes_per_run]
        next_code_run_names = [ocf[1] for ocf in next_code_run]
        ordered_codes_per_run.append(next_code_run)
        ordered_codes_per_run_file_names.append(next_code_run_names)
        ordered_code_files = ordered_code_files[n_codes_per_run:]
    assert len(ordered_code_files) != 0
    ordered_codes_per_run.append(ordered_code_files)
    ordered_code_files_names = [ocf[1] for ocf in ordered_code_files]
    ordered_codes_per_run_file_names.append(ordered_code_files_names)

    n_generated_init_codes = len(ordered_codes_per_run_file_names)

    n_no_changes = 0

    all_runs_by_acc = defaultdict(list)
    current_best_acc = 0

    first_code_dict = None

    found_empty_file = False
    n_generated_codes = 0
    # Run each code on the validation tasks and get identify the best code based on the validation tasks
    for n_code_version, files_code_run in enumerate(ordered_codes_per_run_file_names):
        if found_empty_file:
            break
        prev_code = ''
        prev_val_metrics = None

        for debug_run, code_file in enumerate(files_code_run):
            if found_empty_file:
                break
            code_file_path = os.path.join(output_dir_exp, code_file)

            with open(code_file_path, 'r') as f:
                content = f.read()

            python_code = postprocess_response(model_response=content)
            #assert python_code != '', f'File {code_file_path} does not contain a python program'
            if python_code == '':
                benchmark_name, domain_name = flags.env.split('-')
                log_file = f'./data_analysis_results/runs_with_empty_outputs/{benchmark_name}_{domain_name}.json'
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log = json.load(f)
                else:
                    log = dict()
                log[code_file_path] = 'empty'
                with open(log_file, 'w') as f:
                    json.dump(log, f, indent=2)
                found_empty_file = True

            elif python_code == prev_code:
                n_no_changes += 1
                val_metrics = prev_val_metrics
            else:
                val_metrics = run_evaluation(eval_tasks=validation_tasks,
                                             generalized_plan_code=python_code,
                                             flags=flags)

            if found_empty_file:
                acc = 0
                val_metrics = dict()
            else:
                acc = val_metrics['accuracy']

            current_code_dict = {
                'accuracy_val': acc,
                'n_init_code': n_code_version + 1,
                'n_debug': debug_run,
                'val_metrics': val_metrics,
                'python_code': python_code
            }
            all_runs_by_acc[acc].append(current_code_dict)
            if acc > current_best_acc:
                current_best_acc = acc

            if first_code_dict is None:
                first_code_dict = deepcopy(current_code_dict)

            n_generated_codes += 1

    last_code_dict = deepcopy(current_code_dict)
    last_code_step = (last_code_dict['n_init_code'], last_code_dict['n_debug'])
    first_code_step = (1, 0)

    best_code_dict_first = None
    best_code_dict_last = None

    all_best_codes = all_runs_by_acc[current_best_acc]
    current_best_last_step = (-1, -1)
    current_best_first_step = (100, 100)

    for best_code in all_best_codes:
        n_init_code = best_code['n_init_code']
        n_debug = best_code['n_debug']

        if n_init_code < current_best_first_step[0]:
            current_best_first_step = (n_init_code, n_debug)
            best_code_dict_first = deepcopy(best_code)
        elif n_init_code == current_best_first_step[0]:
            if n_debug < current_best_first_step[1]:
                current_best_first_step = (n_init_code, n_debug)
                best_code_dict_first = deepcopy(best_code)

        if n_init_code > current_best_last_step[0]:
            current_best_last_step = (n_init_code, n_debug)
            best_code_dict_last = deepcopy(best_code)
        elif n_init_code == current_best_last_step[0]:
            if n_debug > current_best_last_step[1]:
                current_best_last_step = (n_init_code, n_debug)
                best_code_dict_last = deepcopy(best_code)

    assert best_code_dict_last is not None and best_code_dict_first is not None

    # run the first, last and two best on the eval data
    eval_metrics_first = None
    if flags.eval_first:
        eval_metrics_first = run_evaluation(eval_tasks=eval_tasks,
                                            generalized_plan_code=first_code_dict['python_code'],
                                            flags=flags)
    first_code_dict['eval_metrics'] = eval_metrics_first

    eval_metrics_last = None
    if flags.eval_last:
        eval_metrics_last = run_evaluation(eval_tasks=eval_tasks,
                                           generalized_plan_code=last_code_dict['python_code'],
                                           flags=flags)
    last_code_dict['eval_metrics'] = eval_metrics_last

    eval_metrics_best_last = None

    assert flags.eval_best_last
    if flags.eval_best_last:
        print('RUNNING ON EVAL TASKS')
        # Do not re-run if identical
        if eval_metrics_first is not None and first_code_step == current_best_last_step:
            eval_metrics_best_last = first_code_dict['eval_metrics']
        elif eval_metrics_last is not None and last_code_step == current_best_last_step:
            eval_metrics_best_last = last_code_dict['eval_metrics']
        else:
            print('RUNNING ON EVAL TASKS')
            eval_metrics_best_last = run_evaluation(eval_tasks=eval_tasks,
                                                    generalized_plan_code=best_code_dict_last['python_code'],
                                                    flags=flags)

    assert eval_metrics_best_last is not None
    best_code_dict_last['eval_metrics'] = eval_metrics_best_last

    eval_metrics_best_first = None
    #print(best_code_dict_last['eval_metrics'])
    if flags.eval_best_first:
        print(flags.eval_best_first)
        # Do not re-run if identical
        if eval_metrics_first is not None and first_code_step == current_best_first_step:
            eval_metrics_best_first = first_code_dict['eval_metrics']
        elif eval_metrics_last is not None and last_code_step == current_best_first_step:
            eval_metrics_best_first = last_code_dict['eval_metrics']
        else:
            eval_metrics_best_first = run_evaluation(eval_tasks=eval_tasks,
                                                     generalized_plan_code=best_code_dict_first['python_code'],
                                                     flags=flags)

    best_code_dict_first['eval_metrics'] = eval_metrics_best_first

    overview_dict = {
        'n_codes_total': n_generated_codes,
        'n_init_codes': n_generated_init_codes,
        'best_code_step_first': current_best_first_step,
        'best_code_step_last': current_best_last_step,
        'last_code_step': (last_code_dict['n_init_code'], last_code_dict['n_debug']),
        'best_accuracy_val': current_best_acc,
        'debugs_wo_changes': n_no_changes
    }

    if found_empty_file or error_occurred:
        error_d = {'accuracy': 0.0}
    else:
        error_d = None

    return_dict = {
        "overview": overview_dict,
        "last_result": last_code_dict,
        "best_result_first": best_code_dict_first,
        "best_result_last": best_code_dict_last,
        "first_result": first_code_dict,
        "last_result_incl_error": error_d
    }

    return return_dict


def run_all_evaluations_all_codes_json(benchmark_name: str,
                                       domain_name: str,
                                       flags: Namespace,
                                       output_file: str = ''):

    print('RUNNING CORRECT EVAL')

    eval_split_names = flags.eval_split_names
    eval_split_file = flags.split_file
    val_split_file = flags.val_split_file
    val_split_name = flags.val_split_name

    input_version = flags.input_version
    experiments_output_folder = flags.experiments_output_folder
    experiments_results_folder = flags.experiments_results_folder
    experiments_names = flags.experiments_names

    if flags.eval_env is not None:
        eval_benchmark_name = flags.eval_env.split('-')[0]
        eval_domain_name = flags.eval_env.split('-')[-1]
    else:
        eval_benchmark_name = benchmark_name
        eval_domain_name = domain_name

    eval_tasks_per_split = dict()
    for eval_name in eval_split_names:

        tasks = create_eval_tasks(benchmark_name=eval_benchmark_name,
                                  domain_name=eval_domain_name,
                                  eval_split_file=eval_split_file,
                                  eval_split_name=eval_name,
                                  input_version=input_version)
        eval_tasks_per_split[eval_name] = tasks

    validation_tasks = create_val_tasks(benchmark_name=benchmark_name,
                                        domain_name=domain_name,
                                        val_split_file=val_split_file,
                                        val_split_name=val_split_name,
                                        input_version=input_version)

    for exp_name in experiments_names:
        output_sub_dir = Path(experiments_output_folder) / exp_name / f'{benchmark_name}-{domain_name}'
        result_sub_dir = Path(experiments_results_folder) / exp_name / f'{benchmark_name}-{domain_name}'

        for time_stamped_folder in os.listdir(result_sub_dir):
            if time_stamped_folder.startswith('.'):
                continue
            t_stamped_out_path = output_sub_dir / str(time_stamped_folder)
            t_stamped_res_path = result_sub_dir / str(time_stamped_folder)
            for seed_folder in os.listdir(t_stamped_res_path):

                if flags.seeds is not None:
                    if seed_folder not in flags.seeds:
                        continue

                print(f'Running on {seed_folder}')
                seed_out_path = t_stamped_out_path / str(seed_folder)
                seed_res_path = t_stamped_res_path / str(seed_folder)

                results_all_eval_splits = dict()

                for eval_name, eval_tasks in eval_tasks_per_split.items():
                    eval_dict = run_evaluation_data_split_all_codes(output_dir_exp=seed_out_path,
                                                                    result_dir_exp=seed_res_path,
                                                                    validation_tasks=validation_tasks,
                                                                    eval_tasks=eval_tasks, flags=flags)

                    results_all_eval_splits[eval_name] = convert_path2str(eval_dict)

                if output_file == '':
                    output_file_path = os.path.join(seed_res_path, 'additional_eval_data.json')
                else:
                    output_file_path = os.path.join(seed_res_path, output_file)
                with open(output_file_path, 'w') as f:
                    json.dump(results_all_eval_splits, f, indent=2)


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


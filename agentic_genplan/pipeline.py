import os
import sys
import json
import time
from typing import Dict, List
from copy import copy
from utils.helper import convert_path2str, create_ordered_results_data
from utils.tasks import TaskData, create_tasks_dict
from utils.paths import create_log_dirname, get_res_file_name, get_benchmark_dir, get_llm_gen_dir
from agentic_genplan.create_tasks import create_tasks
from agents.gen_plan_execution import GeneralizedPlan
from evaluation.evaluation import run_evaluation
from agentic_genplan.create_agent_params import *

from agents.agent_code_gen import AgentCodeGeneration
from agents.agent_code_gen_basic import AgentCodeGenerationBasic
from agents.agent_code_gen_multiple import AgentCodeGenerationMultiple
from agents.agent_code_gen_basic_multiple import AgentCodeGenerationBasicMultiple
from agents.agent_description_gen import AgentDescriptionGen, create_copy_for_new_tasks
from agents.agent_description_gen_basic import AgentDescriptionGenBasic
from agents.agent_strategy_gen import AgentStrategyGen
from agents.agent_strategy_gen_w_debugging_prev_hist import AgentStrategyGenDebuggingPlanGenHist
from agents.agent_strategy_val_planbased import AgentStrategyValidatePlanBased


def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


def run_pipeline(seed_value: int,
                 p_version: str):

    pipeline_started = time.time()

    experiment_id = FLAGS.experiment_id
    time_stamped_folder = create_log_dirname(flags=FLAGS)

    # Output directory for the LLM outputs
    domain_out_dir = os.path.join(experiment_id, FLAGS.env)
    seed_out_dir = Path(FLAGS.save_path) / domain_out_dir / time_stamped_folder / f'seed_{seed_value}'
    os.makedirs(seed_out_dir, exist_ok=True)

    # Output directory for the results
    res_out_dir = Path(FLAGS.results_dir) / domain_out_dir / time_stamped_folder / f'seed_{seed_value}'
    res_out_dir.mkdir(exist_ok=True, parents=True)

    benchmark_name, domain_name = FLAGS.env.split('-')
    benchmark_dir = get_benchmark_dir(benchmark_name=benchmark_name, domain_name=domain_name)
    print(benchmark_dir)

    llm_gen_dir = get_llm_gen_dir(
            output_dir=FLAGS.save_path,
            benchmark_name=benchmark_name,
            domain_name=domain_name
        )

    print('--- Getting the Tasks ---')
    # get the tasks
    prompt_tasks, add_validate_tasks, eval_tasks, add_debug_tasks_names = create_tasks(
        env_name=FLAGS.env,
        flags=FLAGS
    )

    prompt_task_dict = create_tasks_dict(tasks=prompt_tasks)
    agent_configs = FLAGS.agents
    flags_dict = deepcopy(FLAGS.__dict__)
    # set the directory for logging
    for agent_type in agent_configs.keys():
        agent_configs[agent_type]['args']['log_dir'] = seed_out_dir

    print('--- Creating NL Descriptions for prompt tasks ---')
    # call AgentDescriptionGen
    ag_descript_conf = agent_configs['description_generation']
    agent_param = create_agent_description_gen(agent_config=ag_descript_conf,
                                               task_dict=prompt_task_dict,
                                               flags_dict=flags_dict)
    agent_description_generation: AgentDescriptionGen = str_to_class(ag_descript_conf['name'])(
        **agent_param)
    agent_description_generation.generate_descriptions()

    assert prompt_task_dict == agent_description_generation.task_dict

    print('--- Creating NL Descriptions for validation tasks ---')
    val_descript_seed_out = seed_out_dir / 'validation_task_descriptions'
    val_descript_seed_out.mkdir(exist_ok=True, parents=True)
    agent_description_valtasks: AgentDescriptionGen = str_to_class(ag_descript_conf['name'])(
        **agent_param)
    agent_description_valtasks.domain_description = agent_description_generation.domain_description
    agent_description_valtasks.log_dir = val_descript_seed_out
    agent_description_valtasks.increase_counter = False
    agent_description_valtasks.include_plans = False
    # overwrite with validation tasks and already generated domain descriptions
    agent_description_valtasks = create_copy_for_new_tasks(
        orig_agent=agent_description_generation,
        new_agent=agent_description_valtasks
    )
    agent_description_valtasks.generate_descriptions_additional_tasks(
        tasks=add_validate_tasks
    )
    additional_validation_task_dict = agent_description_valtasks.task_dict

    # create a set of the prompt and additional validation tasks to use for the
    # intermediate validations; this will not include any llm-generated tasks but stay fixed
    intermediate_validation_task_dict = dict()
    for key, val in additional_validation_task_dict.items():
        intermediate_validation_task_dict[key] = val
    for key, val in prompt_task_dict.items():
        intermediate_validation_task_dict[key] = val

    if p_version == 'qual_check':
        acc_last_qual_check = run_check_strat_pipeline(
            agent_configs=agent_configs,
            prompt_task_dict=prompt_task_dict,
            validation_task_dict=intermediate_validation_task_dict,
            flags_dict=flags_dict,
            llm_gen_dir=llm_gen_dir,
            res_dir=res_out_dir
        )
    else:
        raise NotImplementedError

    # add all prompt tasks to the validation tasks
    validation_task_dict = dict()
    for task_name, task_data in prompt_task_dict.items():
        validation_task_dict[task_name] = task_data
    for task_name, task_data in additional_validation_task_dict.items():
        validation_task_dict[task_name] = task_data

    # run the code (and pseudocode) generation
    ag_code_gen_conf = agent_configs.get('code_generation', None)

    debug_tasks_names = copy(add_debug_tasks_names)
    if FLAGS.stop_on_val:
        debug_tasks_names.extend(list(validation_task_dict.keys()))
    else:
        for prompt_t in prompt_tasks:
            debug_tasks_names.append(prompt_t.task_name)
    debug_tasks_names = list(set(debug_tasks_names))

    if ag_code_gen_conf is not None:
        agent_code_generation = run_code_generation(
            ag_code_gen_conf=ag_code_gen_conf,
            debug_tasks_names=debug_tasks_names,
            validation_task_dict=validation_task_dict,
            flags_dict=flags_dict,
            res_dir=res_out_dir,
            llm_gen_dir=llm_gen_dir
        )

        codegen_time = time.time() - pipeline_started
        eval_started = time.time()

        debug_metrics = {
            "used_debugging_attempts": agent_code_generation.required_debug_steps,
            "debugs_code": agent_code_generation.debugs_code,
            "debugs_pseudocode": agent_code_generation.debugs_pseudocode,
            "tasks_used_in_debugging": agent_code_generation.tasks_used_for_debugging,
            "step_best_validator": agent_code_generation.step_best_validator,
            "step_best_debug_run": agent_code_generation.best_run
        }

        if flags_dict['eval_last_code']:
            final_generalized_plan_code: GeneralizedPlan = agent_code_generation.last_validator_validate.generalized_plan
            out_data_last = run_evaluation_genplan(
                generalized_plan_code=final_generalized_plan_code,
                validation_task_dict=validation_task_dict,
                eval_tasks=eval_tasks
            )
        else:
            out_data_last = dict()

        out_data = deepcopy(out_data_last)
        out_data["config"] = flags_dict
        out_data['validation_results'] = agent_code_generation.validation_results
        out_data["debug_metrics"] = debug_metrics

        if flags_dict['eval_first_code']:
            # if the last code is also the first code -> do not re-run the evaluation
            if agent_code_generation.step_last_validator == 0 and flags_dict['eval_last_code']:
                first_out_data = deepcopy(out_data_last)
            else:
                first_generalized_plan_code: GeneralizedPlan = agent_code_generation.first_validator_validate.generalized_plan
                first_out_data = run_evaluation_genplan(
                    generalized_plan_code=first_generalized_plan_code,
                    validation_task_dict=validation_task_dict,
                    eval_tasks=eval_tasks
                )
            first_out_data['config'] = flags_dict
            first_out_data = convert_path2str(first_out_data)
            first_out_data = create_ordered_results_data(first_out_data)
            res_file_first = res_out_dir / f'{get_res_file_name(flags=FLAGS, version="first")}.json'
            with open(res_file_first, 'w') as f:
                json.dump(first_out_data, f, indent=2)

        if flags_dict['eval_best_code']:
            # If the best code is the last code -> do not re-run the evaluation
            if agent_code_generation.step_best_validator == agent_code_generation.step_last_validator and flags_dict['eval_last_code']:
                best_out_data = deepcopy(out_data_last)
            # If the best code is the first code -> do not re-run the evaluation
            elif agent_code_generation.step_best_validator == 0 and flags_dict['eval_first_code']:
                best_out_data = deepcopy(first_out_data)
            else:
                best_generalized_plan_code: GeneralizedPlan = agent_code_generation.best_validator_validate.generalized_plan
                best_out_data = run_evaluation_genplan(
                    generalized_plan_code=best_generalized_plan_code,
                    validation_task_dict=validation_task_dict,
                    eval_tasks=eval_tasks
                )
            best_out_data["config"] = flags_dict
            best_out_data = convert_path2str(best_out_data)
            best_out_data = create_ordered_results_data(best_out_data)
            res_file_best = res_out_dir / f'{get_res_file_name(flags=FLAGS, version="best")}.json'
            with open(res_file_best, 'w') as f:
                json.dump(best_out_data, f, indent=2)

        eval_time = time.time() - eval_started

    else:
        val_tasks = [td.task for td in validation_task_dict.values()]
        val_task_names = [task.task_name for task in val_tasks]
        out_data = {
            "config": flags_dict,
            "validation_tasks": val_task_names,
        }
        codegen_time = None
        eval_time = None

    out_data["acc_last_qual_check"] = acc_last_qual_check
    out_data = convert_path2str(out_data)
    out_data = create_ordered_results_data(out_data=out_data)

    res_out_file = res_out_dir / f'{get_res_file_name(flags=FLAGS, version="final")}.json'
    print(f'------ Saving Results to {res_out_file} ------')
    with open(res_out_file, 'w') as f:
        json.dump(out_data, f, indent=2)

    prob_out_file = res_out_dir / f'all_validate_problems.json'
    print(f'------ Saving problems and plans to {prob_out_file} ---------')
    save_val_problems(val_taskdata_dict=validation_task_dict,
                      out_file_path=prob_out_file)

    # Save the descriptions and final strategy
    nl_out_file = res_out_dir / f'final_nl_descriptions.json'
    nl_data = dict()
    for n, (task_name, task_data) in enumerate(list(validation_task_dict.items())):
        if n == 0:
            nl_data['strategy'] = task_data.get_strategy_domain()
            nl_data['domain_description'] = task_data.get_domain_nl()

        nl_data[task_name] = {'t_strategy': task_data.get_strategy_task(),
                              't_description': task_data.get_problem_nl()}

    with open(nl_out_file, 'w') as f:
        json.dump(nl_data, f)

    total_time = time.time() - pipeline_started
    time_log = res_out_dir / 'run_time.csv'
    with open(time_log, 'w') as f:
        f.write(f'Total time:\t{round(total_time, 5)}\n')
        if codegen_time is not None:
            f.write(f'Time until final code generated:\t{round(codegen_time, 5)}\n')
        if eval_time is not None:
            f.write(f'Time for evaluation:\t{round(eval_time, 5)}\n')


def run_code_generation(ag_code_gen_conf: dict,
                        debug_tasks_names: List[str],
                        validation_task_dict: Dict[str, TaskData],
                        flags_dict: dict,
                        res_dir: Path,
                        llm_gen_dir: Path) -> AgentCodeGenerationBasic:

    print('--- Running code generation with debugging ---')
    # call AgentCodeGeneration
    agent_param = create_agent_codegen(agent_config=ag_code_gen_conf,
                                       debug_task_names=debug_tasks_names,
                                       validation_task_dict=validation_task_dict,
                                       flags_dict=flags_dict,
                                       res_dir=res_dir,
                                       llm_gen_dir=llm_gen_dir)
    agent_code_generation: AgentCodeGenerationBasic = str_to_class(ag_code_gen_conf['name'])(**agent_param)
    agent_code_generation.run_complete_generation_loop()

    return agent_code_generation


def run_evaluation_genplan(generalized_plan_code: GeneralizedPlan,
                           validation_task_dict: Dict[str, TaskData],
                           eval_tasks: List[Task]):

    print('--- Running evaluation ---')
    # run evaluation on eval test set

    evaluation_results = run_evaluation(
        eval_tasks=eval_tasks,
        generalized_plan_code=generalized_plan_code.code_str,
        flags=FLAGS
    )

    # run evaluation on validation tasks
    val_tasks = [td.task for td in validation_task_dict.values()]
    validation_results = run_evaluation(
        eval_tasks=val_tasks,
        generalized_plan_code=generalized_plan_code.code_str,
        flags=FLAGS
    )

    eval_task_names = []
    val_task_names = []
    for task in eval_tasks:
        eval_task_names.append(str(task.task_name))
    for task in val_tasks:
        val_task_names.append(task.task_name)

    eval_results = {
        'eval_data': {'n_correct': evaluation_results['successes'],
                      'accuracy': evaluation_results['accuracy']},
        'val_data': {'n_correct': validation_results['successes'],
                     'accuracy': validation_results['accuracy']}
    }

    outdata = {
        "eval_tasks": eval_task_names,
        "eval_metrics": evaluation_results,
        "validation_tasks": val_task_names,
        "validation_metrics": validation_results,
        "all_results_code": eval_results
    }

    return outdata


def save_val_problems(val_taskdata_dict: Dict[str, TaskData],
                      out_file_path: Path):

    data2save = dict()

    for task_name, task_data in val_taskdata_dict.items():

        if task_data.task.optimal_plan is None:
            symbolic_plan = 'None'
        else:
            symbolic_plan = '\n'.join(task_data.task.optimal_plan)
        task_dict = {'problem_str': task_data.task.problem_str,
                     'symbolic_plan': symbolic_plan
                     }
        if task_data.generated_plan is None:
            task_dict['llm_gen_plan'] = 'None'
        else:
            task_dict['llm_gen_plan'] = '\n'.join(task_data.generated_plan)

        data2save[task_name] = task_dict

    with open(out_file_path, 'w') as f:
        json.dump(data2save, f, indent=2)


def run_original_pipeline(agent_configs: dict,
                          prompt_task_dict: dict[str, TaskData],
                          validation_task_dict: dict[str, TaskData],
                          flags_dict: dict,
                          llm_gen_dir: Path,
                          res_dir: Path):

    # call AgentStrategyGen
    ag_strat_gen_conf = agent_configs['strategy_generation']
    agent_param = create_agent_strategy_gen(agent_config=ag_strat_gen_conf,
                                            prompt_task_dict=prompt_task_dict,
                                            validation_task_dict=validation_task_dict,
                                            flags_dict=flags_dict,
                                            llm_gen_dir=llm_gen_dir,
                                            res_dir=res_dir)

    agent_strategy_generation: AgentStrategyGen = str_to_class(ag_strat_gen_conf['name'])(**agent_param)
    agent_strategy_generation.generate_domain_strategy()

    assert prompt_task_dict == agent_strategy_generation.prompt_task_dict


def run_check_strat_pipeline(agent_configs: dict,
                             prompt_task_dict: dict[str, TaskData],
                             validation_task_dict: dict[str, TaskData],
                             flags_dict: dict,
                             llm_gen_dir: Path,
                             res_dir: Path) -> Union[None, float]:

    ag_strat_plan_based_val_conf = agent_configs['strategy_final_validation']
    plan_gen_config = agent_configs.get('plan_generation', None)
    max_iterations = ag_strat_plan_based_val_conf['max_iter']

    accuracy_last_iteration = None

    for i in range(max_iterations):
        run_original_pipeline(agent_configs=agent_configs,
                              prompt_task_dict=prompt_task_dict,
                              validation_task_dict=validation_task_dict,
                              flags_dict=flags_dict,
                              llm_gen_dir=llm_gen_dir,
                              res_dir=res_dir)

        # Run on prompt tasks
        ag_strat_plan_based_val_conf['plan_gen_args'] = plan_gen_config
        agent_param = create_agent_strat_val_planbased(
            agent_config=ag_strat_plan_based_val_conf,
            task_dict=prompt_task_dict,
            flags_dict=flags_dict,
            llm_gen_dir=llm_gen_dir,
            res_dir=res_dir
        )
        agent_strat_val_planbased: AgentStrategyValidatePlanBased = str_to_class(
            ag_strat_plan_based_val_conf['name']
        )(**agent_param)

        reached_quality_threshold, accuracy = agent_strat_val_planbased.assess_strategy_quality()
        accuracy_last_iteration = accuracy

        if not reached_quality_threshold:
            print(f'Did not reach quality threshold')
            raise NotImplementedError

    return accuracy_last_iteration


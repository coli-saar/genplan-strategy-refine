import os
import json
import random
from argparse import Namespace
from typing import Tuple, List, Dict, Union
from pathlib import Path
from utils.tasks import Task
from utils.paths import get_benchmark_dir
from utils.helper import run_symbolic_planner


"""
Create PDDL prompting, training, and evaluation tasks. 
And plans for the prompt and debugging tasks if not available yet.
"""


def create_tasks(
        env_name: str,
        flags: Namespace,
        sort_by_size: bool = False) -> Tuple[List[Task], List[Task], List[Task], List[str]]:
    """
    Split the available dataset into
    - validation tasks
       - prompt tasks
       - additional_debug_tasks
       - additional_validation tasks
    - evaluation tasks
    """

    benchmark_name, domain_name = env_name.split('-')
    input_version = flags.__dict__.get('input_version', 'pddl')

    num_prompt = flags.num_prompt_tasks
    num_debug = flags.num_debug_tasks
    num_validation = flags.num_validation_tasks # this includes the debug but not the prompt tasks
    num_eval = flags.num_eval_tasks

    example_tasks, validate_tasks, eval_tasks, add_debug_names = _get_custom_tasks_seed(
        benchmark_name=benchmark_name,
        domain_name=domain_name,
        num_prompt=num_prompt,
        num_debug=num_debug,
        num_validate=num_validation,
        num_eval=num_eval,
        input_version=input_version,
        sort_by_size=sort_by_size,
        flags=flags)

    return example_tasks, validate_tasks, eval_tasks, add_debug_names


def _get_custom_tasks_seed(benchmark_name: str,
                           domain_name: str,
                           num_prompt: int,
                           num_debug: int,
                           num_validate: int,
                           num_eval: Union[int, None],
                           flags: Namespace,
                           input_version: str = 'pddl',
                           sort_by_size: bool = False) -> Tuple[List[Task], List[Task], List[Task], List[str]]:
    """

    :param benchmark_name:
    :param domain_name:
    :param num_prompt: number of prompt tasks
    :param num_debug: number of tasks for debugging
    :param num_validate: number of tasks for validating the generalized plans
    :param num_eval: number of eval tasks
    :param flags:
    :param input_version: version of the input, i.e. either 'pddl' or 'nl'
                          or 'pddl_objNL' if PDDL input but with adapted object names
    :return:
    """

    benchmark_dir = get_benchmark_dir(benchmark_name=benchmark_name,
                                      domain_name=domain_name)
    print(benchmark_dir)
    domain_file_path = benchmark_dir / 'domain.pddl'
    assert os.path.exists(domain_file_path), f'Domain not found: {domain_file_path}'

    if input_version == 'pddl':
        instance_dir = benchmark_dir / 'problems_pddl'
        plans_dir = benchmark_dir / 'plans_pddl'
    elif input_version == 'nl' or input_version == 'pddl_objNL':
        instance_dir = benchmark_dir / 'problems_pddl_nl_objs'
        plans_dir = benchmark_dir / 'plans_pddl_nl_objs'
    else:
        raise ValueError

    if num_eval is None:
        all_files = list(os.listdir(instance_dir))
        n_problems = len(all_files)
        num_eval = n_problems - num_validate - num_prompt

    Path(plans_dir).mkdir(exist_ok=True, parents=True)

    # Load the problems and create the splits
    current_seed = flags.seed
    seed_split_path = benchmark_dir / 'seed_splits.json'

    seed_num_key = (current_seed, num_prompt, num_validate, num_eval)
    # json doesn't allow tuples as keys -> convert into a string
    seed_num_key_str = '-'.join([str(k) for k in seed_num_key])

    data_splits = None

    # If there exists already a file with split information, check whether the current split-seed configuration had already been created / defined
    if os.path.exists(seed_split_path):
        with open(seed_split_path, 'r', encoding='utf-8') as f:
            seed_split_maps = json.load(f)
        data_splits = seed_split_maps.get(seed_num_key_str, None)
    else:
        seed_split_maps = dict()

    if data_splits is None:
        # If there is no information about that particular split-seed combination then create it

        data_splits = create_random_splits(
            benchmark_dir=benchmark_dir,
            num_prompt=num_prompt,
            num_validate=num_validate,
            num_eval=num_eval,
            sort_by_size=sort_by_size,
            flags=flags,
            instance_dir=instance_dir
        )
        seed_split_maps[seed_num_key_str] = data_splits
        # Update split file
        with open(seed_split_path, 'w', encoding='utf-8') as f:
            json.dump(seed_split_maps, f, indent=2)

    prompt_tasks = create_tasks_split(
        instance_dir=instance_dir,
        instance_names=data_splits['prompt'],
        domain_file_path=domain_file_path)

    validate_split_name = flags.name_val_split
    validation_tasks = create_tasks_split(
        instance_dir=instance_dir,
        instance_names=data_splits[validate_split_name],
        domain_file_path=domain_file_path)

    eval_split_name = flags.name_eval_split
    eval_tasks = create_tasks_split(
        instance_dir=instance_dir,
        instance_names=data_splits[eval_split_name],
        domain_file_path=domain_file_path)

    # Read in or generate the optimal plans for the prompt and debug tasks
    get_plans(tasks=prompt_tasks, plan_dir=plans_dir)

    random.seed(flags.seed)
    if num_debug > num_prompt:
        num_add_debug = num_debug - num_prompt
        add_debug_tasks = random.sample(validation_tasks, num_add_debug)
        get_plans(tasks=add_debug_tasks, plan_dir=plans_dir)
        add_debug_tasks_names = [task.name for task in add_debug_tasks]
    else:
        add_debug_tasks_names = []

    return prompt_tasks, validation_tasks, eval_tasks, add_debug_tasks_names


def get_plans(tasks: List[Task], plan_dir: Path):

    for task in tasks:
        task_name = task.task_name
        plan_file = plan_dir / f'{task_name}_plan.txt'
        if not os.path.exists(plan_file):
            plan = run_symbolic_planner(task=task,
                                        plan_file=str(plan_file))
        else:
            plan = []
            with open(plan_file, 'r') as pf:
                for line in pf.readlines():
                    line = line.strip()
                    if ';' in line:
                        line = line[:line.index(';')]
                    plan.append(line)

        plan = [ac for ac in plan if ac != '']

        if len(plan) > 0:

            assert plan[0].startswith('(')

        task.set_optimal_plan(plan=plan)


def create_tasks_split(instance_dir,
                       instance_names: list,
                       domain_file_path: str) -> List[Task]:
    """
    Create a list of Task objects for all the specified instances
    :param instance_dir:
    :param instance_names:
    :param domain_file_path:
    :return:
    """
    task_list = []
    for inst in instance_names:
        instance_path = instance_dir / inst
        task = Task(domain_file_path=domain_file_path,
                    problem_file_path=instance_path)
        task_list.append(task)

    return task_list


def create_random_splits(benchmark_dir,
                         instance_dir,
                         num_prompt: int,
                         num_validate: int,
                         num_eval: int,
                         flags: Namespace,
                         sort_by_size: bool = False) -> Dict[str, List[str]]:
    """
    Creates a pseudo-random data split by shuffling all the data instances using the current seed
    Then the first num_prompt instances are selected as prompt examples, the next num_train instances as train instances and next num_eval instances for evaluation
    :param benchmark_dir:
    :param instance_dir:
    :param num_prompt:
    :param num_validate:
    :param num_eval:
    :param flags:
    :param sort_by_size:
    :return: a dictionary of the form
            {"prompt": [list of all prompt file names],
             "validate": [list of all train ...],
             "eval": [...]}
    """
    all_instances = []
    for file in os.listdir(instance_dir):
        if file.endswith('.pddl'):
            all_instances.append(file)

    random.seed(flags.seed)
    if sort_by_size:
        domain_file_path = benchmark_dir / 'domain.pddl'
        all_tasks: List[Task] = create_tasks_split(instance_dir=instance_dir,
                                                   instance_names=all_instances,
                                                   domain_file_path=domain_file_path)
        all_tasks = sorted(all_tasks, key=lambda t: t.size)
        all_instances = [t.problem_file_name for t in all_tasks]
    else:
        random.shuffle(all_instances)

    total_task_num = num_prompt + num_validate + num_eval
    assert len(all_instances) >= num_prompt + num_validate + num_eval, f'There are not enough instances available for {num_prompt} prompt, {num_validate} validation and {num_eval} eval tasks. Only {len(all_instances)} tasks are available and {total_task_num} are needed.'

    # Split the dataset
    example_tasks = all_instances[:num_prompt]
    validation_tasks = all_instances[num_prompt:(num_prompt + num_validate)]
    eval_tasks = all_instances[(num_prompt + num_validate):total_task_num]

    data_splits = {
        "prompt": example_tasks,
        "validate": validation_tasks,
        "eval": eval_tasks
    }
    return data_splits



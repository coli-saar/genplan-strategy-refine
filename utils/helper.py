import os
import inspect
import subprocess
import json
import re
import random
from argparse import Namespace
from pathlib import Path
from typing import Set, List, Union, Dict, Tuple
from collections import OrderedDict
from jinja2 import Template
from llm_models.llm_base_class import LLMModel
from utils.tasks import Task
from utils.paths import get_timestamp_for_log, VAL, FASTDOWNWARD, PROMPT_DIR, TEMP_DIR

os.environ['FAST_DOWNWARD'] = FASTDOWNWARD
os.environ['VAL'] = VAL


# TODO
def parse_strategy_response(model_response):
    parts = model_response.split('"""')
    if len(parts) == 3 and 'Strategy:' in parts[1]:
        strategy = parts[1].replace('Strategy:', '')
        strategy = strategy.strip()
        return strategy
    elif len(parts) == 1:
        parts = model_response.split('```')
        if len(parts) == 3 and 'Strategy:' in parts[1]:
            strategy = parts[1].replace('Strategy:', '')
            strategy = strategy.strip()
            return strategy

    return model_response


def create_prompt_template(prompt_file, flags) -> Union[Template, None]:
    """

    :param prompt_file: path to the prompt file relative to the prompts directory
    :return:
    """
    if prompt_file is None or prompt_file == '':
        return None

    if flags['prompt_dir'] is not None:
        prompt_file_path = os.path.join(flags['prompt_dir'], prompt_file)
    else:
        prompt_file_path = PROMPT_DIR / prompt_file
    with open(prompt_file_path, 'r', encoding='utf-8') as pf:
        template = Template(pf.read())

    return template


def log_agent_system_prompt(log_dir: Path,
                            llm_model: LLMModel,
                            agent_name: str):

    log_dir.mkdir(exist_ok=True, parents=True)
    system_prompt = llm_model.get_initial_prompt()

    system_prompt_log_file = log_dir / f'0_prompt_{agent_name}.txt'

    # make sure not to overwrite in case there are several agents with the same name
    counter = 1
    while system_prompt_log_file.exists():
        system_prompt_log_file = log_dir / f'0_{counter}_prompt_{agent_name}.txt'
        counter += 1

    with open(system_prompt_log_file, 'w') as f:
        f.write(system_prompt)


def log_llm_interaction(log_dir: Path,
                        llm_model: LLMModel,
                        agent_name: str,
                        flags: dict,
                        python_code: bool = False):
    """

    :param log_dir:
    :param llm_model:
    :param agent_name:
    :param flags:
    :param python_code: whether
    :return:
    """

    counter = flags['llm_counter']

    log_llm_interaction_for_counter(log_dir=log_dir,
                                    llm_model=llm_model,
                                    agent_name=agent_name,
                                    counter=counter,
                                    python_code=python_code)

    flags['llm_counter'] += 1


def log_llm_interaction_validation_counter(log_dir: Path,
                                           llm_model: LLMModel,
                                           agent_name: str,
                                           flags: dict):
    counter = flags['llm_counter'] - 1

    additional_counter = 1

    file_path = log_dir / f'{counter}_{additional_counter}_{agent_name}_prompt.txt'

    while file_path.exists():
        file_path = log_dir / f'{counter}_{additional_counter}_{agent_name}_prompt.txt'
        if not file_path.exists():
            break
        additional_counter += 1

    new_counter = f'{counter}_{additional_counter}'
    log_llm_interaction_for_counter(log_dir=log_dir,
                                    llm_model=llm_model,
                                    agent_name=agent_name,
                                    counter=new_counter,
                                    python_code=False)


def log_llm_interaction_for_counter(log_dir: Path,
                                    llm_model: LLMModel,
                                    agent_name: str,
                                    counter: [int, str],
                                    python_code: bool = False):
    history_log_file = log_dir / f'all_model_histories.jsonlines'

    prompt_log_file = log_dir / f'{counter}_{agent_name}_prompt.txt'
    if python_code:
        response_log_file = log_dir / f'{counter}_{agent_name}_response_code.txt'
    else:
        response_log_file = log_dir / f'{counter}_{agent_name}_response.txt'

    if os.path.exists(prompt_log_file) or os.path.exists(response_log_file):
        raise FileExistsError(f'{prompt_log_file} already exists')

    model_hist = llm_model.get_history()
    last_prompt = model_hist[-2]['content']
    last_response = model_hist[-1]['content']

    print(f'{last_prompt}\n\n\n\n{last_response}')

    # Write to prompt log
    with open(prompt_log_file, 'w') as f:
        f.write(last_prompt)

    # Write to response log
    with open(response_log_file, 'w') as f:
        f.write(last_response)

    # Log the full model history
    log_hist = {'response_file': f'{response_log_file}',
                'model_history': model_hist}

    with open(history_log_file, 'a') as f:
        json.dump(log_hist, f)
        f.write('\n')

    if llm_model.reasoning_model:
        reasoning_log = log_dir / f'all_reasoning_outputs.jsonlines'
        log_reasoning = {'response_file': f'{response_log_file}',
                         'reasoning': llm_model.last_reasoning}
        with open(reasoning_log, 'a') as f:
            json.dump(log_reasoning, f)
            f.write('\n')


def get_var_name(var):
    """
    From: https://www.geeksforgeeks.org/get-variable-name-as-string-in-python/
    :param var:
    :return:
    """
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame)[1]
    local_vars = caller_frame.frame.f_locals

    for name, value in local_vars.items():
        if value is var:
            return name


class VariableContainer:
    """
    From: https://www.geeksforgeeks.org/convert-string-into-variable-name-in-python/
    """
    def __init__(self):
        self.variables = {}

    def add_variable(self, name, value):
        self.variables[name] = value

    def get_variable(self, name):
        return self.variables.get(name)


def set_to_reproducible_str(s: Set) -> str:
    """Create a string representation for a set deterministically."""
    return "{" + ", ".join(map(repr, sorted(s))) + "}"


def parse_validation_response(response: str) -> str:
    if 'answer: yes' in response.lower():
        return 'yes'
    elif 'answer: no' in response.lower():
        return 'no'
    else:
        return 'NA'


def create_problem_str(task: Task, version: str, abbrev_problem: bool, shuffle: bool = False) -> str:
    """

    :param task:
    :param version:
    :return:
    """
    objects = list(task.objects)
    init = list(task.init)
    goal = list(task.goal)

    if not shuffle:
        objects.sort()
        init.sort()
        goal.sort()
    else:
        random.shuffle(objects)
        #random.shuffle(init)
        init.sort()
        random.shuffle(goal)

    objects_str = ', '.join([f"'{str(o)}'" for o in objects])
    init_str = ', '.join([str(i) for i in init])
    goal_str = ', '.join([str(g) for g in goal])
    if version == 'python':
        problem_str = f'objects = {{{objects_str}}}\n'
        problem_str += f'init = {{{init_str}}}\n'
        problem_str += f'goal = {{{goal_str}}}\n'
    elif abbrev_problem:
        problem_str = task.get_abbreviated_problem_str()
    else:
        problem_str = task.problem_str

    return problem_str


def convert_path2str(dict_value):
    """
    Recursively converts all Path objects in the dictionary and nested dicts and lists
     into strings
    :param dict_value:
    :return:
    """
    if isinstance(dict_value, Path):
        return str(dict_value)

    elif isinstance(dict_value, list):

        new_dict_value = [convert_path2str(v) for v in dict_value]
        return new_dict_value

    elif isinstance(dict_value, dict):
        new_dict_value = dict()
        for key, value in dict_value.items():
            new_value = convert_path2str(value)
            new_dict_value[key] = new_value

        return new_dict_value

    elif isinstance(dict_value, Namespace):
        return dict()

    else:
        return dict_value


def create_ordered_results_data(out_data: dict):

    ordered_data = OrderedDict()
    ordered_keys = [
        "config",
        "eval_tasks",
        "eval_metrics",
        "validation_tasks",
        "validation_metrics",
        "validation_results",
        "debug_metrics",
        "acc_last_qual_check",
        "all_results_code"
    ]
    orig_keys = list(out_data.keys())
    for key in ordered_keys:
        try:
            value = out_data[key]
            ordered_data[key] = value
            orig_keys.remove(key)
        except KeyError:
            continue

    assert len(orig_keys) == 0, f'{orig_keys}'

    return ordered_data


def run_symbolic_planner(
        task: Task,
        plan_file: Union[str, None] = None,
        timeout: int = 30,
        error_if_not_plan: bool = True,
        optimal: bool = True) -> List[str]:

    domain_file = task.domain_file_path
    instance_file = task.problem_file_path

    failed, failure_issue = run_symbolic_planner_on_file(
        domain_file=domain_file,
        instance_file=instance_file,
        plan_file=plan_file,
        timeout=timeout,
        optimal=optimal
    )

    if os.path.exists(plan_file):

        plan = []
        with open(plan_file, 'r') as f:
            for line in f.readlines():
                l = line.strip()
                if not l.startswith(';'):
                    plan.append(l)
        if 'tmp_plan_file' in str(plan_file):
            os.remove(plan_file)

        # Run val on the generated plan because fastdownward ignores wrong typing!!
        plan_str = '\n'.join(plan)
        valid_plan = run_validator(task=task, plan=plan_str)
        if not valid_plan:
            return ['Not known']

        return plan

    else:
        assert failed
        assert failure_issue is not None
        if error_if_not_plan:
            raise RuntimeError(f'Could not generate the plan for {instance_file} using A* and fastdownward. Type of error {failure_issue}')
        else:
            return [failure_issue]


def run_validator(task: Task, plan: str) -> bool:
    # run VAL
    # return if the plan is valid

    val = os.environ.get('VAL')
    assert os.path.exists(val)
    # split plan into list of actions
    actions = plan.split('\n')

    # save them in a temporary file
    os.makedirs(TEMP_DIR, exist_ok=True)

    plan_file = TEMP_DIR / f'./tmp_plan_file_{get_timestamp_for_log()}'
    with open(plan_file, 'w') as pf:
        for ac in actions:
            pf.write(f'{ac}\n')

    cmd = f'{val}/validate -v {task.domain_file_path} {task.problem_file_path} {plan_file}'
    val_response = os.popen(cmd).read()

    if os.path.exists(plan_file):
        os.remove(plan_file)

    if 'Plan valid' in val_response:
        return True
    else:
        return False


def run_symbolic_planner_on_file(
        domain_file,
        instance_file,
        plan_file: Union[str, None] = None,
        timeout: int = 30,
        optimal: bool = True) -> Tuple[bool, Union[str, None]]:

    TEMP_DIR.mkdir(exist_ok=True, parents=True)

    fast_downward_path = os.getenv("FAST_DOWNWARD")
    assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
    fast_downward_executable = f'{fast_downward_path}/fast-downward.py'

    if plan_file is None:
        plan_file = TEMP_DIR / f'tmp_plan_file_{get_timestamp_for_log()}'

    if optimal:
        cmd_tokens = [fast_downward_executable, '--plan-file', plan_file, domain_file, instance_file, '--search',
                      'astar(lmcut())']

    else:
        cmd_tokens = [fast_downward_executable, '--plan-file', plan_file, domain_file, instance_file, "--evaluator", "h=ff()", "--search", "lazy_greedy([h], preferred=[h])"]

    console_output = None

    failed = False
    failure_issue = None
    try:
        run_result = subprocess.run(args=cmd_tokens, timeout=timeout, check=True, capture_output=True)
        console_output = str(run_result.stdout)
    except subprocess.TimeoutExpired as e:
        failure_issue = 'timed_out'
        failed = True
    except subprocess.CalledProcessError as e:
        failed = True
        console_output = e.output
        if 'exit status 12' in str(e) or 'stopped without finding a solution':  # error code for no solution found
            failure_issue = 'unsolvable'
        elif 'exit status 30' in str(e):    # error code for syntax / parsing
            failure_issue = f'invalid pddl'
            error_message = re.findall(r'b"[\S\n ]+"', console_output)
            failure_issue += f': {error_message}'
        else:
            failure_issue = 'Not known'

    return failed, failure_issue



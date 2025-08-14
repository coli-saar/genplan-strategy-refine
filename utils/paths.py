import os
from pathlib import Path
from datetime import datetime
from argparse import Namespace


_DIR = Path(__file__).parent.parent
PDDL_DIR = _DIR / "data"
CACHE_DIR = _DIR / "llm_cache"
TEMP_DIR = _DIR / Path('temp_files2')
PROMPT_DIR = _DIR / "prompts"
FASTDOWNWARD = '/scratch/kstein/fast-downward-22.12'
VAL = '/scratch/kstein/planner_tools/VAL'
LOG_DIR = _DIR / 'logs_error'
LLM_GEN_DIR = _DIR / 'llm_generated'


def get_llm_gen_plan_dir(llm_gen_dir: Path,
                         symbolic: bool,
                         llm_gen_prob: bool) -> Path:
    if symbolic:
        plans_dir = llm_gen_dir / 'plans_pddl'
    elif not llm_gen_prob:
        plans_dir = llm_gen_dir / 'plans_orig_prob_llm'
    else:
        plans_dir = llm_gen_dir / 'plans_new_prob_llm'

    plans_dir.mkdir(exist_ok=True, parents=True)

    return plans_dir


def get_llm_gen_problem_dir(llm_gen_dir: Path, valid: bool) -> Path:

    if valid:
        problems_dir = llm_gen_dir / 'problems_pddl'
    else:
        problems_dir = llm_gen_dir / 'invalid_problems_pddl'

    problems_dir.mkdir(exist_ok=True, parents=True)

    return problems_dir


def get_llm_gen_dir(output_dir, benchmark_name: str, domain_name: str) -> Path:

    directory = LLM_GEN_DIR / output_dir / benchmark_name / domain_name
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)

    get_llm_gen_problem_dir(llm_gen_dir=directory, valid=True)
    get_llm_gen_problem_dir(llm_gen_dir=directory, valid=False)
    get_llm_gen_plan_dir(llm_gen_dir=directory, symbolic=True, llm_gen_prob=False)
    get_llm_gen_plan_dir(llm_gen_dir=directory, symbolic=False, llm_gen_prob=True)
    get_llm_gen_plan_dir(llm_gen_dir=directory, symbolic=False, llm_gen_prob=False)

    return directory


def get_benchmark_dir(benchmark_name, domain_name):
    """
    Returns the path to the dataset directory
    :param benchmark_name:
    :param domain_name:
    :return:
    """
    benchmark_dir = PDDL_DIR / benchmark_name / domain_name
    return benchmark_dir


def get_results_file_path(domain_out_dir: str, flags: Namespace) -> Path:
    time_stamp_folder = create_log_dirname(flags=flags)
    out_dir = Path(flags.results_dir) / domain_out_dir / time_stamp_folder
    results_file_name = f"results_{flags.env}_{flags.seed}.json"
    results_file_path = out_dir / results_file_name

    return results_file_path


def get_timestamp_for_log() -> str:
    now = str(datetime.now())
    day_now = now.split(' ')[0]
    time_now = now.split(' ')[1]
    time_now = time_now.split('.')[0]  # remove milli sec

    unique_timestamp = day_now + '-' + time_now.replace(':', '-')

    return unique_timestamp


def create_log_dirname(flags: Namespace) -> str:

    model_name = flags.llm_configs['default']['llm_parameter']['model_type']
    time_stamp = flags.time_stamp
    dir_name = f'{model_name}_{time_stamp}'

    return dir_name


def get_res_file_name(flags: Namespace,
                      version: str) -> str:
    """
    Get the name for the results files
    version: "best" or "final"
    """
    return f"results_{version}_{flags.env}_{flags.seed}"




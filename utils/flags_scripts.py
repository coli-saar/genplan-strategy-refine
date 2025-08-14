"""Command line flags."""

import argparse
import json


FLAGS = argparse.Namespace()  # set by parse_flags() below


def create_parser_main() -> argparse.ArgumentParser:
    """Create the argument parser.

    Separated out for testing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--p-version", required=False, type=str, default='')
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--exp", required=False, type=str, default='')

    return parser


def parse_flags_main_pipeline() -> None:
    """Parse the command line flags for the main pipeline and update global FLAGS."""
    parser = create_parser_main()
    args = parser.parse_args()
    param_dict = args.__dict__
    config_file = param_dict.pop('config')

    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    FLAGS.__dict__.update(param_dict)
    FLAGS.__dict__.update(config_dict)
    FLAGS.__dict__['llm_counter'] = 1

    # experiment id (i.e. experiment name) in cmd arguments overwrites the 'experiment_id' value from the config
    if args.exp != '':
        FLAGS.experiment_id = args.exp

    # pipeline version in cmd arguments overwrites the one from the config
    if args.p_version != '':
        FLAGS.p_version = args.p_version

    set_defaults_main()


def create_parser_eval() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--eval_env', required=False, default=None)
    parser.add_argument('--version', required=False, default='json')
    parser.add_argument('--out', required=False, default='')
    parser.add_argument('--overwrite', required=False, type=bool, default=False, action=argparse.BooleanOptionalAction)

    return parser


def parse_flags_eval_pipeline() -> None:

    parser = create_parser_eval()
    args = parser.parse_args()
    param_dict = args.__dict__
    config_file = param_dict.pop('config')
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    FLAGS.__dict__.update(param_dict)
    FLAGS.__dict__.update(config_dict)
    FLAGS.__dict__['horizon'] = 1000000
    if 'overwrite' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['overwrite'] = False
    if 'include_llm_tasks' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['include_llm_tasks'] = False
    if 'randomize_input_order' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['randomize_input_order'] = True
    if 'n_randomizations' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['n_randomizations'] = 4
    if 'eval_env' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['eval_env'] = None
    if 'seeds' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['seeds'] = None


def create_parser_planbased_llm_eval() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--config', required=True)

    return parser


def parse_flags_planbased_llm_eval() -> None:

    parser = create_parser_planbased_llm_eval()
    args = parser.parse_args()
    param_dict = args.__dict__
    config_file = param_dict.pop('config')
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    FLAGS.__dict__.update(param_dict)
    FLAGS.__dict__.update(config_dict)
    FLAGS.__dict__['llm_counter'] = 1

    set_defaults_planbased_llm_eval()


def set_defaults_planbased_llm_eval():

    if 'abbreviate_problem_strs' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['abbreviate_problem_strs'] = False
        FLAGS.__dict__['abbrev_max_values'] = None

    if 'name_eval_split' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['name_eval_split'] = 'eval'

    if 'name_val_split' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['name_val_split'] = 'validate'


def set_defaults_main():

    if 'p_version' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['p_version'] = 'qual_check'

    if 'horizon' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['horizon'] = 1000000

    if 'timeout' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['timeout'] = 45

    if 'safety_checks' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['safety_checks'] = True

    if 'inputs_in_feedback' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['inputs_in_feedback'] = 'python'

    if 'prompt_problem_distribution' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['prompt_problem_distribution'] = 'none'

    if 'abbreviate_problem_strs' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['abbreviate_problem_strs'] = False
        FLAGS.__dict__['abbrev_max_values'] = None

    if 'save_path' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['save_path'] = 'outputs'

    if 'results_dir' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['results_dir'] = 'results'

    if 'name_eval_split' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['name_eval_split'] = 'eval'

    if 'name_val_split' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['name_val_split'] = 'validate'

    if 'code_hist' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['code_hist'] = 'full'

    if 'input_version' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['input_version'] = 'pddl'

    if 'eval_last_code' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['eval_last_code'] = True

    if 'eval_best_code' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['eval_best_code'] = True

    if 'eval_first_code' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['eval_first_code'] = True

    if 'stop_on_val' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['stop_on_val'] = False

    if 'prompt_dir' not in FLAGS.__dict__.keys():
        FLAGS.__dict__['prompt_dir'] = None

    flags_dict = FLAGS.__dict__
    if 'randomize_input_order' not in FLAGS.__dict__.keys():
        if 'randomize_input_order' in flags_dict['validator']['args'].keys():
            FLAGS.__dict__['randomize_input_order'] = flags_dict['validator']['args']['randomize_input_order']
        else:
            FLAGS.__dict__['randomize_input_order'] = True

    if 'n_randomizations' not in FLAGS.__dict__.keys():
        if 'n_randomizations' in flags_dict['validator']['args'].keys():
            FLAGS.__dict__['n_randomizations'] = flags_dict['validator']['args']['n_randomizations']
        else:
            FLAGS.__dict__['n_randomizations'] = 4

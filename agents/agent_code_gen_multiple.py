from typing import Dict, List, Union
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from llm_models.llm_base_class import LLMModel
from agents.agent_code_gen import AgentCodeGeneration
from agents.validator_feedback_gen import GenPlanValidator
from agents.agent_code_gen_basic import postprocess_response
from utils.tasks import TaskData


class AgentCodeGenerationMultiple(AgentCodeGeneration):

    def __init__(self,
                 llm_model: LLMModel,
                 validation_task_dict: Dict[str, TaskData],
                 debug_task_names: List[str],
                 code_gen_prompt_file: str,
                 reflection_prompt_file: str,
                 refine_code_prompt_file: str,
                 validator_param: dict,
                 number_parallel_codes: int,
                 log_dir: Path,
                 flags: dict,
                 plan_based_val_args: Union[dict, None] = None,
                 use_llm_gen_tasks: bool = True,
                 include_plans: bool = True,
                 include_nl: bool = False,
                 only_symb_plans: bool = False,
                 typed_hints: bool = False,
                 number_of_examples: int = 1,
                 force_examples: bool = False,
                 max_debug_steps: int = 10,
                 adaptive: bool = False,
                 log_code_exec: bool = False,
                 debug_hist_len: Union[int, None] = None):
        """

        :param llm_model:
        :param validation_task_dict:
        :param code_gen_prompt_file:
        :param reflection_prompt_file:
        :param refine_code_prompt_file:
        :param validator_param:
        :param log_dir:
        :param flags:
        :param plan_based_val_args:
        :param use_llm_gen_tasks: if True, then also self-generated tasks are used                      during debugging
        :param typed_hints:
        :param number_of_examples:
        :param force_examples: if False then examples will only be included if there exist plans that the LLM generated according to the strategy
                             else number_of_examples examples will be selected where LLM-generated ones are preferred but not only those are selected
        :param max_debug_steps:
        """

        super().__init__(llm_model=llm_model,
                         validation_task_dict=validation_task_dict,
                         debug_task_names=debug_task_names,
                         code_gen_prompt_file=code_gen_prompt_file,
                         reflection_prompt_file=reflection_prompt_file,
                         refine_code_prompt_file=refine_code_prompt_file,
                         validator_param=validator_param,
                         log_dir=log_dir,
                         flags=flags,
                         plan_based_val_args=plan_based_val_args,
                         use_llm_gen_tasks=use_llm_gen_tasks,
                         include_plans=include_plans,
                         include_nl=include_nl,
                         only_symb_plans=only_symb_plans,
                         typed_hints=typed_hints,
                         number_of_examples=number_of_examples,
                         force_examples=force_examples,
                         max_debug_steps=max_debug_steps,
                         log_code_exec=log_code_exec,
                         debug_hist_len=debug_hist_len)

        self.n_parallel = number_parallel_codes
        self.shuffle_problem_str = True

        self.next_prompts = []

        self.data_all_runs = dict()
        self.best_run = None
        self.best_coverage = None

        self.adaptive = adaptive
        self.max_total = (max_debug_steps + 1) * number_parallel_codes

        self.prev_code = None

    def run_complete_generation_loop(self):

        self.create_all_initial_prompts()
        counter = 0
        while len(self.next_prompts) > 0:

            self.debug_attempt = 0
            self._generate_first_code()

            if not self.last_validator_debug.plan_succeeded_all:
                self._run_debugging_loop()

            current_best_coverage = len(list(self.best_validator_validate.successful_tasks.keys()))
            if self.best_run is None:
                self.best_run = counter
                self.best_coverage = current_best_coverage
            elif current_best_coverage >= self.best_coverage:
                self.best_run = counter
                self.best_coverage = current_best_coverage

            data_current_run = {
                'coverage_best_code': current_best_coverage,
                'first_validator_validate': deepcopy(self.first_validator_validate),
                'last_validator_validate': deepcopy(self.last_validator_validate),
                'best_validator_validate': deepcopy(self.best_validator_validate),
                'last_validator_debug': deepcopy(self.last_validator_debug),
                'required_debug_steps': self.required_debug_steps,
                'step_best_validator': self.step_best_validator,
                'step_last_validator': self.step_last_validator,
                'debugs_code': self.debugs_code,
                'debugs_pseudocode': self.debugs_pseudocode,
                'tasks_used_for_debugging': deepcopy(self.tasks_used_for_debugging),
                'debug_attempt': self.debug_attempt,
                'validation_results': deepcopy(self.validation_results),
                'last_code': deepcopy(self.last_code),
                'last_pseudocode': deepcopy(self.last_pseudocode)
            }
            self.data_all_runs[counter] = data_current_run

            if self.last_validator_debug.plan_succeeded_all:
                break

            counter += 1

            # reset all variables for the next round
            self.reset_for_next_run()

        self.finalize_with_best_run()

        self.log_final_codes()

    def finalize_with_best_run(self):

        best_run_data = self.data_all_runs[self.best_run]
        self.first_validator_validate = best_run_data['first_validator_validate']
        self.last_validator_validate = best_run_data['last_validator_validate']
        self.best_validator_validate = best_run_data['best_validator_validate']
        self.last_validator_debug = best_run_data['last_validator_debug']

        self.required_debug_steps = best_run_data['required_debug_steps']
        self.step_best_validator = best_run_data['step_best_validator']
        self.step_last_validator = best_run_data['step_last_validator']
        self.debugs_code = best_run_data['debugs_code']
        self.debugs_pseudocode = best_run_data['debugs_pseudocode']
        self.tasks_used_for_debugging = best_run_data['tasks_used_for_debugging']

        self.debug_attempt = best_run_data['debug_attempt']
        self.validation_results = best_run_data['validation_results']

        self.last_code = best_run_data['last_code']
        self.last_pseudocode = best_run_data['last_pseudocode']

    def reset_for_next_run(self):

        self.first_validator_validate: Union[GenPlanValidator, None] = None
        self.last_validator_validate: Union[GenPlanValidator, None] = None
        self.best_validator_validate: Union[GenPlanValidator, None] = None
        self.last_validator_debug: Union[GenPlanValidator, None] = None

        self.required_debug_steps = 0
        self.step_best_validator = 0
        self.step_last_validator = 0
        self.debugs_code = 0
        self.debugs_pseudocode = None
        self.tasks_used_for_debugging = []

        self.debug_attempt = 0
        self.validation_results = OrderedDict()

        self.last_failed_task_name = ''
        self.last_code = None
        self.last_pseudocode = None

        self.llm_model.reset_history()

    def generate_code(self) -> str:

        prompt = self.next_prompts.pop(0)

        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output(code=True)
        response = postprocess_response(model_response=response)

        self.last_code = response

        return response

    def create_all_initial_prompts(self):

        all_prompt_params = []
        problem_strs = []

        no_change = 0
        n_potential_init_codes = self.n_parallel if not self.adaptive else self.max_total
        while len(all_prompt_params) < n_potential_init_codes:

            prompt_params = self.get_param_main_prompt()
            examples = prompt_params['examples']
            if examples in problem_strs:
                no_change += 1
                continue
            else:
                all_prompt_params.append(prompt_params)
                problem_strs.append(examples)
                no_change = 0

            if no_change == 30:
                break

        for prompt_param in all_prompt_params:
            prompt = self.code_gen_prompt_template.render(**prompt_param)
            self.next_prompts.append(prompt)


def compare_code_responses(code1, code2):

    cleaned_lines_code1 = standardize_code(code1)
    cleaned_lines_code2 = standardize_code(code2)

    return cleaned_lines_code1 == cleaned_lines_code2


def standardize_code(code: str):
    cleaned_lines = []

    lines = code.strip().split('\n')
    for line in lines:
        if not line:
            continue

        if '#' in line:
            begin_comment = line.index('#')
            line = line[:begin_comment]

        line = line.rstrip()
        cleaned_lines.append(line)

    return cleaned_lines


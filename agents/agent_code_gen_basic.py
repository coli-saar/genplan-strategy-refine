import os
import sys
import random
from copy import copy
from typing import List, Dict, Tuple, Union
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from llm_models.llm_base_class import LLMModel
from agents.validator_feedback_gen import GenPlanValidator
from utils.tasks import TaskData, Task
from utils.utils import create_prompt_template, create_problem_str, log_agent_system_prompt, log_llm_interaction


def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


"""
Agent for generating code without the pseudo code step and without self-reflection
"""


class AgentCodeGenerationBasic:

    def __init__(self,
                 llm_model: LLMModel,
                 validation_task_dict: Dict[str, TaskData],
                 debug_task_names: List[str],
                 code_gen_prompt_file: str,
                 refine_code_prompt_file: str,
                 validator_param: dict,
                 log_dir: Path,
                 flags: dict,
                 use_llm_gen_tasks: bool = True,
                 include_plans: bool = True,
                 include_nl: bool = False,
                 only_symb_plans: bool = False,
                 typed_hints: bool = False,
                 number_of_examples: int = 1,
                 force_examples: bool = False,
                 max_debug_steps: int = 10,
                 log_code_exec: bool = False,
                 debug_hist_len: Union[int, None] = None):
        """

        :param llm_model:
        :param validation_task_dict:
        :param debug_task_names:
        :param code_gen_prompt_file:
        :param refine_code_prompt_file:
        :param validator_param:
        :param log_dir:
        :param flags:
        :param use_llm_gen_tasks: if True, then also self-generated tasks are used                      during debugging
        :param typed_hints:
        :param number_of_examples:
        :param force_examples: if False then examples will only be included if there exist plans that the LLM generated according to the strategy
                             else number_of_examples examples will be selected where LLM-generated ones are preferred but not only those are selected
        :param max_debug_steps:
        """
        self.agent_name = 'CodeGen'
        self.flags = flags

        self.llm_model = llm_model
        self.validation_task_dict = validation_task_dict
        self.debugging_task_dict = self.get_debug_task_dict(debug_task_names=debug_task_names)

        self.validator_param = validator_param

        self.code_gen_prompt_template = create_prompt_template(code_gen_prompt_file, self.flags)

        self.refine_code_prompt_template = create_prompt_template(refine_code_prompt_file, self.flags)

        self.max_debug_steps = max_debug_steps
        self.num_examples = min(number_of_examples, len(self.validation_task_dict.keys()))
        self.force_examples = force_examples
        self.typed_hints = typed_hints
        self.include_plans = include_plans
        self.include_nl = include_nl
        self.only_symb_plans = only_symb_plans

        self.use_llm_gen_tasks = use_llm_gen_tasks

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

        if debug_hist_len not in [None, 0]:
            raise NotImplementedError
        self.debug_hist_len = debug_hist_len

        self.debug_attempt = 0
        self.validation_results = OrderedDict()

        self.last_failed_task_name = ''
        self.last_code = None
        self.last_pseudocode = None

        self.shuffle_problem_str = False
        self.best_run = 0

        # Logging set-up
        self.log_dir = log_dir
        log_agent_system_prompt(log_dir=self.log_dir,
                                llm_model=self.llm_model,
                                agent_name=self.agent_name)

        if log_code_exec:
            self.code_exec_log_dir = self.set_up_log_code_exec()
        else:
            self.code_exec_log_dir = ''

    def log_llm_input_output(self, code: bool = False):
        log_llm_interaction(log_dir=self.log_dir,
                            llm_model=self.llm_model,
                            agent_name=self.agent_name,
                            flags=self.flags,
                            python_code=code)

    def set_up_log_code_exec(self):
        code_exec_log_dir = os.path.join(self.log_dir, 'code_execution_logs')
        Path(code_exec_log_dir).mkdir(exist_ok=True, parents=True)
        return code_exec_log_dir

    def get_code_exec_log_path(self):
        if self.code_exec_log_dir == '':
            return ''

        log_file = f'code_execution_debug_step_{self.debug_attempt}.txt'
        log_path = os.path.join(self.code_exec_log_dir, log_file)

        return log_path

    def log_final_codes(self):
        if self.code_exec_log_dir == '':
            code_log_dir = self.set_up_log_code_exec()
        else:
            code_log_dir = self.code_exec_log_dir

        log_file_best = os.path.join(
            code_log_dir,
            f'best_generated_code_{self.step_best_validator}.txt'
        )
        log_file_last = os.path.join(
            code_log_dir,
            f'last_generated_code_{self.debug_attempt}.txt'
        )

        with open(log_file_best, 'w') as f:
            f.write(self.best_validator_validate.generalized_plan.code_str)

        with open(log_file_last, 'w') as f:
            f.write(self.last_validator_validate.generalized_plan.code_str)

        last_code_str_debug = self.last_validator_debug.generalized_plan.code_str
        last_code_str_validate = self.last_validator_validate.generalized_plan.code_str
        assert last_code_str_debug == last_code_str_validate

    def get_debug_task_dict(self, debug_task_names: list):
        debug_task_dict = dict()
        for task_name, task_data in self.validation_task_dict.items():
            if task_name in debug_task_names:
                debug_task_dict[task_name] = task_data

        return debug_task_dict

    def run_complete_generation_loop(self):
        self.debug_attempt = 0
        self._generate_first_code()

        if not self.last_validator_debug.plan_succeeded_all:
            self._run_debugging_loop()

        self.log_final_codes()

    def _generate_first_code(self):

        # Generate code
        code_response = self.generate_code()

        # Pass the code to the validator and
        # validate on all debugging and validation tasks
        self.run_and_update_validators(code_to_validate=code_response)

    def _run_debugging_loop(self):

        # Run debugging loop
        while self.debug_attempt <= self.max_debug_steps:
            self.debug_attempt += 1
            self.required_debug_steps += 1
            self.step_last_validator += 1

            # Take feedback and information about solved and failed problem and refine code
            refined_code = self.refine_code()

            # Pass the code to the validator and
            # validate on all debugging and validation tasks
            self.run_and_update_validators(code_to_validate=refined_code)

            if self.last_validator_debug.plan_succeeded_all:
                return

    def run_and_update_validators(self, code_to_validate):
        # Pass the code to the validator and validate on the debugging tasks
        self.last_validator_debug = self.run_validation(
            code_response=code_to_validate,
            task_dict=self.debugging_task_dict
        )

        # Run validation on all validation tasks
        # if FLAGS.stop_on_val is True, then debugging_task_dict == validation_task_dict
        # -> no need to rerun
        if self.flags['stop_on_val']:
            validator_validate = deepcopy(self.last_validator_debug)
        else:
            validator_validate = self.run_validation(
                code_response=code_to_validate,
                task_dict=self.validation_task_dict
            )

        self.last_validator_validate = validator_validate

        if self.debug_attempt == 0:
            self.best_validator_validate = validator_validate
            self.first_validator_validate = validator_validate
        else:
            self.update_best_validator()

        return validator_validate

    def update_best_validator(self):

        n_last_solved = len(list(self.last_validator_validate.successful_tasks.keys()))
        n_best_solved = len(list(self.best_validator_validate.successful_tasks.keys()))

        if n_last_solved >= n_best_solved:
            self.best_validator_validate = self.last_validator_validate
            self.step_best_validator = copy(self.debug_attempt)

    def select_initial_prompt_examples(self) -> List[TaskData]:
        """
        Select as many tasks as examples as specified by self.num_examples
        Select primarily tasks for which there exists a valid plan generated by
           the LLM based on the strategy
        Fill-up with other tasks if not enough
        :return:
        """
        selected = []
        names_only_optimal_plan = []
        for task_name, task_data in self.debugging_task_dict.items():

            if (not self.use_llm_gen_tasks) and task_data.llm_generated_task:
                continue

            if task_data.generated_plan is not None:
                selected.append(task_data)
            elif task_data.get_plan() is not None:  # look only at solvable instances
                names_only_optimal_plan.append(task_name)

        if len(selected) < self.num_examples and self.force_examples:
            while len(selected) < self.num_examples:
                t_name = names_only_optimal_plan.pop(0)
                selected.append(self.debugging_task_dict[t_name])
        elif len(selected) < self.num_examples:
            selected = selected
        else:
            selected = selected[:self.num_examples]

        return selected

    def get_param_main_prompt(self) -> dict:
        """
        Generates the dictionary with the parameters for the main prompt, i.e.
        for the first code generation or for the first pseudocode generation if
        there is pseudocode generation
        :return:
        """
        random_taskdata = list(self.debugging_task_dict.values())[0]
        domain_description = random_taskdata.get_domain_nl()
        domain_strategy = random_taskdata.get_strategy_domain()
        if self.typed_hints:
            action_hint = random_taskdata.task.actions_hint_typed
            predicate_hint = random_taskdata.task.predicates_hint_typed
        else:
            action_hint = random_taskdata.task.actions_hint
            predicate_hint = random_taskdata.task.predicates_hint

        if random_taskdata.task.typed:
            objects_input = '`objects` is a set of the name-type pairs of all objects'
            object_type_str = 'Set[Tuple[str, str]]'
        else:
            objects_input = '`objects` is the set of the names of all objects'
            object_type_str = 'Set[str]'

        example_tasks = self.select_initial_prompt_examples()
        examples = []
        for ex_enum, ex_task in enumerate(example_tasks):
            ex_num = ex_enum + 1 if len(example_tasks) != 1 else ''
            problem_str = create_problem_str(task=ex_task.task,
                                             version='python',
                                             abbrev_problem=self.flags['abbreviate_problem_strs'],
                                             shuffle=self.shuffle_problem_str)

            problem_str = problem_str.replace('objects =', f'objects{ex_num} =')
            problem_str = problem_str.replace('init =', f'init{ex_num} =')
            problem_str = problem_str.replace('goal =', f'goal{ex_num} =')
            example_str = f'{problem_str}'

            if self.include_nl:
                problem_nl = ex_task.get_problem_nl()
                prefix = f'Description of the example task{ex_num}:\n{problem_nl}'

                example_str = f'{prefix}\n\nPython formalization of the task{ex_num}:\n{example_str}'

            if self.include_plans:
                if self.only_symb_plans:
                    plan = ex_task.task.get_optimal_plan()
                else:
                    plan = ex_task.get_plan()
                example_str = f'{example_str}\nValid solution for the task{ex_num}:\n{plan}'
            examples.append(example_str)

        single_example = True if len(example_tasks) == 1 else False

        prompt_param = {'domain_descript': domain_description,
                        'strategy': domain_strategy,
                        'object_input': objects_input,
                        'action_hint': action_hint,
                        'predicate_hint': predicate_hint,
                        'object_type_str': object_type_str,
                        'examples': examples,
                        'single': single_example}

        return prompt_param

    def generate_code(self) -> str:

        prompt_param = self.get_param_main_prompt()

        prompt = self.code_gen_prompt_template.render(**prompt_param)
        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output(code=True)

        response = postprocess_response(model_response=response)
        self.last_code = response

        return response

    def run_validation(self,
                       code_response: str,
                       task_dict: Dict[str, TaskData]) -> GenPlanValidator:

        code_validator_param = deepcopy(self.validator_param)
        code_validator_param['genplan_code'] = code_response
        code_validator_param['validation_tasks'] = []

        for task_data in task_dict.values():
            if (not self.use_llm_gen_tasks) and task_data.llm_generated_task:
                continue
            code_validator_param['validation_tasks'].append(task_data.task)

        validator_name = code_validator_param.pop('name')
        validator: GenPlanValidator = str_to_class(validator_name)(**code_validator_param)

        if self.code_exec_log_dir != '':
            code_log_file_path = self.get_code_exec_log_path()
            validator.set_log_file(log_file=code_log_file_path)

        validator.run_validation_all_tasks()

        task_metrics = validator.metrics['tasks_metrics']
        res_to_save = {
            'solved_tasks': [],
            'n_correct': 0,
            'all_order_same_eval': 0
        }
        n_total = 0
        for instance, results in task_metrics.items():
            n_total += 1
            if results['success']:
                res_to_save['solved_tasks'].append(instance)
                res_to_save['n_correct'] += 1
            if not 'all_same_eval' in results:
                print(results)
                print(validator.randomize_input_order)
                print(validator.n_randomizations)
            if results['all_same_eval']:
                res_to_save['all_order_same_eval'] += 1
        res_to_save['accuracy'] = res_to_save['n_correct'] / n_total
        self.validation_results[self.debug_attempt] = res_to_save

        return validator

    def get_param_feedback_prompt(self):
        """
        Generate prompt parameter for the feedback prompt, i.e.
        in the basic set-up this is the refine_code_prompt and in the
        set-up with self reflection it is the reflection_prompt
        :return:
        """
        solved_tasks_names = list(self.last_validator_debug.successful_tasks.keys())

        python_error_tasks = self.last_validator_debug.python_error_tasks
        plan_error_tasks = self.last_validator_debug.plan_error_tasks
        if len(python_error_tasks.keys()) > 0:
            # If the same task provided in the prev. feedback is still not sovled use the same task again
            if self.last_failed_task_name != '' and self.last_failed_task_name in python_error_tasks.keys():
                failed_task_name = self.last_failed_task_name
            else:
                failed_task_name = random.choice(list(python_error_tasks.keys()))
            failed_task = python_error_tasks[failed_task_name]['task']
            failed_task_feedback = python_error_tasks[failed_task_name]['feedback']
        else:
            if self.last_failed_task_name != '' and self.last_failed_task_name in plan_error_tasks.keys():
                failed_task_name = self.last_failed_task_name
            else:
                failed_task_name = random.choice(list(plan_error_tasks.keys()))
            failed_task = plan_error_tasks[failed_task_name]['task']
            failed_task_feedback = plan_error_tasks[failed_task_name]['feedback']

        self.tasks_used_for_debugging.append(failed_task_name)

        failed_problem = create_problem_str(
            task=failed_task,
            version='python',
            abbrev_problem=self.flags['abbreviate_problem_strs'])

        solved_problems = []
        if not self.last_validator_debug.include_pos_results:
            prompt_param = {'solved_problems': solved_problems,
                            'incl_positive': False,
                            'failed_problem': failed_problem,
                            'feedback': failed_task_feedback}
            return prompt_param

        for t_name in solved_tasks_names:
            task = self.last_validator_debug.successful_tasks[t_name]['task']
            problem_str = create_problem_str(task=task, version='python',
                                             abbrev_problem=self.flags['abbreviate_problem_strs'])
            feedback_correct_plan = self.last_validator_debug.successful_tasks[t_name]['feedback']
            solved_problems.append(f'{problem_str}\n{feedback_correct_plan}')

        prompt_param = {'solved_problems': solved_problems,
                        'incl_positive': True,
                        'failed_problem': failed_problem,
                        'feedback': failed_task_feedback}

        return prompt_param

    def refine_code(self) -> str:

        self.debugs_code += 1

        prompt_param = self.get_param_feedback_prompt()

        prompt = self.refine_code_prompt_template.render(**prompt_param)
        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output(code=True)

        response = postprocess_response(model_response=response)

        self.last_code = response

        if self.debug_hist_len == 0 and self.debugs_code > 0:
            self.shorten_debug_hist()

        return response

    def shorten_debug_hist(self):

        current_history = self.llm_model.get_history()
        len_initial_hist = len(self.llm_model.get_initial_history())
        assert len(current_history) == 5
        new_history = current_history[:len_initial_hist + 1]

        last_code = self.last_code
        last_code_formatted = f'```python\n{last_code}\n```'

        # TODO: decide
        new_history.append({'role': 'assistant', 'content': last_code_formatted})
        #new_history.append(current_history[-1])

        self.llm_model.update_history(new_history=new_history)


def postprocess_response(model_response):
    python_code = _parse_python_code_from_response(model_response)
    cleaned_code, mistake = remove_example_calls(python_code)
    # TODO: mistake currently not used but need to give feedback because otherwise this will not work

    return cleaned_code


def _parse_python_code_from_response(response: str) -> str:
    """
    Parse out python code if python code is indicated by ```python ... ```
    Otherwise treat the complete response as python code
    :param response:
    :return:
    """
    python_code_prefix = "```python"
    if python_code_prefix in response:
        python_start = response.index(python_code_prefix)
        python_remainder = response[python_start + len(python_code_prefix):]
        if "```" in python_remainder:
            python_end = python_remainder.index("```")
        else:
            python_end = len(python_remainder)
        python_response = python_remainder[:python_end]
        return python_response
    return response


def remove_example_calls(generated_code: str) -> Tuple[str, str]:
    """
    Remove any call to the get_plan function and everything following a potential call of the function
    :param generated_code:
    :return:
    """
    code_lines = generated_code.split('\n')
    found_function = False

    cleaned_code_lines = []

    for line in code_lines:
        if "def generate_solution" in line:
            found_function = True
        if line.startswith('def generate_solution(') and line.strip().endswith(')'):
            break

        cleaned_code_lines.append(line)

    cleaned_code = '\n'.join(cleaned_code_lines)
    if not found_function:
        return cleaned_code, 'function-not-generated'
    else:
        return cleaned_code, ''


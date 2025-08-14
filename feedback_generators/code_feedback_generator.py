import sys
import time
from argparse import Namespace
from typing import Union, Dict, List
from utils.tasks import Task
from feedback_generators.simulator_pddl import RawPDDLEnvironment


# Needs to be defined the same script where the classes are available through import
def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


class CodeFeedbackGenerator:
    """
    Feedback types:
    - infinite loops
    - python exceptions
    - plan did not reach the goal -> which parts are satisfied, which parts not
    - plan is not executable
       - unsatisfied preconditions
       - not a valid action
       - wrong PDDL syntax
       - wrong number of args
       - wrong type of args
    """

    def __init__(self,
                 generalized_plan,
                 pddl_env_name: str,
                 pddl_env_args: dict,
                 task: Union[Task, None],
                 timeout: int,
                 flags: Namespace,
                 format_plan: bool = False,
                 enum_plan: bool = False,
                 inference: bool = False):
        self.flags = flags
        self.inference = inference  # whether only for evaluation

        self.timeout = timeout
        self.generalized_plan = generalized_plan
        self.pddl_env_name = pddl_env_name
        self.pddl_env_args = pddl_env_args
        self.pddl_env: Union[None, RawPDDLEnvironment] = None
        self.task = task
        self.current_pddl_env_args = dict()

        self.format_plan = format_plan if not enum_plan else True
        self.enum_plan = enum_plan

        self.result_dict: Dict = {'success': False,
                                  'feedback': '',
                                  'error-type': [],
                                  'plan-length': None}

    def get_success(self):
        return self.result_dict['success']

    def get_feedback(self):
        return self.result_dict['feedback']

    def get_error_type(self):
        return self.result_dict['error-type']

    def reset_result_dict(self):
        result_dict: Dict = {'success': False,
                             'feedback': '',
                             'error-type': [],
                             'plan-length': None,
                             'error_occurred': None}
        self.result_dict = result_dict

    def update_task_and_env(self,
                            task: Task,
                            domain_file_path,
                            instance_file_path):
        self.task = task
        self.reset_result_dict()
        self.update_pddl_env(domain_file_path=domain_file_path,
                             instance_file_path=instance_file_path)

    def update_pddl_env(self, domain_file_path, instance_file_path):

        args = {'instance_file': instance_file_path,
                'domain_file': domain_file_path}
        self.current_pddl_env_args = {
            'instance_file': instance_file_path,
            'domain_file': domain_file_path
        }

        if self.pddl_env is not None:
            self.pddl_env.remove_temp_files()

        # add additional args, e.g. the NL description file for the NL environment
        args.update(self.pddl_env_args)
        self.pddl_env = str_to_class(self.pddl_env_name)(**args)

    def reset_pddl_env(self):
        args = self.current_pddl_env_args

        if self.pddl_env is not None:
            self.pddl_env.remove_temp_files()

        # add additional args, e.g. the NL description file for the NL environment
        args.update(self.pddl_env_args)
        self.pddl_env = str_to_class(self.pddl_env_name)(**args)


    def validate_and_generate_feedback(self, version: str, args: dict):

        if version == 'python-exception' or version == 'timeout':
            if 'plan-length' in self.result_dict.keys():
                plan_len = self.result_dict.get('plan-length', None)
                if plan_len is not None:
                    print(args)
                assert plan_len is None

        if version == 'python-exception':
            self._get_feedback_python_exception(**args)
            self.result_dict['error_occurred'] = 'python-exception'
        elif version == 'timeout':
            self._get_feedback_infinite_loop(**args)
            self.result_dict['error_occurred'] = 'timeout'
        elif version == 'plan':
            self._get_feedback_generated_plan(**args)
        else:
            raise NotImplementedError

    def _get_feedback_infinite_loop(self, info: str):

        # For some reasons the result_dict is sometimes empty when the code
        # gets interrupted due to time out
        if len(list(self.result_dict.keys())) == 0:
            self.reset_result_dict()
            self.result_dict['success'] = False

        feedback = f"\nThe code was interrupted because it did not terminate within the time" \
                   f" limit ({self.timeout} seconds). " \
                   f"This is likely caused by an infinite loop. Please check the loops again. "

        self.result_dict['feedback'] = info + feedback
        self.result_dict['error-type'].append('timeout')

    def _get_feedback_python_exception(self, traceback: List[str]):
        tb_lines = [l.replace(str(self.generalized_plan.filepath), "<file-name-omitted>")
                    for l in traceback if "agentic-generalized-planning" not in l]
        tb_str = "".join(tb_lines)
        message = f'The code raised the following exception:\n{tb_str}'
        self.result_dict["feedback"] = message
        self.result_dict["error-type"].append("python-exception")

    def _get_feedback_generated_plan(self, plan):

        if self.format_plan:
            formatted_plan = self.get_enum_formatted_plan(plan=plan)
            plan_in_feedback = formatted_plan
        else:
            plan_in_feedback = plan
        enum_info = ' (comments added for readability)' if self.enum_plan else ''

        # Make sure a simulator has been instantiated and is still in the initial state
        assert self.pddl_env is not None
        assert self.pddl_env.current_step_id == 0

        if not isinstance(plan, list):
            feedback = f"The code returned {plan}, which is not a list of actions. " \
                       f"Please make sure that your code returns a list of actions, i.e. of type List[str]"
            self.result_dict["feedback"] = feedback
            self.result_dict["error-type"].append("output-not-plan")
            return

        self.result_dict['plan-length'] = len(plan)

        # Deal with the case that an empty plan is returned
        if len(plan) == 0:
            completely_executable = True
            feedback, executable, reached_goal_at_end, error_types = self.pddl_env.check_completion()

        else:
            plan_str = '\n'.join(plan)
            executable, reached_goal_at_end = self.pddl_env.validate_complete_plan(plan_str=plan_str)

            completely_executable = executable

            # TODO:
            if not executable:
                feedback = f'The code returned the following output:\n{plan_in_feedback}\nbut not all actions are executable.'
                self.result_dict["feedback"] = feedback
                self.result_dict["error-type"].extend(['unsat-preconditions'])
                self.result_dict['error_occurred'] = 'unsat-preconditions'

        if not completely_executable and not self.inference:

            completely_executable = True
            self.reset_pddl_env()
            for action_id, action_str in enumerate(plan):
                simulator_feedback, executable, reached_goal, error_types = self.pddl_env.step(action_instr=action_str)
                completely_executable = False if not executable else completely_executable
                if not executable:
                    if self.enum_plan:
                        feedback = f'The code returned the following output{enum_info}:\n{plan_in_feedback}\n'
                        feedback += f'The action {action_str} at step {action_id} is not executable because {simulator_feedback}'
                    else:
                        feedback = f'The code returned the following output:\n{plan_in_feedback}\n'
                        feedback += f'The action {action_str} at index {action_id} is not executable because {simulator_feedback}'

                    self.result_dict["feedback"] = feedback
                    self.result_dict["error-type"].extend(error_types)
                    self.result_dict['error_occurred'] = 'unsat-preconditions'
                    return

                if reached_goal:
                    reached_goal_at_end = True
                else:
                    reached_goal_at_end = False

        if completely_executable and not reached_goal_at_end:

            goal_feedback = self.pddl_env.goal_feedback
            feedback = f'The code returned the following output{enum_info}:\n{plan_in_feedback}\n{goal_feedback}'

            self.result_dict["feedback"] = feedback
            self.result_dict["error-type"].append('not-reached-goal')
            self.result_dict['error_occurred'] = 'goal-not-reached'

        elif reached_goal_at_end:
            feedback = f'The code returned the correct output:\n{plan}'
            self.result_dict['feedback'] = feedback
            self.result_dict['success'] = True

    def get_enum_formatted_plan(self, plan: list[str]) -> str:

        formatted_plan = '[\n'
        for action_num, action in enumerate(plan):
            if self.enum_plan:
                formatted_plan += f'\t{action},\t# step {action_num}\n'
            else:
                formatted_plan += f'\t{action},\n'

        formatted_plan += ']'

        return formatted_plan


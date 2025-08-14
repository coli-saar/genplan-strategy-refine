import sys
from argparse import Namespace
from typing import Union, Dict, List
from utils.tasks import Task
from feedback_generators.simulator_pddl import RawPDDLEnvironment


# Needs to be defined the same script where the classes are available through import
def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


class BasicFeedbackGenerator:
    """
    Feedback only based on VAL for a generated PDDL plan
    """
    def __init__(self,
                 pddl_env_name: str,
                 pddl_env_args: dict,
                 task: Union[Task, None],
                 flags: Namespace,
                 enum_plan: bool = False
                 ):

        self.flags = flags
        self.pddl_plan = None
        self.pddl_env_name = pddl_env_name
        self.pddl_env_args = pddl_env_args
        self.pddl_env: Union[None, RawPDDLEnvironment] = None
        self.task = task

        self.enum_plan = enum_plan

        self.result_dict: Dict = {'success': False,
                                  'feedback': '',
                                  'error-type': [],
                                  'plan-length': None}

    def update_task_and_env(self,
                            task: Task,
                            pddl_plan: str):
        domain_file_path = task.domain_file_path
        instance_file_path = task.problem_file_path
        self.pddl_plan = pddl_plan
        self.task = task
        self.reset_result_dict()
        self.update_pddl_env(domain_file_path=domain_file_path,
                             instance_file_path=instance_file_path)

    def reset_result_dict(self):
        result_dict: Dict = {'success': False,
                             'feedback': '',
                             'error-type': [],
                             'plan-length': None}
        self.result_dict = result_dict

    def update_pddl_env(self, domain_file_path, instance_file_path):

        args = {'instance_file': instance_file_path,
                'domain_file': domain_file_path}

        # add additional args, e.g. the NL description file for the NL environment
        args.update(self.pddl_env_args)
        self.pddl_env = str_to_class(self.pddl_env_name)(**args)

    def get_feedback_generated_plan(self):

        plan = self.pddl_plan.strip().split('\n')

        # Remove all empty lines / entries
        plan = [action for action in plan if action != '']

        enum_info = ' (step numbers added for readability)' if self.enum_plan else ''
        if self.enum_plan:
            plan_in_feedback = self.get_enum_plan(plan=plan)
        else:
            plan_in_feedback = self.pddl_plan.strip()

        # Make sure a simulator has been instantiated and is still in the initial state
        assert self.pddl_env is not None
        assert self.pddl_env.current_step_id == 0

        self.result_dict['plan-length'] = len(plan)

        reached_goal_at_end = False
        # Deal with the case that an empty plan is returned
        if len(plan) == 0:
            completely_executable = True
            feedback, executable, reached_goal_at_end, error_types = self.pddl_env.check_completion()

        else:
            completely_executable = True
            for action_id, action_str in enumerate(plan):
                simulator_feedback, executable, reached_goal, error_types = self.pddl_env.step(action_instr=action_str)
                completely_executable = False if not executable else completely_executable
                if not executable:

                    if self.enum_plan:
                        feedback = f'The following plan was created based on the strategy{enum_info}:\n{plan_in_feedback}\n'
                    else:
                        feedback = f'The following plan was created based on the strategy{enum_info}:\n{plan_in_feedback}\n'
                    feedback += f'The action {action_str} at step {action_id} is not executable because {simulator_feedback}'

                    self.result_dict["feedback"] = feedback
                    self.result_dict["error-type"].extend(error_types)
                    return

                if reached_goal:
                    reached_goal_at_end = True
                else:
                    reached_goal_at_end = False

        if completely_executable and not reached_goal_at_end:

            goal_feedback = self.pddl_env.goal_feedback
            feedback = f'The following plan was created based on the strategy{enum_info}:\n{plan_in_feedback}\n{goal_feedback}'

            self.result_dict["feedback"] = feedback
            self.result_dict["error-type"].append('not-reached-goal')

        elif reached_goal_at_end:
            feedback = f'The plan is correct'
            self.result_dict['feedback'] = feedback
            self.result_dict['success'] = True

    def get_enum_plan(self, plan: list[str]):

        enum_plan = ''
        for action_num, action in enumerate(plan):
            enum_plan += f'{action}\t(step {action_num})\n'

        return enum_plan


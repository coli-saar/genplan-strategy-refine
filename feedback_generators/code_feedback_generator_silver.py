import os
from typing import Union, List, Tuple
from argparse import Namespace
from utils.tasks import Task
from feedback_generators.code_feedback_generator import CodeFeedbackGenerator
from utils.paths import TEMP_DIR, get_timestamp_for_log


class CodeFeedbackGeneratorBasic(CodeFeedbackGenerator):

    def __init__(self,
                 generalized_plan,
                 task: Union[Task, None],
                 timeout: int,
                 flags: Namespace):

        super().__init__(generalized_plan=generalized_plan,
                         pddl_env_name='',
                         pddl_env_args=dict(),
                         task=task,
                         timeout=timeout,
                         flags=flags)

    def update_pddl_env(self, domain_file_path, instance_file_path):
        """
        This is not needed for the basic Feedback Generator
        :param domain_file_path:
        :param instance_file_path:
        :return:
        """
        return

    def _get_feedback_infinite_loop(self, info: str):

        feedback = f"\nThe code was interrupted because it timed out (possible infinite loop)." \

        self.result_dict['feedback'] = info + feedback
        self.result_dict['error-type'].append('timeout')

    def _get_feedback_generated_plan(self, plan):

        if not isinstance(plan, list):
            feedback = f"The code returned {plan}, which is not a list of actions. "
            self.result_dict["feedback"] = feedback
            self.result_dict["error-type"].append("output-not-plan")
            return

        self.result_dict['plan-length'] = len(plan)

        for t, action in enumerate(plan):
            if not self.task.action_has_valid_syntax(action=action):
                feedback = f"The code returned this plan: {plan}\n" \
                           f"However, the action {action} is invalid at step {t}.\n" \
                           f"NOTE: the valid operators are: {self.task.actions_hint}."

                self.result_dict["feedback"] = feedback
                self.result_dict["error-type"].append('wrong-action')
                return

        plan_is_valid, val_feedback, mistake = validate_plan(task=self.task, plan=plan)

        if plan_is_valid:
            self.result_dict['feedback'] = 'Generalized plan succeeded.'
            self.result_dict['success'] = True
            return

        #feedback = f'Given this task:\n{self.task.problem_str}\n' \
        feedback = f'The code failed. It returned the following plan: {plan}.\n{val_feedback}'
        self.result_dict['feedback'] = feedback
        self.result_dict['success'] = False
        self.result_dict["error-type"].append(mistake)


def validate_plan(task: Task, plan: List[str]) -> Tuple[bool, str, str]:
    """
    Use VAL to check if a plan solves a PDDL problem.
    Return feedback in the same way as Silver et al.
    """
    val = os.environ.get('VAL')

    plan_str = ""
    for t, action in enumerate(plan):
        plan_str += f"{t}: {action}\n"

    # save them in a temporary file
    plan_file = TEMP_DIR / f'./tmp_plan_file_{get_timestamp_for_log()}'
    with open(plan_file, 'w') as pf:
        pf.write(plan_str)

    cmd = f'{val}/validate -v {task.domain_file_path} {task.problem_file_path} {plan_file}'
    val_response = os.popen(cmd).read()

    if os.path.exists(plan_file):
        os.remove(plan_file)

    if "Plan valid" in val_response:
        return True, "Plan succeeded.", ''

    repair_phrase = "Plan Repair Advice:"
    if repair_phrase in val_response:
        msg = val_response[val_response.index(repair_phrase) + len(repair_phrase) : ]
        msg, _ = msg.split("Failed plans:")
        msg = "NOTE: " + msg.strip()
        mistake = "operator-semantics-invalid"
    else:
        msg = "NOTE: The plan did not achieve the goal."
        mistake = 'not-reached-goal'
    return False, msg, mistake

import random
import time
from typing import List, Tuple
from ordered_set import OrderedSet
from itertools import permutations
from argparse import Namespace
from utils.tasks import Task
from agents.gen_plan_execution import GeneralizedPlan, CodeExecutor
from utils.metrics import initialize_task_metrics, GENPLAN_ERROR_TYPES
from utils.helper import create_problem_str


"""
Validate a generalized plan, i.e. a python code and provide feedback
"""


class GenPlanValidator:

    def __init__(self,
                 genplan_code: str,
                 feedback_gen_name: str,
                 feedback_gen_args: dict,
                 flags: Namespace,
                 validation_tasks: List[Task],
                 include_pos_results: bool = True,
                 randomize_input_order: bool = True,
                 n_randomizations: int = 4):

        self.generalized_plan = GeneralizedPlan(code_str=genplan_code)
        self.code_executor = CodeExecutor(gen_plan=self.generalized_plan,
                                          code_feedback_gen_name=feedback_gen_name,
                                          code_feedback_gen_args=feedback_gen_args,
                                          flags=flags)
        self.validation_tasks = validation_tasks
        self.flags = flags
        self.include_pos_results = include_pos_results
        self.randomize_input_order = randomize_input_order
        self.n_randomizations = n_randomizations

        self.plan_succeeded_all = False
        self.plan_succeeded_so_far = True
        self.parsing_error_found = False

        self.metrics = self.initialize_metrics()

        self.successful_tasks = dict()
        self.python_error_tasks = dict()
        self.plan_error_tasks = dict()

        self.log_file = ''

    def initialize_metrics(self) -> dict:

        metrics = initialize_task_metrics()
        metrics.update({
            "tasks_metrics": {}
        })

        return metrics

    def update_metrics(self, task_metrics: dict, task_name: str):

        self.metrics['tasks_metrics'][task_name] = task_metrics
        for err in GENPLAN_ERROR_TYPES:
            self.metrics[err] += task_metrics[err]

    def log_generated_code(self, gen_code):
        log_file = self.log_file
        if log_file == '':
            return

        with open(log_file, 'w') as f:
            f.write(f'#----- Generated Python Code -----#\n')
            f.write(gen_code)
            f.write('\n\n\n### Inputs and Outputs ###\n\n\n')

    def log_inputs_outputs(self,
                           task: Task,
                           objects: set,
                           init: set,
                           goal: set,
                           feedback):

        log_file = self.log_file
        if log_file == '':
            return

        with open(log_file, 'a') as f:
            f.write(f'#----- {task.task_name} -----#\n')
            f.write(f'{objects}\n{init}\n{goal}\n\n')
            f.write(f'{feedback}\n\n\n')

    def set_log_file(self, log_file):
        self.log_file = log_file

    def run_validation_all_tasks(self):
        self.log_generated_code(gen_code=self.generalized_plan.code_str)

        for i, task in enumerate(self.validation_tasks):
            print(f'\n--- Executing generated code on task {i}/{len(self.validation_tasks)} ---\n')

            if not self.randomize_input_order:
                successful, feedback, task_metrics_from_run = self.code_executor.run_genplan_on_task(task=task)
                self.log_inputs_outputs(
                    task=task,
                    feedback=feedback,
                    objects=task.objects,
                    init=task.init,
                    goal=task.goal
                )
            else:
                successful, feedback, task_metrics_from_run = self.run_on_randomized_order(task=task)

            task_metrics = dict()       # requires new dictionary to avoid overwriting issues
            task_metrics.update(task_metrics_from_run)
            task_metrics['success'] = successful
            task_metrics['feedback'] = feedback

            if 'The code was interrupted' in feedback:
                #assert task_metrics['python-exception'] == 1
                task_metrics['python-exception'] = 0
                task_metrics['timeout'] = 1

            if task_metrics['python-exception'] and 'AssertionError' in feedback:
                if not 'assert' in self.generalized_plan.code_str:
                    raise ValueError(f'Assertion error occurred when running the generated code on task {task.task_name}: \n{feedback}')

            non_plan_error, plan_error = self.get_error_type_class(task_metrics=task_metrics)
            task_name = task.task_name
            if successful:
                self.successful_tasks[task_name] = {'task': task, 'feedback': feedback}
            else:
                self.plan_succeeded_so_far = False
                if non_plan_error:
                    self.python_error_tasks[task_name] = {'task': task, 'feedback': feedback}
                elif plan_error:
                    self.plan_error_tasks[task_name] = {'task': task, 'feedback': feedback}
                else:
                    #raise ValueError('Validator did not save any mistakes but did not evaluate to True')
                    self.plan_error_tasks[task_name] = {'task': task, 'feedback': feedback}

            self.update_metrics(task_metrics=task_metrics, task_name=task_name)

        if self.plan_succeeded_so_far:
            self.plan_succeeded_all = True

    def run_on_randomized_order(self, task: Task):
        # Run on several input set orderings and only count as solved if solved for
        # all orderings

        input_orderings = get_randomized_inputs(
            task=task,
            n_randomizations=self.n_randomizations
        )
        assert len(input_orderings) != 0

        all_successful = True
        all_not_successful = True

        first_failed_feedback = None
        first_task_metrics_from_run = None

        for (objects, init, goal) in input_orderings:
            successful, feedback, task_metrics_from_run = self.code_executor.run_genplan_on_task(
                task=task,
                objects=objects,
                init_state=init,
                goal_state=goal
            )
            self.log_inputs_outputs(task=task,
                                    feedback=feedback,
                                    objects=objects,
                                    init=init,
                                    goal=goal)

            if not successful:
                all_successful = False

            if not successful and first_failed_feedback is None:
                first_failed_feedback = feedback
                first_task_metrics_from_run = dict()
                first_task_metrics_from_run.update(task_metrics_from_run)

            if successful:
                all_not_successful = False
                #return successful, feedback, task_metrics_from_run
            else:
                # TODO: remove again
                break

        if all_successful:
            task_metrics_from_run['all_same_eval'] = True
            return successful, feedback, task_metrics_from_run
        else:
            if all_not_successful:
                first_task_metrics_from_run['all_same_eval'] = True
            else:
                first_task_metrics_from_run['all_same_eval'] = False
            assert first_failed_feedback is not None and first_task_metrics_from_run is not None
            return False, first_failed_feedback, first_task_metrics_from_run

    def get_error_type_class(self, task_metrics: dict) -> Tuple[bool, bool]:

        relevant_errors = []

        for error_t in GENPLAN_ERROR_TYPES:
            if error_t in task_metrics.keys():
                if task_metrics[error_t] != 0:
                    relevant_errors.append(error_t)

        not_plan_errors = ['timeout',
                           'python-exception',
                           'action-formatting',
                           'output-not-plan']

        non_plan_error = False
        plan_error = False
        for error in relevant_errors:
            if error in not_plan_errors:
                non_plan_error = True
            else:
                plan_error = True

        return non_plan_error, plan_error

    def generate_final_feedback_message(self) -> str:

        if self.plan_succeeded_all:
            return f'The generated code successfully solved all validation tasks.'

        feedback_final = ''
        if self.include_pos_results:
            for success_task_dict in self.successful_tasks.values():
                task = success_task_dict['task']
                feedback = success_task_dict['feedback']
                problem_str = create_problem_str(task=task,
                                                 version='python',
                                                 abbrev_problem=self.flags.abbreviate_problem_strs)
                feedback += f'Given the problem:\n{problem_str}\n{feedback}\n\n'
                feedback_final += feedback

        if len(self.python_error_tasks.keys()) != 0:
            debug_task_name = list(self.python_error_tasks.keys())[0]
        else:
            debug_task_name = list(self.plan_error_tasks.keys())[0]

        task = self.python_error_tasks[debug_task_name]['task']
        problem_str = create_problem_str(task=task,
                                         version='python',
                                         abbrev_problem=self.flags.abbreviate_problem_strs)
        feedback = self.python_error_tasks[debug_task_name]['feedback']
        feedback += f'Given the problem:\n{problem_str}\n{feedback}\n\n'
        feedback_final += feedback

        return feedback_final


def get_randomized_inputs(task: Task, n_randomizations: int):

    object_orderings = []
    init_orderings = []
    goal_orderings = []

    limit = 3 * n_randomizations

    count = 0
    while len(object_orderings) < n_randomizations:
        objects = list(task.objects)
        random.shuffle(objects)
        object_set = OrderedSet(objects)
        if object_set not in object_orderings:
            object_orderings.append(object_set)

        count += 1
        if count == limit:
            break

    count = 0
    while len(init_orderings) < n_randomizations:
        init_state = list(task.init)
        random.shuffle(init_state)
        init_set = OrderedSet(init_state)
        if init_set not in init_orderings:
            init_orderings.append(init_set)

        count += 1
        if count == limit:
            break

    count = 0
    while len(goal_orderings) < n_randomizations:
        goal_state = list(task.goal)
        random.shuffle(goal_state)
        goal_set = OrderedSet(goal_state)
        if goal_set not in goal_orderings:
            goal_orderings.append(goal_set)

        count += 1
        if count == limit:
            break

    if len(goal_orderings) < n_randomizations:
        missing_n = n_randomizations - len(goal_orderings)
        try:
            new_goal_orders = random.sample(goal_orderings, k=missing_n)
        except ValueError:
            new_goal_orders = random.choices(goal_orderings, k=missing_n)
        random.shuffle(new_goal_orders)
        goal_orderings.extend(new_goal_orders)

    if len(object_orderings) < n_randomizations:
        missing_n = n_randomizations - len(object_orderings)
        try:
            new_object_orders = random.sample(object_orderings, k=missing_n)
        except ValueError:
            new_object_orders = random.choices(object_orderings, k=missing_n)
        random.shuffle(new_object_orders)
        object_orderings.extend(new_object_orders)

    if len(init_orderings) < n_randomizations:
        missing_n = n_randomizations - len(init_orderings)
        try:
            new_init_orders = random.sample(init_orderings, k=missing_n)
        except ValueError:
            new_init_orders = random.choices(init_orderings, k=missing_n)
        random.shuffle(new_init_orders)
        init_orderings.extend(new_init_orders)

    input_orderings = []
    for i in range(n_randomizations):

        ordered_goal = goal_orderings[i]
        ordered_objects = object_orderings[i]
        ordered_init = init_orderings[i]

        input_orderings.append((ordered_objects, ordered_init, ordered_goal))

    return input_orderings



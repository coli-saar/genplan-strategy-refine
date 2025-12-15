from typing import List
import time
from argparse import Namespace
from ast import literal_eval
from utils.tasks import Task, TaskSimple
from utils.metrics import GENPLAN_ERROR_TYPES
from agents.gen_plan_execution import GeneralizedPlan, CodeExecutor
from agents.validator_feedback_gen import get_randomized_inputs


def run_evaluation(eval_tasks: List[Task],
                   generalized_plan_code: str,
                   flags: Namespace,
                   use_inference_feedback_gen: bool = True) -> dict:
    """
    run the evaluation on the eval dataset
    :param eval_tasks:
    :param generalized_plan_code:
    :param flags:
    :param use_inference_feedback_gen:
    :return:
    """

    eval_metrics = dict()
    num_successes = 0
    num_eval = len(eval_tasks)
    solved_tasks = []
    not_solved_tasks = []

    print(f'------------- Running Evaluation ---------------')
    inference_feedback_generator = flags.inference_feedback_generator
    inf_feedback_gen_args = inference_feedback_generator['args']
    inf_feedback_gen_args['inference'] = use_inference_feedback_gen

    generalized_plan = GeneralizedPlan(generalized_plan_code)
    code_executor = CodeExecutor(gen_plan=generalized_plan,
                                 code_feedback_gen_name=inference_feedback_generator['name'],
                                 code_feedback_gen_args=inf_feedback_gen_args,
                                 flags=flags)

    for i, eval_task in enumerate(eval_tasks):
        # task_id = f'{eval_task.problem_name}'  # seed and domain name are part of the outputfile path
        task_name = eval_task.task_name

        if not flags.randomize_input_order:
            success, feedback, task_metrics_from_run = code_executor.run_genplan_on_task(task=eval_task)
        # Run on several input set orderings and only count as solved if solved for
        # all orderings
        else:
            input_orderings = get_randomized_inputs(
                task=eval_task,
                n_randomizations=flags.n_randomizations
            )

            for (objects, init, goal) in input_orderings:
                success, feedback, task_metrics_from_run = code_executor.run_genplan_on_task(
                    task=eval_task,
                    objects=objects,
                    init_state=init,
                    goal_state=goal
                )
                if not success:
                    break

        success_str = "Solved" if success else "Failed"
        print(f"SUCCESS: {success_str}")

        if success:
            num_successes += 1

        task_metrics = dict()
        task_metrics.update(task_metrics_from_run)
        task_metrics['success'] = success
        task_metrics['feedback'] = feedback

        conversion_start = time.perf_counter()
        task_2 = TaskSimple(domain_file_path=eval_task.domain_file_path,
                            problem_file_path=eval_task.problem_file_path,
                            print_paths=False)
        conversion_time = time.perf_counter() - conversion_start
        task_metrics['total_time'] = task_metrics['run_time'] + conversion_time

        if 'The code was interrupted' in feedback:
            #assert task_metrics['python-exception'] == 1
            task_metrics['python-exception'] = 0
            task_metrics['timeout'] = 1

        if success:
            # extract the found plan
            assert "The code returned the correct output:\n" in feedback
            generated_plan_str = feedback.replace("The code returned the correct output:\n", '')
            generated_plan = literal_eval(generated_plan_str)
            task_metrics['plan'] = generated_plan
            solved_tasks.append(task_name)
        else:
            task_metrics['plan'] = None
            not_solved_tasks.append(task_name)

        for genplan_error in GENPLAN_ERROR_TYPES:
            if task_metrics[genplan_error] == 0:
                del task_metrics[genplan_error]

        eval_metrics[task_name] = task_metrics

    eval_metrics['successes'] = num_successes
    eval_metrics['accuracy'] = round(num_successes / num_eval, 3)
    eval_metrics['solved_tasks'] = solved_tasks
    eval_metrics['not_solved_tasks'] = not_solved_tasks
    print(eval_metrics)
    return eval_metrics






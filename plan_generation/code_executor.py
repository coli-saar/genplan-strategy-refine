import os
import sys
import time
import signal
import traceback
from copy import deepcopy
import multiprocessing as mp
from typing import Tuple, Union

from agents.gen_plan_execution import make_vars_accessible, GeneralizedPlan
from utils.tasks import Task, TaskSimple
from utils.metrics import initialize_task_metrics
from utils.helper import VariableContainer
from feedback_generators.code_feedback_generator import CodeFeedbackGenerator
from feedback_generators.code_feedback_generator_silver import CodeFeedbackGeneratorBasic


# Needs to be defined the same script where the classes are available through import
def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


class CodeExecutor:

    def __init__(self,
                 gen_plan: GeneralizedPlan,
                 timeout: int = 45):

        self.gen_plan: GeneralizedPlan = gen_plan
        self.timeout = timeout
        self.timed_out = False
        self.task = None

        self.var_container = make_vars_accessible()

        self.result_dict = dict()

    def update_task(self, task: Union[Task, TaskSimple]):
        self.task = task
        domain_file_path, instance_file_path = self.task.get_file_paths()
        self.result_dict = dict()

    def run_genplan_on_task(self,
                            task: Union[Task, TaskSimple],
                            objects: Union[set, None] = None,
                            init_state: Union[set, None] = None,
                            goal_state: Union[set, None] = None) -> dict:

        self.update_task(task=task)
        assert self.task is not None

        start_time = time.perf_counter()
        time_out_happened = False

        if self.timeout is None:
            self._run_genplan_on_task_no_timeout(
                task=self.task,
                result_dict=dict(),
                objects=objects,
                init_state=init_state,
                goal_state=goal_state
            )

        else:
            manager = mp.Manager()
            result_proxy_dict = manager.dict()
            p = mp.Process(target=self._run_genplan_on_task_no_timeout,
                           args=(self.task,
                                 result_proxy_dict,
                                 objects,
                                 init_state,
                                 goal_state
                                 ))
            p.start()
            p.join(self.timeout)

            self.result_dict = result_proxy_dict

            # Timeout reached
            if p.is_alive():
                time_out_happened = True
                # Treated like a KeyboardInterrupt.
                assert p.pid is not None
                os.kill(p.pid, signal.SIGINT)
                # Give it a few more seconds then kill for good.
                p.join(3)
                p.kill()
                self.result_dict["timeout_reached"] = True

        duration = time.perf_counter() - start_time

        if self.timed_out:
            assert self.timeout is not None and duration > self.timeout

        return self.result_dict

    def _run_genplan_on_task_no_timeout(self,
                                        task: Task,
                                        result_dict: dict,
                                        objects: Union[set, None] = None,
                                        init_state: Union[set, None] = None,
                                        goal_state: Union[set, None] = None
                                        ):

        safety_checks = True

        try:
            if objects is None and init_state is None and goal_state is None:
                objects = task.objects
                init_state = task.init
                goal_state = task.goal
            start_time = time.perf_counter()
            plan = self.gen_plan.run(
                objects=objects,
                init=init_state,
                goal=goal_state,
                safety_check=safety_checks)
            run_time = time.perf_counter() - start_time

            result_dict["plan"] = plan

        except BaseException as e:
            run_time = time.perf_counter() - start_time
            exc_type, exc_value, exc_tb = sys.exc_info()

            tb: list = traceback.format_exception(exc_type, value=exc_value, tb=exc_tb)
            result_dict["error_msg"] = tb
            result_dict["plan"] = []

        result_dict["run_time"] = run_time


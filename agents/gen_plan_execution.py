import os
import sys
import time
import signal
import importlib.util
import tempfile
import traceback
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple, Union
from argparse import Namespace

from utils.tasks import Task
from utils.metrics import initialize_task_metrics
from utils.utils import VariableContainer
from feedback_generators.code_feedback_generator import CodeFeedbackGenerator
from feedback_generators.code_feedback_generator_silver import CodeFeedbackGeneratorBasic


# Needs to be defined the same script where the classes are available through import
def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


def make_vars_accessible() -> VariableContainer:
    var_container = VariableContainer()
    return var_container


@dataclass(frozen=True)
class GeneralizedPlan:
    """Wrapper around a generalized plan code string."""

    code_str: str

    @cached_property
    def filepath(self) -> Path:
        """Get a file with the code string implemented in it."""
        filename = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".py").name)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.code_str)
        return filename

    def run(self,
            objects: set,
            init: set,
            goal: set,
            safety_check: bool = True) -> List[str]:
        """Run the generalized plan to get a plan for the task."""

        if safety_check:
            self.run_safety_checks()

        # Import generate_solution
        module_name = f"{self.filepath.stem}"
        spec = importlib.util.spec_from_file_location(module_name, self.filepath)
        assert spec is not None, 'gen_plan_execution.py line 56: spec is None'
        assert spec.loader is not None, 'gen_plan_execution.py line 57: spec.loader is None'
        module = importlib.util.module_from_spec(spec)
        assert module is not None, 'gen_plan_execution.py line 59: module is None'
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Run the generalized plan.
        # TODO: deal with this differently maybe
        if 'def generate_solution' not in self.code_str and 'def get_plan' in self.code_str:
            return_value = module.get_plan(objects, init, goal)
        else:
            return_value = module.generate_solution(objects, init, goal)
        return return_value

    def run_safety_checks(self):

        safety_risks = [
            'import sys',
            'import os',
            'import shutils'
        ]

        for potential_risk in safety_risks:
            if potential_risk in self.code_str:
                raise RuntimeError(f'Attention! The generated code includes {potential_risk} which is a potential safety risk. It will not be executed. If you are sure you trust the code then disable the saftey checks in the config file.')


class CodeExecutor:

    def __init__(self,
                 gen_plan: GeneralizedPlan,
                 code_feedback_gen_name: str,
                 code_feedback_gen_args: dict,
                 flags: Namespace):

        self.gen_plan: GeneralizedPlan = gen_plan
        self.horizon = flags.horizon,
        self.timeout = flags.timeout
        self.flags = flags
        self.timed_out = False
        self.task = None

        self.var_container = make_vars_accessible()

        self.code_feedback_gen: CodeFeedbackGenerator = self.create_feedback_generator(
            feedback_generator_name=code_feedback_gen_name, feedback_generator_args=code_feedback_gen_args)

        self.result_dict = dict()

        self.task_metrics: dict = initialize_task_metrics()
        self.feedback = None
        self.success = None

    def create_feedback_generator(self,
                                  feedback_generator_name: str,
                                  feedback_generator_args: dict) -> CodeFeedbackGenerator:

        feed_gen_args = deepcopy(feedback_generator_args)
        for arg_name, arg_val in feed_gen_args.items():
            if isinstance(arg_val, str):
                if arg_val.startswith('VAR-'):
                    look_up_name = arg_val.replace('VAR-', '')
                    feed_gen_args[arg_name] = self.var_container.get_variable(name=look_up_name)

        feedback_args = {
            'generalized_plan': self.gen_plan,
            'task': None,
            'timeout': self.timeout,
            'flags': self.flags
        }
        # add additional args needed for creating the FeedbackGenerator
        feedback_args.update(feed_gen_args)

        feedback_generator = str_to_class(feedback_generator_name)(**feedback_args)
        return feedback_generator

    def update_task(self, task: Task):
        self.task = task
        domain_file_path, instance_file_path = self.task.get_file_paths()
        self.code_feedback_gen.update_task_and_env(
            task=task,
            domain_file_path=domain_file_path,
            instance_file_path=instance_file_path)
        self.result_dict = dict()
        self.task_metrics = initialize_task_metrics()

    def run_genplan_on_task(self,
                            task: Task,
                            objects: Union[set, None] = None,
                            init_state: Union[set, None] = None,
                            goal_state: Union[set, None] = None) -> Tuple[bool, str, dict]:

        self.update_task(task=task)
        assert self.task is not None
        assert self.code_feedback_gen.task == self.task

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
            self.result_dict = self.code_feedback_gen.result_dict

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
            self.code_feedback_gen.result_dict = self.result_dict

            # Timeout reached
            if p.is_alive():
                time_out_happened = True
                # Treated like a KeyboardInterrupt.
                assert p.pid is not None
                os.kill(p.pid, signal.SIGINT)
                # Give it a few more seconds then kill for good.
                p.join(3)
                p.kill()
                feedback_gen_args = {'version': 'timeout', 'args': {'info': result_proxy_dict.get('info', '')}}
                self.code_feedback_gen.result_dict['feedback_gen_args'] = feedback_gen_args

                if 'run_time' not in self.code_feedback_gen.result_dict.keys():
                    self.code_feedback_gen.result_dict["run_time"] = self.timeout

        duration = time.perf_counter() - start_time

        self.task_metrics["duration"] = duration

        version = self.code_feedback_gen.result_dict['feedback_gen_args']['version']
        feedback_args = self.code_feedback_gen.result_dict['feedback_gen_args']['args']

        self.code_feedback_gen.validate_and_generate_feedback(version=version, args=feedback_args)
        self.task_metrics['error_occurred'] = self.code_feedback_gen.result_dict['error_occurred']

        self.task_metrics["plan-length"] = self.result_dict.get("plan-length", None)
        if self.timed_out:
            assert self.timeout is not None and duration > self.timeout

        if "error-type" in self.result_dict:
            for error_t in self.result_dict["error-type"]:
                self.task_metrics[error_t] += 1

        self.task_metrics["run_time"] = self.code_feedback_gen.result_dict["run_time"]

        if 'success' not in self.result_dict.keys():
            print(self.result_dict)
            print(self.task_metrics)
            print(time_out_happened)

        return self.result_dict['success'], self.result_dict['feedback'], self.task_metrics

    def _run_genplan_on_task_no_timeout(self,
                                        task: Task,
                                        result_dict: dict,
                                        objects: Union[set, None] = None,
                                        init_state: Union[set, None] = None,
                                        goal_state: Union[set, None] = None
                                        ):

        safety_checks = self.flags.__dict__.get('safety_checks', True)

        try:
            if objects is None and init_state is None and goal_state is None:
                objects = task.objects
                init_state = task.init
                goal_state = task.goal
            start_time = time.perf_counter() #time.time()
            plan = self.gen_plan.run(
                objects=objects,
                init=init_state,
                goal=goal_state,
                safety_check=safety_checks)
            run_time = time.perf_counter() - start_time

            feedback_gen_args = {'version': 'plan', 'args': {'plan': plan}}

        except BaseException as e:
            run_time = time.perf_counter() - start_time
            exc_type, exc_value, exc_tb = sys.exc_info()

            tb: list = traceback.format_exception(exc_type, value=exc_value, tb=exc_tb)
            feedback_gen_args = {'version': 'python-exception', 'args': {'traceback': tb}}

        self.code_feedback_gen.result_dict['run_time'] = run_time

        # For some reason, the self.code_feedback_gen object is reset / not changed
        # when the changes happen inside the mp.process call -> make sure it is maintained
        result_dict.update(self.code_feedback_gen.result_dict)
        result_dict['feedback_gen_args'] = feedback_gen_args


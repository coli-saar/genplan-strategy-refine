import sys
from typing import List, Dict, Union
from pathlib import Path
from llm_models.llm_base_class import LLMModel
from agents.agent_strategy_val_planbased import AgentStrategyValidatePlanBased
from utils.tasks import TaskData, update_domain_strategy
from utils.utils import create_prompt_template, log_agent_system_prompt, log_llm_interaction, parse_strategy_response


def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


"""
Different Versions:
Basic:
- generate a strategy 
- requires domain_strategy_prompt_file

Merge domain + task strategies:
- assumes that there is already a first version of the domain-level strategy
- generates a strategy for an individual task instance 
- can merge a domain-level strategy with a task-level strategy into new domain-level strategy
- requires sep_strategy_prompt_file and domain_strategy_merge_prompt_file

Separate strategy gen + merge task strategies:
- first generate a task-level strategy for each task
- then merge the task-level strategy into one domain-level strategy
- requires sep_strategy_prompt_file and domain_strategy_merge_prompt_file (but different prompt than for "merge strategies"
- Currently no longer used
"""


class AgentStrategyGen:

    def __init__(self,
                 llm_model: LLMModel,
                 prompt_task_dict: Dict[str, TaskData],
                 version: str,
                 log_dir: Path,
                 flags: dict,
                 domain_strategy_prompt_file: Union[str, None] = None,
                 sep_strategy_prompt_file: Union[str, None] = None,
                 domain_strategy_merge_prompt_file: Union[str, None] = None,
                 plan_based_val_args: Union[dict, None] = None
                 ):
        """

        :param llm_model:
        :param prompt_task_dict:
        :param version:
        :param log_dir:
        :param flags:
        :param domain_strategy_prompt_file:
        :param sep_strategy_prompt_file:
        :param domain_strategy_merge_prompt_file:
        :param plan_based_val_args:
        """

        self.agent_name = 'StrategyGen'
        self.flags = flags

        self.llm_model = llm_model
        self.prompt_task_dict = prompt_task_dict

        if version != 'nl':
            raise not NotImplementedError
        self.version = version

        self.domain_strategy = None

        if plan_based_val_args:
            val_agent_name = plan_based_val_args.pop('name')
            self.agent_planbased_val: AgentStrategyValidatePlanBased = str_to_class(val_agent_name)(
                **plan_based_val_args)
        else:
            self.agent_planbased_val = None


        self.domain_strategy_prompt_template = self.initialize_prompt_template(domain_strategy_prompt_file)
        self.sep_strategy_prompt_template = self.initialize_prompt_template(sep_strategy_prompt_file)
        self.domain_strategy_merge_prompt_template = self.initialize_prompt_template(domain_strategy_merge_prompt_file)

        if self.domain_strategy_prompt_template is not None:
            self.strategy_gen_version = 'single_step'
            assert self.sep_strategy_prompt_template is None
            assert self.domain_strategy_merge_prompt_template is None
        else:
            self.strategy_gen_version = 'merge'
            assert self.sep_strategy_prompt_template is not None
            assert self.domain_strategy_merge_prompt_template is not None

        # Logging set-up
        self.log_dir = log_dir
        log_agent_system_prompt(log_dir=self.log_dir,
                                llm_model=self.llm_model,
                                agent_name=self.agent_name)

    def log_llm_input_output(self):
        log_llm_interaction(log_dir=self.log_dir,
                            llm_model=self.llm_model,
                            agent_name=self.agent_name,
                            flags=self.flags)

    def initialize_prompt_template(self, prompt_file):
        if prompt_file is None:
            return None

        prompt_template = create_prompt_template(prompt_file=prompt_file, flags=self.flags)
        return prompt_template

    def generate_domain_strategy(self):

        if self.strategy_gen_version == 'single_step':
            prompt_domain_strat = self.get_prompt_single_domain_strat()
            domain_strategy = self._generate_domain_strategy(prompt_domain_strategy=prompt_domain_strat)
        else:
            self.generate_all_task_strategies()
            domain_strategy = self.merge_task_strategies()

        # Get estimate of quality of strategy
        if self.agent_planbased_val is not None:
            update_domain_strategy(
                task_dict=self.agent_planbased_val.task_dict,
                summarized_strategy=domain_strategy
            )
            self.agent_planbased_val.assess_strategy_quality()
            self.agent_planbased_val.save_validation_results()


    def _generate_domain_strategy(self, prompt_domain_strategy) -> str:
        self.llm_model.reset_history()
        domain_strategy, _ = self.llm_model.generate(user_message=prompt_domain_strategy)
        self.log_llm_input_output()

        domain_strategy = parse_strategy_response(model_response=domain_strategy)

        self.domain_strategy = domain_strategy
        update_domain_strategy(task_dict=self.prompt_task_dict,
                               summarized_strategy=domain_strategy)

        return domain_strategy

    def get_prompt_single_domain_strat(self) -> str:

        plans_available = True
        for task_name, task_data in self.prompt_task_dict.items():
            plan = task_data.get_plan_nl()
            if plan is None:
                plans_available = False

        prompt_param = {'problems': [],
                        'incl_plan': plans_available}

        for task_name, task_data in self.prompt_task_dict.items():
            if self.version == 'nl':
                prompt_param['domain_nl'] = task_data.get_domain_nl()
                problem_str = task_data.get_problem_nl()

                if plans_available:
                    plan = task_data.get_plan_nl()
                    problem_str += f'\n\nSolution:\n{plan}'
                prompt_param['problems'].append(problem_str)

        prompt_strategy = self.domain_strategy_prompt_template.render(**prompt_param)
        return prompt_strategy

    def generate_all_task_strategies(self):

        for task_name, task_data in self.prompt_task_dict.items():
            # exclude unsolvable tasks
            if task_data.solvable == 0:
                continue

            self.generate_task_strategy(task_name=task_name, task_data=task_data)

    def generate_task_strategy(self, task_name: str, task_data: TaskData) -> None:

        prompt_param = {}
        prompt_param['domain'] = task_data.get_domain_nl()

        problem_str = task_data.get_problem_nl()
        plan = task_data.get_plan_nl()
        incl_plan = False
        if plan is not None:
            incl_plan = True
            problem_str += f'\n\nSolution:\n{plan}'
        prompt_param['problem'] = problem_str
        prompt_param['incl_plan'] = incl_plan

        prompt_sep_strategy = self.sep_strategy_prompt_template.render(**prompt_param)
        self.llm_model.reset_history()
        task_strategy, _ = self.llm_model.generate(user_message=prompt_sep_strategy)
        self.log_llm_input_output()
        task_strategy = parse_strategy_response(model_response=task_strategy)
        self.prompt_task_dict[task_name].set_strategy_task(task_strategy)


    def merge_task_strategies(self) -> str:
        """
        Generate general domain strategy based on individual, task-specific strategies
        :return:
        """
        assert self.domain_strategy_merge_prompt_template is not None
        prompt_param = {'domain_nl': '',
                        'strategies': []}
        counter = 0
        for task_data in self.prompt_task_dict.values():
            # skip unsolvable task instances
            if task_data.solvable == 0:
                continue

            if prompt_param['domain_nl'] == '':
                prompt_param['domain_nl'] = task_data.get_domain_nl()

            problem_strategy_str = f'Task instance no. {counter}:\n{task_data.get_problem_nl()}\n\n\n'
            problem_strategy_str += f'The strategy for solving task instance no. {counter}\n{task_data.get_strategy_task()}'

            prompt_param['strategies'].append(problem_strategy_str)
            counter += 1

        prompt_sum_strategy = self.domain_strategy_merge_prompt_template.render(**prompt_param)
        domain_strategy = self._generate_domain_strategy(prompt_domain_strategy=prompt_sum_strategy)

        return domain_strategy


    def merge_domain_and_task_strategy(self, new_task_name: str):
        """
        Generate general domain strategy based on a general domain strategy and a task-specific strategy
        :param new_task_name:
        :return:
        """
        assert self.domain_strategy_merge_prompt_template is not None
        new_task_data = self.prompt_task_dict[new_task_name]
        domain_nl = new_task_data.get_domain_nl()
        assert self.domain_strategy is not None
        domain_strategy = self.domain_strategy

        problem_strategy_str = f'Task description:\n{new_task_data.get_problem_nl()}\n\n'
        problem_strategy_str += f'Task-specific strategy:\n{new_task_data.get_strategy_task()}'

        prompt_param = {'domain_nl': domain_nl,
                        'domain_strategy': domain_strategy,
                        'problem_strategy_str': problem_strategy_str}

        prompt_sum_strategy = self.domain_strategy_merge_prompt_template.render(**prompt_param)
        domain_strategy = self._generate_domain_strategy(prompt_domain_strategy=prompt_sum_strategy)

        return domain_strategy

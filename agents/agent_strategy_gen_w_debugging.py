import random
import sys
from typing import Dict, Union
from pathlib import Path
from llm_models.llm_base_class import LLMModel
from agents.agent_strategy_gen import AgentStrategyGen
from agents.agent_strategy_val_planbased import AgentStrategyValidatePlanBased
from utils.tasks import TaskData, update_domain_strategy
from utils.helper import create_prompt_template, log_agent_system_prompt, log_llm_interaction, parse_strategy_response


def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


class AgentStrategyGenDebugging(AgentStrategyGen):
    # what about plan_based_val_args: should work in the same way as without debugging
    # what about debug_tasks: can remove them; are part of the plan_based_val_args
    def __init__(self,
                 llm_model: LLMModel,
                 llm_model_debug: LLMModel,
                 prompt_task_dict: Dict[str, TaskData],
                 version: str,
                 log_dir: Path,
                 flags: dict,
                 reflection_prompt_file: str,
                 refine_strategy_prompt_file: str,
                 max_debug_steps: int,
                 domain_strategy_prompt_file: Union[str, None] = None,
                 sep_strategy_prompt_file: Union[str, None] = None,
                 domain_strategy_merge_prompt_file: Union[str, None] = None,
                 plan_based_val_args: Union[dict, None] = None,
                 select_best_strat: bool = True
                 ):

        super().__init__(llm_model=llm_model,
                         prompt_task_dict=prompt_task_dict,
                         version=version,
                         log_dir=log_dir,
                         flags=flags,
                         domain_strategy_prompt_file=domain_strategy_prompt_file,
                         sep_strategy_prompt_file=sep_strategy_prompt_file,
                         domain_strategy_merge_prompt_file=domain_strategy_merge_prompt_file,
                         plan_based_val_args=plan_based_val_args)

        self.agent_name = 'StrategyGenDebug'

        self.max_debug_steps = max_debug_steps
        self.debug_attempt = 0
        self.llm_model_debug = llm_model_debug

        self.reflection_prompt_template = create_prompt_template(reflection_prompt_file, flags=self.flags)
        self.refine_strategy_prompt_template = create_prompt_template(refine_strategy_prompt_file, flags=self.flags)

        self.select_best_strat = select_best_strat
        self.best_strategy = None
        self.acc_best_strategy = None

        self.last_failed_task_name = ''

        assert self.agent_planbased_val is not None

    def log_debug_llm_input_output(self):
        log_llm_interaction(log_dir=self.log_dir,
                            llm_model=self.llm_model_debug,
                            agent_name=self.agent_name,
                            flags=self.flags)

    def generate_domain_strategy(self):
        print(f'----- Generating First Strategy -----')
        first_strategy = self.generate_first_strategy()
        update_domain_strategy(task_dict=self.agent_planbased_val.task_dict,
                               summarized_strategy=first_strategy)

        # Run LLM-based validation
        _, accuracy = self.agent_planbased_val.assess_strategy_quality()
        self.agent_planbased_val.save_validation_results()

        self.best_strategy = first_strategy
        self.acc_best_strategy = accuracy

        # Get accuracy and if all correct then return
        if accuracy == 1.0:
            return

        # Run self-reflection and debugging
        while self.debug_attempt < self.max_debug_steps:

            print(f'---- Running Debugging on Strategy -----')

            self.debug_attempt += 1

            self.llm_model_debug.reset_history()

            reflection_response = self.run_self_reflection()

            # Refine strategy
            refined_strategy = self.run_refine_strategy()
            update_domain_strategy(task_dict=self.agent_planbased_val.task_dict,
                                   summarized_strategy=refined_strategy)
            update_domain_strategy(task_dict=self.prompt_task_dict,
                                   summarized_strategy=refined_strategy)
            self.domain_strategy = refined_strategy

            # Run LLM-based validation (model history for the plan generation is reset automatically)
            _, accuracy = self.agent_planbased_val.assess_strategy_quality()
            self.agent_planbased_val.save_validation_results()

            if accuracy >= self.acc_best_strategy:
                self.acc_best_strategy = accuracy
                self.best_strategy = refined_strategy

            # Get accuracy and if all correct then return
            if accuracy == 1.0:
                return

        if self.select_best_strat:
            self.update_for_best_strategy()

    def update_for_best_strategy(self):
        update_domain_strategy(task_dict=self.prompt_task_dict,
                               summarized_strategy=self.best_strategy)
        update_domain_strategy(task_dict=self.agent_planbased_val.task_dict,
                               summarized_strategy=self.best_strategy)
        self.domain_strategy = self.best_strategy

    def generate_first_strategy(self):
        if self.strategy_gen_version == 'single_step':
            prompt_domain_strat = self.get_prompt_single_domain_strat()
            domain_strategy = self._generate_domain_strategy(prompt_domain_strategy=prompt_domain_strat)
        else:
            self.generate_all_task_strategies()
            domain_strategy = self.merge_task_strategies()

        return domain_strategy

    def run_self_reflection(self) -> str:
        prompt_param = self.get_param_reflection_prompt()
        prompt = self.reflection_prompt_template.render(**prompt_param)

        response, _ = self.llm_model_debug.generate(user_message=prompt)
        self.log_debug_llm_input_output()
        return response

    def get_param_reflection_prompt(self) -> dict:
        random_task = list(self.prompt_task_dict.values())[0]

        if random_task.task.typed:
            action_hint = random_task.task.actions_hint_typed
        else:
            action_hint = random_task.task.actions_hint
        prompt_param = {
            "domain_nl": random_task.get_domain_nl(),
            "domain_strategy": self.domain_strategy,
            "action_hint": action_hint
        }

        # Check whether problems were solved, get nl descriptions and plans
        solved_problems_tup = []
        not_solved_problems_tup = dict()
        not_solved_problems_names = []

        for task_name, val_res in self.agent_planbased_val.validation_results.items():
            problem_nl = self.agent_planbased_val.task_dict[task_name].get_problem_nl()
            gen_plan = val_res['generated_plan']
            if val_res['valid']:
                if gen_plan.strip() == '':
                    gen_plan = ''
                solved_problems_tup.append((problem_nl, gen_plan))
            else:
                feedback = val_res['feedback']
                not_solved_problems_tup[task_name] = (problem_nl, gen_plan, feedback)
                not_solved_problems_names.append(task_name)

        solved_problems = []
        for prob_enum, problem_tup in enumerate(solved_problems_tup):
            solved_problems.append(f'"""\nTask instance {prob_enum}:\n{problem_tup[0]}\n\nCorrect generated solution:\n{problem_tup[1]}\n"""')
        prompt_param['solved_problems'] = solved_problems

        # Get one of the not solved problem instances and the feedback
        if self.last_failed_task_name != '' and self.last_failed_task_name in not_solved_problems_names:
            failed_problem_name = not_solved_problems_tup[self.last_failed_task_name]
        else:
            failed_problem_name = random.choice(not_solved_problems_names)

        failed_problem_tup = not_solved_problems_tup[failed_problem_name]
        self.last_failed_task_name = failed_problem_name

        prompt_param['failed_problem'] = failed_problem_tup[0]
        prompt_param['feedback'] = failed_problem_tup[2]

        return prompt_param

    def run_refine_strategy(self) -> str:

        prompt = self.refine_strategy_prompt_template.render()
        response, _ = self.llm_model_debug.generate(user_message=prompt)
        self.log_debug_llm_input_output()

        response = parse_strategy_response(model_response=response)

        return response
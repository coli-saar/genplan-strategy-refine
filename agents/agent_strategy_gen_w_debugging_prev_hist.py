import random
import sys
from typing import List, Dict, Union
from pathlib import Path
from llm_models.llm_base_class import LLMModel
from agents.agent_strategy_gen_w_debugging import AgentStrategyGenDebugging
from agents.agent_strategy_val_planbased import AgentStrategyValidatePlanBased
from utils.tasks import TaskData, update_domain_strategy


def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


class AgentStrategyGenDebuggingPlanGenHist(AgentStrategyGenDebugging):
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
                         llm_model_debug=llm_model_debug,
                         prompt_task_dict=prompt_task_dict,
                         version=version,
                         log_dir=log_dir,
                         flags=flags,
                         domain_strategy_prompt_file=domain_strategy_prompt_file,
                         sep_strategy_prompt_file=sep_strategy_prompt_file,
                         domain_strategy_merge_prompt_file=domain_strategy_merge_prompt_file,
                         plan_based_val_args=plan_based_val_args,
                         reflection_prompt_file=reflection_prompt_file,
                         refine_strategy_prompt_file=refine_strategy_prompt_file,
                         max_debug_steps=max_debug_steps,
                         select_best_strat=select_best_strat
                         )

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

    def run_self_reflection(self) -> str:
        prompt_param = self.get_param_reflection_prompt()
        prompt = self.reflection_prompt_template.render(**prompt_param)

        # Add the history of the LLM which generated the wrong plan
        failed_problem_name = self.last_failed_task_name
        plan_gen_hist = self.agent_planbased_val.plan_gen_llm_hists[failed_problem_name]

        if len(plan_gen_hist) != 3:
            print(plan_gen_hist)
        assert len(plan_gen_hist) == 3
        self.llm_model_debug.update_history(new_history=plan_gen_hist)

        response, _ = self.llm_model_debug.generate(user_message=prompt)
        self.log_debug_llm_input_output()
        return response

    def get_param_reflection_prompt(self) -> dict:

        # Check whether problems were solved, get nl descriptions and plans
        not_solved_problems_tup = dict()
        not_solved_problems_names = []

        for task_name, val_res in self.agent_planbased_val.validation_results.items():
            problem_nl = self.agent_planbased_val.task_dict[task_name].get_problem_nl()
            gen_plan = val_res['generated_plan']
            if not val_res['valid']:

                feedback = val_res['feedback']
                not_solved_problems_tup[task_name] = (problem_nl, gen_plan, feedback)
                not_solved_problems_names.append(task_name)

        # Get one of the not solved problem instances and the feedback
        if self.last_failed_task_name != '' and self.last_failed_task_name in not_solved_problems_names:
            failed_problem_name = self.last_failed_task_name
        else:
            failed_problem_name = random.choice(not_solved_problems_names)

        failed_problem_tup = not_solved_problems_tup[failed_problem_name]
        self.last_failed_task_name = failed_problem_name

        prompt_param = {
            "feedback": failed_problem_tup[2]
        }

        return prompt_param

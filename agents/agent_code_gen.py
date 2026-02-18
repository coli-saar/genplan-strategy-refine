import os
import sys
from typing import Dict, List, Union
from pathlib import Path
from llm_models.llm_base_class import LLMModel
from agents.agent_code_gen_basic import AgentCodeGenerationBasic, postprocess_response
from agents.validator_feedback_gen import GenPlanValidator
from agents.agent_strategy_val_planbased import AgentStrategyValidatePlanBased
from utils.tasks import TaskData
from utils.helper import create_prompt_template


def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


class AgentCodeGeneration(AgentCodeGenerationBasic):

    def __init__(self,
                 llm_model: LLMModel,
                 validation_task_dict: Dict[str, TaskData],
                 debug_task_names: List[str],
                 code_gen_prompt_file: str,
                 reflection_prompt_file: str,
                 refine_code_prompt_file: str,
                 validator_param: dict,
                 log_dir: Path,
                 flags: dict,
                 plan_based_val_args: Union[dict, None] = None,
                 use_llm_gen_tasks: bool = True,
                 include_plans: bool = True,
                 include_nl: bool = False,
                 only_symb_plans: bool = False,
                 typed_hints: bool = False,
                 number_of_examples: int = 1,
                 force_examples: bool = False,
                 max_debug_steps: int = 10,
                 log_code_exec: bool = False,
                 debug_hist_len: Union[int, None] = None):
        """

        :param llm_model:
        :param validation_task_dict:
        :param code_gen_prompt_file:
        :param reflection_prompt_file:
        :param refine_code_prompt_file:
        :param validator_param:
        :param log_dir:
        :param flags:
        :param plan_based_val_args:
        :param use_llm_gen_tasks: if True, then also self-generated tasks are used                      during debugging
        :param typed_hints:
        :param number_of_examples:
        :param force_examples: if False then examples will only be included if there exist plans that the LLM generated according to the strategy
                             else number_of_examples examples will be selected where LLM-generated ones are preferred but not only those are selected
        :param max_debug_steps:
        """

        super().__init__(llm_model=llm_model,
                         validation_task_dict=validation_task_dict,
                         debug_task_names=debug_task_names,
                         code_gen_prompt_file=code_gen_prompt_file,
                         refine_code_prompt_file=refine_code_prompt_file,
                         validator_param=validator_param,
                         log_dir=log_dir,
                         flags=flags,
                         use_llm_gen_tasks=use_llm_gen_tasks,
                         include_plans=include_plans,
                         include_nl=include_nl,
                         only_symb_plans=only_symb_plans,
                         typed_hints=typed_hints,
                         number_of_examples=number_of_examples,
                         force_examples=force_examples,
                         max_debug_steps=max_debug_steps,
                         log_code_exec=log_code_exec,
                         debug_hist_len=debug_hist_len)

        self.debugs_pseudocode = 0

        self.reflection_prompt_template = create_prompt_template(reflection_prompt_file, self.flags)

        if plan_based_val_args:
            val_agent_name = plan_based_val_args.pop('name')
            self.agent_planbased_val: AgentStrategyValidatePlanBased = str_to_class(val_agent_name)(
                **plan_based_val_args)
        else:
            self.agent_planbased_val = None

    def _generate_first_code(self):

        # Generate code
        code_response = self.generate_code()

        # Pass the code to the validator and
        # validate on all debugging and validation tasks
        self.run_and_update_validators(code_to_validate=code_response)

    def _run_debugging_loop(self):

        # Run debugging loop
        while self.debug_attempt < self.max_debug_steps:
            self.debug_attempt += 1
            self.required_debug_steps += 1
            self.step_last_validator += 1

            # Take feedback and information about solved and failed problem and run self-reflection
            pseudo_code_wrong = None
            reflection_response = self.run_self_reflection()

            # fix the code and try again
            refined_code = self.refine_code()

            # Pass the code to the validator and
            # validate on all debugging and validation tasks
            self.run_and_update_validators(code_to_validate=refined_code)

            if self.last_validator_debug.plan_succeeded_all:
                return

    def generate_code(self) -> str:

        prompt_param = self.get_param_main_prompt()
        prompt = self.code_gen_prompt_template.render(**prompt_param)

        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output(code=True)
        response = postprocess_response(model_response=response)

        self.last_code = response

        return response

    def run_self_reflection(self):
        """

        :return:
        """

        prompt_param = self.get_param_feedback_prompt()

        prompt = self.reflection_prompt_template.render(**prompt_param)
        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output()
        return response

    def refine_code(self) -> str:

        self.debugs_code += 1

        prompt = self.refine_code_prompt_template.render()
        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output(code=True)

        response = postprocess_response(model_response=response)

        self.last_code = response

        if self.debug_hist_len == 0 and self.debugs_code > 0:
            self.shorten_debug_hist()

        return response

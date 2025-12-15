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
                 pseudocode_prompt_file: Union[str, None],
                 code_gen_prompt_file: str,
                 reflection_prompt_file: str,
                 refine_pseudocode_prompt_file: Union[str, None],
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
        :param pseudocode_prompt_file:
        :param code_gen_prompt_file:
        :param reflection_prompt_file:
        :param refine_pseudocode_prompt_file:
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

        self.pseudocode_prompt_template = create_prompt_template(pseudocode_prompt_file, self.flags) if pseudocode_prompt_file else None
        self.refine_pseudocode_prompt_template = create_prompt_template(refine_pseudocode_prompt_file, self.flags) if refine_pseudocode_prompt_file else None

        self.reflection_prompt_template = create_prompt_template(reflection_prompt_file, self.flags)

        if plan_based_val_args:
            val_agent_name = plan_based_val_args.pop('name')
            self.agent_planbased_val: AgentStrategyValidatePlanBased = str_to_class(val_agent_name)(
                **plan_based_val_args)
        else:
            self.agent_planbased_val = None


    def _generate_first_code(self):

        if self.pseudocode_prompt_template:
            # Generate the pseudocode
            pseudo_code_response = self.generate_pseudo_code()

            # Get estimate of quality of strategy, i.e. pseudocode
            if self.agent_planbased_val is not None:
                self.agent_planbased_val.agent_plan_gen.update_pseudo_code(pseudo_code=pseudo_code_response)
                self.agent_planbased_val.assess_strategy_quality()
                self.agent_planbased_val.save_validation_results()

        # Generate code
        code_response = self.generate_code()

        # Pass the code to the validator and
        # validate on all debugging and validation tasks
        self.run_and_update_validators(code_to_validate=code_response)

    def _run_debugging_loop(self):

        # Run debugging loop
        while self.debug_attempt <= self.max_debug_steps:
            self.debug_attempt += 1
            self.required_debug_steps += 1
            self.step_last_validator += 1

            # Take feedback and information about solved and failed problem and run self-reflection
            pseudo_code_wrong = None
            reflection_response = self.run_self_reflection()

            # if only self-reflection + correction for code but not for pseudocode
            if self.refine_pseudocode_prompt_template is None:
                pseudo_code_wrong = False
            # if LLM decides that pseudocode is wrong
            elif 'the pseudocode needs to be revised' in reflection_response.lower():
                pseudo_code_wrong = True
            # if LLM decides that code is wrong
            elif 'the python code needs to be revised' in reflection_response.lower():
                pseudo_code_wrong = False
            # if LLM does not reply with an output that can be parsed
            else:
                counter = 0
                while pseudo_code_wrong is None:
                    if counter >= 5:
                        raise ValueError('LLM is not stating whether code or pseudocode needs to be refined; check outputs')
                    re_prompt = 'Please tell me explicitly what needs to be revised. Answer only with one single word. "pseudocode" for revising the pseudocode or "code" for revising the python code'
                    new_response, _ = self.llm_model.generate(user_message=re_prompt)
                    new_response = new_response.strip().lower()
                    self.log_llm_input_output()
                    if new_response == 'pseudocode':
                        pseudo_code_wrong = True
                    elif new_response == 'code':
                        pseudo_code_wrong = False
                    counter += 1

            # If place for fixing error is in strategy: fix the pseudocode and try again
            if pseudo_code_wrong:
                self.refine_pseudo_code()

                # Update the history
                current_history = self.llm_model.get_history()
                len_initial_hist = len(self.llm_model.get_initial_history())
                # We want to keep
                # 1) the initial history,
                # 2) the first prompt for generating the pseudocode,
                # 3) then newly generated pseudocode
                new_history = current_history[:len_initial_hist + 1]
                new_history.append(current_history[-1])

                self.llm_model.update_history(new_history=new_history)

                # Generate code
                code_response = self.generate_code()

                # Pass the code to the validator and
                # validate on all debugging and validation tasks
                self.run_and_update_validators(code_to_validate=code_response)

            # If place for fixing is the code itself: fix the code and try again
            else:
                # TODO: option to reduce the history


                refined_code = self.refine_code()

                # Pass the code to the validator and
                # validate on all debugging and validation tasks
                self.run_and_update_validators(code_to_validate=refined_code)

            if self.last_validator_debug.plan_succeeded_all:
                return

    def generate_pseudo_code(self) -> str:

        prompt_param = self.get_param_main_prompt()

        prompt = self.pseudocode_prompt_template.render(**prompt_param)
        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output()

        self.last_pseudocode = response

        return response

    def generate_code(self) -> str:

        # If pseudocode gets generated, then main information is in the pseudocode prompt
        # therefore, not parameters are needed for the code prompt
        if self.pseudocode_prompt_template:
            prompt = self.code_gen_prompt_template.render()

        # Otherwise, the main parameters are needed
        else:
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

    def refine_pseudo_code(self) -> str:

        self.debugs_pseudocode += 1

        prompt = self.refine_pseudocode_prompt_template.render()
        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output()

        self.last_pseudocode = response

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

    def shorten_debug_hist(self):

        current_history = self.llm_model.get_history()
        len_initial_hist = len(self.llm_model.get_initial_history())

        if self.pseudocode_prompt_template is None:
            assert len(current_history) == 7
            new_history = current_history[:len_initial_hist + 1]
        else:
            assert len(current_history) == 9
            new_history = current_history[:len_initial_hist + 3]

        last_code = self.last_code
        last_code_formatted = f'```python\n{last_code}\n```'

        # TODO: decide
        new_history.append({'role': 'assistant', 'content': last_code_formatted})
        #new_history.append(current_history[-1])

        self.llm_model.update_history(new_history=new_history)

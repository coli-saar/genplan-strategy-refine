import os
import sys
import json
from copy import copy
from typing import Dict, Tuple, Union
from pathlib import Path
from llm_models.llm_base_class import LLMModel
from agents.agent_plan_generation import AgentPlanGen, AgentPlanGenPseudoCode
from feedback_generators.feedback_generator_plan import BasicFeedbackGenerator
from utils.tasks import TaskData
from utils.utils import log_agent_system_prompt


"""
Agent for assessing the quality of the strategy or pseudocode based on generated PDDL plans
Does the following:
- for each valid problem (only solvable or also unsolvable?)
    - follow the strategy to come up with a solution
    - convert this into a PDDL plan
    - validate the PDDL plans
- check how many valid plans we have and return whether above or below threshold

"""


# Needs to be defined the same script where the classes are available through import
def str_to_class(class_name: str):
    return getattr(sys.modules[__name__], class_name)


class AgentStrategyValidatePlanBased:

    def __init__(self,
                 llm_model: LLMModel,
                 plan_gen_args: Union[str, dict],
                 task_dict: Dict[str, TaskData],
                 log_dir: Path,
                 llm_gen_dir: Path,
                 res_dir: Path,
                 flags: dict,
                 feedback_gen_args: Union[dict, None] = None,
                 threshold: float = 0.0,
                 typed_hints: bool = False):
        """

        :param llm_model:
        :param plan_gen_args:
                if a dict: then it should contain all parameters for creating an                instances of the AgentPlanGen class or a subclass
                if a string: then it should be the path to the plan generation prompt
                            and a default agent will be created, i.e. an instance
                            of the AgentPlanGen class with default parameter
        :param task_dict:
        :param log_dir:
        :param llm_gen_dir:
        :param res_dir:
        :param flags:
        :param feedback_gen_args:
        :param threshold:
        :param typed_hints:
        """
        self.agent_name = 'StrategyValPlan'
        self.flags = flags
        self.typed_hints = typed_hints
        self.threshold = threshold

        self.llm_model = llm_model
        self.task_dict = task_dict
        self.llm_gen_dir = llm_gen_dir

        # Logging set-up
        self.log_dir = log_dir
        log_agent_system_prompt(log_dir=self.log_dir,
                                llm_model=self.llm_model,
                                agent_name=self.agent_name)

        self.agent_plan_gen: Union[AgentPlanGen, AgentPlanGenPseudoCode] = self.set_up_plan_gen(plan_gen_args)
        self.feedback_generator = self.set_up_feedback_gen(feedback_gen_args=feedback_gen_args)

        self.res_dir = res_dir
        self.validation_results = self.set_up_results_dict()
        self.validation_inputs = dict()
        self.plan_gen_llm_hists = dict()

    def set_up_results_dict(self) -> Dict[str, dict]:
        res_dict = dict()
        for task_name in self.task_dict.keys():
            res_dict[task_name] = {'valid': None,
                                   'feedback': None,
                                   'generated_plan': None}

        return res_dict

    def set_up_feedback_gen(self, feedback_gen_args: Union[dict, None]) -> Union[BasicFeedbackGenerator, None]:

        if feedback_gen_args is None:
            return None

        feedback_gen_args['flags'] = self.flags
        feedback_gen_args['task'] = None
        feedback_generator = BasicFeedbackGenerator(**feedback_gen_args)

        return feedback_generator

    def set_up_plan_gen(self, plan_gen_args: Union[str, dict]) \
            -> Union[AgentPlanGen, AgentPlanGenPseudoCode]:

        if isinstance(plan_gen_args, dict):
            agent_name = plan_gen_args.pop('name')
            plan_gen_args['typed_hints'] = self.typed_hints

        elif isinstance(plan_gen_args, str):
            plan_gen_prompt_file = plan_gen_args
            plan_gen_args = {
                'llm_model': self.llm_model,
                'plan_gen_prompt_file': plan_gen_prompt_file,
                'typed_hints': self.typed_hints,
                'llm_gen_dir': self.llm_gen_dir,
                'flags': self.flags
            }
            agent_name = 'AgentPlanGen'
        else:
            raise TypeError

        log_dir = self.log_dir / 'planbased_validation'
        log_dir.mkdir(exist_ok=True, parents=True)
        plan_gen_args['log_dir'] = log_dir
        plan_gen_args['increase_counter'] = False

        agent_plan_gen = str_to_class(agent_name)(**plan_gen_args)

        return agent_plan_gen

    def assess_strategy_quality(self) -> Tuple[bool, float]:

        valid_task_list = []

        for task_name, task_data in self.task_dict.items():

            # skip unsolvable instances
            if task_data.solvable == 0:
                continue

            plan, valid, feedback = self.generate_and_validate_plan(task_data)
            self.validation_results[task_name]['valid'] = valid
            self.validation_results[task_name]['feedback'] = feedback
            self.validation_results[task_name]['generated_plan'] = plan
            valid_task_list.append(valid)

        ratio_valid = sum(valid_task_list) / len(valid_task_list)

        if ratio_valid < self.threshold:
            return False, ratio_valid
        else:
            return True, ratio_valid

    def generate_and_validate_plan(self, task_data: TaskData) -> Tuple[str, bool, str]:

        if self.feedback_generator is None:
            # Do everything using the methods of the AgentPlanGen, i.e.
            # generate plan, validate using val
            # if valid update the generated_plan attribute of the task accordingly and save the plan
            generated_plan_str, valid = self.agent_plan_gen.generate_and_validate_plan(task_data=task_data)
            feedback = 'The generated plan does not solve the task.' if not valid\
                else 'The generated plan successfully solves the task.'
        else:
            generated_plan_str = self.agent_plan_gen.generate_plan(task_data=task_data)
            self.feedback_generator.update_task_and_env(task=task_data.task,
                                                        pddl_plan=generated_plan_str)
            self.feedback_generator.get_feedback_generated_plan()
            valid = self.feedback_generator.result_dict['success']
            self.agent_plan_gen.update_task_plan(valid=valid)
            feedback = self.feedback_generator.result_dict['feedback']

        if valid:
            # save valid plan
            self.agent_plan_gen.save_new_plan()

        # keep track of the llm input
        model_hist = self.agent_plan_gen.llm_model.get_history()
        self.plan_gen_llm_hists[task_data.task_name] = copy(model_hist)

        assert len(model_hist) == 3
        last_out = model_hist.pop(-1)
        last_in = model_hist[-1]
        assert last_in['role'] == 'user'
        assert last_out['role'] == 'assistant'
        self.validation_inputs[task_data.task_name] = model_hist

        return generated_plan_str, valid, feedback

    def save_validation_results(self):

        val_res_file = self.res_dir / 'intermediate_validation_results.json'

        if os.path.exists(val_res_file):
            with open(val_res_file, 'r') as f:
                prev_val_results = json.load(f)
        else:
            prev_val_results = dict()

        new_counter = self.flags['llm_counter'] - 1
        # reduce by 1 because counter is always already increased after saving an interaction
        prev_val_results[new_counter] = self.validation_results

        with open(val_res_file, 'w') as f:
            json.dump(prev_val_results, f, indent=2)

        # Also save the parameters for the prompt in order to run additional runs later
        val_input_file = self.res_dir / 'intermediate_validation_inputs.json'

        if os.path.exists(val_input_file):
            with open(val_input_file, 'r') as f:
                prev_inputs = json.load(f)
        else:
            prev_inputs = dict()

        prev_inputs[new_counter] = self.validation_inputs

        with open(val_input_file, 'w') as f:
            json.dump(prev_inputs, f, indent=2)


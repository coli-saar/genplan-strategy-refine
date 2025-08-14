from pathlib import Path
from typing import Union
from collections import defaultdict
from llm_models.llm_base_class import LLMModel
from utils.tasks import TaskData
from utils.utils import create_prompt_template, run_validator, log_llm_interaction, log_llm_interaction_validation_counter
from utils.paths import get_llm_gen_plan_dir


class AgentPlanGen:

    def __init__(self,
                 llm_model: LLMModel,
                 plan_gen_prompt_file: str,
                 typed_hints: bool,
                 log_dir: Path,
                 llm_gen_dir: Path,
                 flags: dict,
                 increase_counter: bool = True,
                 input_version: str = 'nl',
                 allow_refinement: bool = False):
        self.agent_name = 'PlanGen'

        self.plan_gen_prompt_template = create_prompt_template(plan_gen_prompt_file, flags=flags)
        self.typed_hints = typed_hints
        self.llm_model = llm_model
        self.llm_gen_dir = llm_gen_dir
        self.allow_refinement = allow_refinement

        self.log_dir = log_dir
        self.flags = flags
        self.increase_counter = increase_counter

        self.task_data: Union[None, TaskData] = None
        self.generated_plan = None

        self.input_version = input_version

        self.results_overview = defaultdict(dict)

    def log_llm_input_output(self):
        if self.increase_counter:
            log_llm_interaction(log_dir=self.log_dir,
                                llm_model=self.llm_model,
                                agent_name=self.agent_name,
                                flags=self.flags)
        else:
            log_llm_interaction_validation_counter(log_dir=self.log_dir,
                                                   llm_model=self.llm_model,
                                                   agent_name=self.agent_name,
                                                   flags=self.flags)

    def reset_for_new_task(self):
        self.task_data = None
        self.generated_plan = None
        self.llm_model.reset_history()

    def create_prompt_param(self) -> dict:

        if self.typed_hints:
            action_hint = self.task_data.task.actions_hint_typed
        else:
            action_hint = self.task_data.task.actions_hint

        if self.input_version == 'nl':
            prompt_param = {'domain_descrip': self.task_data.get_domain_nl(),
                            'domain_strategy': self.task_data.get_strategy_domain(),
                            'problem_descrip': self.task_data.get_problem_nl(),
                            'problem_pddl': None,
                            'action_hint': action_hint}

        elif self.input_version == 'nl_pddl':
            prompt_param = {'domain_descrip': self.task_data.get_domain_nl(),
                            'domain_strategy': self.task_data.get_strategy_domain(),
                            'problem_descrip': self.task_data.get_problem_nl(),
                            'problem_pddl': self.task_data.get_problem_str(),
                            'action_hint': action_hint}
        else:
            raise ValueError

        return prompt_param

    def generate_plan(self, task_data: TaskData) -> str:

        self.reset_for_new_task()
        self.task_data = task_data

        prompt_param = self.create_prompt_param()
        prompt_plan_gen = self.plan_gen_prompt_template.render(**prompt_param)

        response, _ = self.llm_model.generate(user_message=prompt_plan_gen)
        self.log_llm_input_output()

        generated_solution = response.split('Solution:')[-1]
        lines = generated_solution.strip().split('\n')
        generated_plan = []
        for line in lines:
            if line.startswith('('):
                generated_plan.append(line)
        generated_plan_str = '\n'.join(generated_plan)
        self.generated_plan = generated_plan_str

        self.results_overview[task_data.task_name]['llm_plan'] = generated_plan

        return generated_plan_str

    def validate_plan(self) -> bool:

        valid = run_validator(task=self.task_data.task, plan=self.generated_plan)
        return valid

    def generate_and_validate_plan(self, task_data: TaskData):
        generated_plan: str = self.generate_plan(task_data=task_data)
        valid = self.validate_plan()
        self.update_task_plan(valid=valid)
        if valid:
            self.save_new_plan()

        self.results_overview[task_data.task_name]['symb_val'] = valid

        return generated_plan, valid

    def update_task_plan(self, valid: bool):

        if valid:
            valid_plan = self.generated_plan.split('\n')

            self.task_data.set_generated_plan(plan=valid_plan)
        else:
            self.task_data.set_generated_plan(plan=None)

    def save_new_plan(self):

        valid_plan = self.generated_plan.split('\n')

        new_plan_file_name = f'{self.task_data.task.task_name}_plan.txt'
        assert self.task_data.llm_generated_task is not None
        new_plan_dir = get_llm_gen_plan_dir(
            llm_gen_dir=self.llm_gen_dir,
            symbolic=False,
            llm_gen_prob=self.task_data.llm_generated_task
        )
        new_plan_file_path = new_plan_dir / new_plan_file_name
        with open(new_plan_file_path, 'w') as f:
            for action in valid_plan:
                f.write(f'{action}\n')
            f.write(f'; cost = {len(valid_plan)} (unit cost)\n')


class AgentPlanGenPseudoCode(AgentPlanGen):

    def __init__(self,
                 llm_model: LLMModel,
                 plan_gen_prompt_file: str,
                 typed_hints: bool,
                 log_dir: Path,
                 llm_gen_dir: Path,
                 flags: dict,
                 increase_counter: bool = True,
                 allow_refinement: bool = False):

        super().__init__(llm_model=llm_model,
                         plan_gen_prompt_file=plan_gen_prompt_file,
                         typed_hints=typed_hints,
                         log_dir=log_dir,
                         llm_gen_dir=llm_gen_dir,
                         flags=flags,
                         allow_refinement=allow_refinement,
                         increase_counter=increase_counter)

        self.pseudo_code = None

    def update_pseudo_code(self, pseudo_code: str):
        self.pseudo_code = pseudo_code

    def create_prompt_param(self) -> dict:

        assert self.pseudo_code

        if self.typed_hints:
            action_hint = self.task_data.task.actions_hint_typed
        else:
            action_hint = self.task_data.task.actions_hint

        prompt_param = {'domain_descrip': self.task_data.get_domain_nl(),
                        'domain_strategy': self.pseudo_code,
                        'problem_descrip': self.task_data.get_problem_nl(),
                        'action_hint': action_hint}

        return prompt_param


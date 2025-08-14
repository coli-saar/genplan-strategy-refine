from typing import List, Dict, Tuple, Union
from pathlib import Path
from agents.agent_description_gen_basic import AgentDescriptionGenBasic
from llm_models.llm_base_class import LLMModel
from utils.tasks import Task, TaskData
from utils.utils import create_prompt_template


class AgentDescriptionGen(AgentDescriptionGenBasic):

    def __init__(self,
                 llm_model: LLMModel,
                 domain_prompt_file: str,
                 problem_prompt_file: str,
                 task_dict: Dict[str, TaskData],
                 log_dir: Path,
                 flags: dict,
                 incl_plans: bool = True,
                 imp_aspects_prompt_file: Union[None, str] = None,
                 sum_prompt_file: Union[None, str] = None,
                 increase_counter: bool = True
                 ):

        super().__init__(llm_model=llm_model,
                         domain_prompt_file=domain_prompt_file,
                         task_dict=task_dict,
                         log_dir=log_dir,
                         flags=flags,
                         incl_plans=incl_plans,
                         increase_counter=increase_counter)

        self.problem_prompt_template = create_prompt_template(prompt_file=problem_prompt_file, flags=self.flags)
        self.domain_description_prompt = None

        self.important_aspects_prompt = create_prompt_template(imp_aspects_prompt_file, flags=self.flags) if imp_aspects_prompt_file is not None else None
        self.sum_prompt_file = create_prompt_template(sum_prompt_file, flags=self.flags) if sum_prompt_file is not None else None

    def generate_descriptions(self):

        self.domain_description = self.generate_domain_description()

        for task in self.tasks:
            self.gen_and_update_task_description(task=task)

    def generate_descriptions_additional_tasks(self, tasks: List[Task]):
        """
        Generates the problem descriptions for the provided tasks and adds the new tasks
        to the task_dict
        :param tasks: List of Task objects
        :return:
        """
        self.tasks.extend(tasks)

        for task in tasks:
            assert task.task_name not in self.task_dict.keys()
            self.task_dict[task.task_name] = TaskData(task)
            self.gen_and_update_task_description(task=task)

    def gen_and_update_task_description(self, task: Task):

        if not self.include_plans:
            actually_incl_plans = False
        else:
            actually_incl_plans = True if self.task_dict[task.task_name].get_plan() is not None else False

        output_descript = self.generate_problem_description(task=task, include_plan=actually_incl_plans)

        problem_description, plan_description = self.parse_problem_descript_output(
            model_output=output_descript,
            incl_plan=actually_incl_plans
        )
        if plan_description is not None:
            optimal = False if self.task_dict[task.task_name].generated_plan else True
            self.task_dict[task.task_name].set_nl_plan_description(plan_description=plan_description,
                                                                   optimal=optimal)

        self.task_dict[task.task_name].set_nl_domain_description(self.domain_description)
        self.task_dict[task.task_name].set_nl_problem_description(problem_description)

    def parse_domain_descript_output(self, model_response: str) -> str:

        parts = model_response.split('"""')
        if len(parts) == 3 and 'Description of the game:' in parts[1]:
            domain_description = parts[1].replace('Description of the game:', '')
            domain_description = domain_description.strip()
            return domain_description
        elif len(parts) == 3 and 'Summary' in parts[1]:
            domain_description = parts[1].replace('Summary:', '')
            domain_description = domain_description.strip()
            return domain_description
        elif len(parts) == 1:
            parts = model_response.split('```')
            if len(parts) == 3 and 'Description of the game:' in parts[1]:
                domain_description = parts[1].replace('Description of the game:', '')
                domain_description = domain_description.strip()
                return domain_description

        return model_response

    def parse_problem_descript_output(self,
                                      model_output: str,
                                      incl_plan) -> Tuple[str, str]:

        problem_description = None
        plan_description = None

        assert 'Problem description:' in model_output
        if incl_plan:
            assert 'Plan description:' in model_output

        if model_output.count('"""') == 4 or model_output.count('"""') == 2:
            parts = model_output.split('"""')
        elif model_output.count('```') == 4 or model_output.count('```') == 2:
            parts = model_output.split('```')
        else:
            parts = []
        for p in parts:
            if 'Problem description:' in p and 'Plan description:' not in p:
                problem_description = p.replace('Problem description:', '').strip()
            if 'Plan description:' in p and 'Problem description:' not in p:
                plan_description = p.replace('Plan description:', '').strip()

        if problem_description is not None:
            if incl_plan:
                assert plan_description is not None
            return problem_description, plan_description

        else:
            parts = model_output.split('Plan description:')
            problem_description = parts[0].replace('Problem description:', '').strip()
            if incl_plan:
                plan_description = parts[1].strip()
            return problem_description, plan_description

    def generate_domain_description(self, include_plans=False) -> str:
        """

        :param include_plans: has no effect here; only needed because overwriting
        :return:
        """

        domain_str = self.tasks[0].domain_str
        prompt_param = {'domain_str': domain_str}
        prompt = self.domain_prompt_template.render(**prompt_param)
        self.domain_description_prompt = prompt

        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output()

        if self.important_aspects_prompt is not None:
            assert self.sum_prompt_file is not None
            prompt = self.important_aspects_prompt.render()
            response, _ = self.llm_model.generate(user_message=prompt)
            self.log_llm_input_output()
            prompt = self.sum_prompt_file.render()
            response, _ = self.llm_model.generate(user_message=prompt)
            self.log_llm_input_output()

        response = self.parse_domain_descript_output(model_response=response)

        return response

    def generate_problem_description(self, task: Task, include_plan: bool) -> str:
        # Take only system prompt + potentially few-shot examples and add the generated domain description
        initial_history = self.llm_model.get_initial_history()
        new_history = initial_history.copy()
        new_history.extend([{'role': self.llm_model.role_user,
                             'content': self.domain_description_prompt},
                            {'role': self.llm_model.role_assistant,
                             'content': self.domain_description}])

        self.llm_model.update_history(new_history=new_history)

        problem_str = task.problem_str

        if not include_plan:
            plan_str = False
        else:
            assert self.task_dict[task.task_name].get_plan() is not None
            plan_str = self.task_dict[task.task_name].get_plan_str()

        prompt_param = {'problem_str': problem_str, 'plan_str': plan_str}
        prompt = self.problem_prompt_template.render(**prompt_param)

        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output()

        return response


def create_copy_for_new_tasks(orig_agent: AgentDescriptionGen,
                              new_agent: AgentDescriptionGen):

    new_agent.llm_model.update_history(new_history=orig_agent.llm_model.get_history())

    new_agent.task_dict = dict()
    new_agent.tasks = new_agent.get_tasks()

    new_agent.domain_description = orig_agent.domain_description
    new_agent.domain_description_prompt = orig_agent.domain_description_prompt

    return new_agent


from typing import List, Dict, Tuple
from pathlib import Path
from llm_models.llm_base_class import LLMModel
from utils.tasks import Task, TaskData
from utils.utils import create_prompt_template, log_llm_interaction, log_agent_system_prompt
from utils.utils import log_llm_interaction_validation_counter


"""
Version similar to Silver et al.
"""


class AgentDescriptionGenBasic:

    def __init__(self,
                 llm_model: LLMModel,
                 domain_prompt_file: str,
                 task_dict: Dict[str, TaskData],
                 log_dir: Path,
                 flags: dict,
                 incl_plans: bool = False,
                 increase_counter: bool = True
                 ):

        self.agent_name = 'DescriptionGen'
        self.flags = flags

        self.increase_counter = increase_counter

        self.llm_model = llm_model
        self.task_dict = task_dict
        self.tasks = self.get_tasks()
        self.domain_description = None
        self.domain_prompt_template = create_prompt_template(prompt_file=domain_prompt_file, flags=self.flags)

        self.include_plans = incl_plans
        self.all_plans_available = incl_plans

        self.problems_enum = dict()

        # Logging set-up
        self.log_dir = log_dir
        log_agent_system_prompt(log_dir=self.log_dir,
                                llm_model=self.llm_model,
                                agent_name=self.agent_name)

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

    def check_for_all_plans(self):
        """
        Check whether there is a plan for all tasks
        :return:
        """
        all_plans_available = True
        for task_data in self.task_dict.values():

            if task_data.get_plan() is None:
                all_plans_available = False
        return all_plans_available

    def get_tasks(self) -> List[Task]:
        tasks = []
        for task_data in self.task_dict.values():
            tasks.append(task_data.task)
        return tasks

    def generate_descriptions(self):

        if not self.include_plans:
            actually_incl_plans = False
        else:
            actually_incl_plans = self.check_for_all_plans()

        model_response = self.generate_domain_description(include_plans=actually_incl_plans)
        self.domain_description, raw_domain_description = self.parse_domain_descript_output(model_response=model_response)

        # remove domain description from model output for extracting the problem descriptions
        cleaned_model_output = model_response.replace(raw_domain_description, '')
        problem_descriptions = self.parse_problem_descript_output(model_output=cleaned_model_output,
                                                                  incl_plan=actually_incl_plans)

        for t_enum, (prob, plan) in enumerate(problem_descriptions):
            task_name = self.problems_enum[t_enum + 1]
            self.task_dict[task_name].set_nl_domain_description(self.domain_description)
            self.task_dict[task_name].set_nl_problem_description(prob)
            if plan is not None:
                optimal = False if self.task_dict[task_name].generated_plan else True
                self.task_dict[task_name].set_nl_plan_description(plan_description=plan,
                                                                  optimal=optimal)

    def generate_domain_description(self, include_plans) -> str:
        domain_str = self.tasks[0].domain_str
        problem_strs = []
        for t_enum, task in enumerate(self.tasks):
            problem_str = f'Problem instance {t_enum + 1}:\n{task.problem_str}'

            if include_plans and self.task_dict[task.task_name].get_plan() is not None:
                plan_str = f'\n\nPlan {t_enum + 1}:\n{self.task_dict[task.task_name].get_plan_str()}'
                problem_str = problem_str + plan_str

            self.problems_enum[t_enum + 1] = task.task_name

            problem_strs.append(problem_str)

        prompt_param = {'domain_str': domain_str,
                        'problems': problem_strs,
                        'incl_plans': include_plans}

        prompt = self.domain_prompt_template.render(**prompt_param)
        self.domain_description_prompt = prompt

        response, _ = self.llm_model.generate(user_message=prompt)
        self.log_llm_input_output()

        return response

    def parse_domain_descript_output(self, model_response: str) -> Tuple[str, str]:

        assert 'Description of the game:' in model_response

        descript_started = False
        delim = None
        domain_description = ''
        raw_domain_description = ''
        response_lines = model_response.strip().split('\n')
        for line in response_lines:
            raw_domain_description += f'{line}\n'
            if '"""' in line:
                if delim is None:
                    delim = '"""'
                else:
                    assert descript_started and delim == '"""'
                    break

            elif '```' in line:
                if delim is None:
                    delim = '```'
                else:
                    assert descript_started and delim == '```'
                    break

            elif 'Description of the game:' in line:
                descript_started = True

            elif 'Problem description ' in line:
                raise ValueError

            elif descript_started:
                domain_description += f'{line}\n'

        assert domain_description
        return domain_description, raw_domain_description

    def parse_problem_descript_output(self,
                                      model_output: str,
                                      incl_plan) -> List[Tuple[str, str]]:
        descriptions = dict()

        for t_enum in self.problems_enum.keys():
            assert f'Problem description {t_enum}' in model_output
            if incl_plan:
                assert f'Plan description {t_enum}' in model_output

        prob_descript_started = False
        plan_descript_started = False
        delim = None
        problem_description = []
        plan_description = []
        current_enum = -1
        response_lines = model_output.strip().split('\n')

        for line in response_lines:
            if '"""' in line:
                if delim is None:
                    delim = '"""'
                else:
                    assert prob_descript_started and delim == '"""'
                    if incl_plan:
                        assert plan_descript_started
                        descriptions[current_enum] = ('\n'.join(problem_description), '\n'.join(plan_description))
                    else:
                        descriptions[current_enum] = ('\n'.join(problem_description), None)
                    prob_descript_started = False
                    plan_descript_started = False
                    delim = None
                    problem_description = []
                    plan_description = []

            elif '```' in line:
                if delim is None:
                    delim = '```'
                else:
                    assert prob_descript_started and delim == '```'
                    if incl_plan:
                        assert plan_descript_started
                        descriptions[current_enum] = ('\n'.join(problem_description), '\n'.join(plan_description))
                    else:
                        descriptions[current_enum] = ('\n'.join(problem_description), None)
                    prob_descript_started = False
                    plan_descript_started = False
                    delim = None
                    problem_description = []
                    plan_description = []

            elif 'Problem description' in line:
                extracted_num = line.strip().replace('Problem description ', '')
                extracted_num = extracted_num.split(':')[0]
                current_enum = int(extracted_num)
                prob_descript_started = True

            elif 'Plan description' in line:
                assert prob_descript_started
                extracted_num = line.strip().replace('Plan description ', '')
                extracted_num = extracted_num.split(':')[0]
                extracted_num = int(extracted_num)
                assert extracted_num == current_enum
                prob_descript_started = False
                plan_descript_started = True

            elif prob_descript_started:
                problem_description.append(line)

            elif plan_descript_started:
                plan_description.append(line)

        assert problem_description == [] and plan_description == []

        descriptions_enum = list(descriptions.items())
        descriptions_enum.sort()
        descriptions = [tup[1] for tup in descriptions_enum]

        return descriptions


    def generate_descriptions_additional_tasks(self, tasks: List[Task]):
        """
        Generates the problem descriptions for the provided tasks and adds the new tasks
        to the task_dict
        :param tasks: List of Task objects
        :return:
        """
        self.tasks.extend(tasks)
        for task in tasks:

            if not self.include_plans:
                actually_incl_plans = False
            else:
                actually_incl_plans = True if self.task_dict[task.task_name].get_plan() is not None else False

            assert task.task_name not in self.task_dict.keys()
            self.task_dict[task.task_name] = TaskData(task)
            domain_str = self.tasks[0].domain_str
            problem_strs = []
            problem_str = f'Problem instance 1:\n{task.problem_str}'

            if actually_incl_plans:
                plan_str = f'\n\nPlan 1:\n{self.task_dict[task.task_name].get_plan_str()}'
                problem_str = problem_str + plan_str

            self.problems_enum[1] = task.task_name

            problem_strs.append(problem_str)
            prompt_param = {'domain_str': domain_str,
                            'problems': problem_strs,
                            'incl_plans': actually_incl_plans}

            prompt = self.domain_prompt_template.render(**prompt_param)
            self.domain_description_prompt = prompt

            response, _ = self.llm_model.generate(user_message=prompt)
            self.log_llm_input_output()

            # parse the problem description

            domain_description, raw_domain_description = self.parse_domain_descript_output(
                model_response=response)

            # remove domain description from model output
            cleaned_model_output = response.replace(raw_domain_description, '')
            problem_descriptions = self.parse_problem_descript_output(
                model_output=cleaned_model_output,
                incl_plan=actually_incl_plans
            )
            assert len(problem_descriptions) == 1
            task_name = task.task_name
            self.task_dict[task_name].set_nl_problem_description(problem_descriptions[0][0])

            self.task_dict[task_name].set_nl_domain_description(self.domain_description)

            if actually_incl_plans:
                optimal = False if self.task_dict[task_name].generated_plan else True
                self.task_dict[task_name].set_nl_plan_description(
                    plan_description=problem_descriptions[0][1],
                    optimal=optimal
                )


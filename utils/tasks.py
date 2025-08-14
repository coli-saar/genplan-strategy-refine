import os
from functools import cached_property
from collections import defaultdict, OrderedDict
from typing import Union, List, Set, Tuple, Any, DefaultDict, Dict
from pathlib import Path
from tarski.io import PDDLReader
from pddlgym.parser import Literal, PDDLDomainParser, PDDLProblemParser
from utils.pddl_processing.pddl_parsing import parse_ordered_predicates, parse_actions


class Task:

    def __init__(self,
                 domain_file_path: str,
                 problem_file_path: str):
        #print(problem_file_path)
        self.domain_file_path = Path(domain_file_path)
        self.problem_file_path = Path(problem_file_path)

        self.domain_str: str = self.read_in_domain()
        self.problem_str: str = self.read_in_problem()
        self.problem_file_name = os.path.split(problem_file_path)[-1]

        self.task_name = self.problem_file_name.replace('.pddl', '')

        self.domain, self.problem = self.create_problem_and_domain()
        self.pddl_reader_tarski = PDDLReader(raise_on_error=True)
        self.pddl_reader_tarski.parse_domain(domain_file_path)
        self.actions: Dict[str, OrderedDict] = parse_actions(pddl_reader=self.pddl_reader_tarski)
        self.predicates: Dict[str, OrderedDict] = parse_ordered_predicates(domain_file=self.domain_file_path, pddl_reader=self.pddl_reader_tarski)

        # optimal plan
        self.optimal_plan = None

        # whether the task is solvable
        # 0 = False, 1 = True, 2 = Not known yet
        self.solvable = 2

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        other_objs = other.objects
        other_init = other.init
        other_goal = other.goal

        if other_goal != self.goal:
            return False
        if other_init != self.init:
            return False
        if other_objs != self.objects:
            return False
        else:
            return True

    def set_optimal_plan(self, plan: List[str]):
        plan = [action for action in plan if not action.startswith(';')]
        self.optimal_plan = plan
        self.solvable = 1

    def get_optimal_plan(self) -> List[str]:
        return self.optimal_plan

    def create_problem_and_domain(self) -> Tuple[PDDLDomainParser, PDDLProblemParser]:

        domain = PDDLDomainParser(
            self.domain_file_path,
            expect_action_preds=False,
            operators_as_actions=True
        )

        problem = PDDLProblemParser(
            self.problem_file_path,
            domain.domain_name,
            domain.types,
            domain.predicates,
            domain.actions,
            domain.constants,
        )

        return domain, problem

    def read_in_domain(self) -> str:
        with open(self.domain_file_path, 'r', encoding='utf-8') as df:
            domain_str = df.read().lower()
        domain_str = self.remove_comments_pddl(pddl_str=domain_str)
        domain_str = domain_str.strip()
        return domain_str

    # TODO: Think about standardizing the line breaks etc.
    def read_in_problem(self) -> str:
        with open(self.problem_file_path, 'r', encoding='utf-8') as pf:
            problem_str = pf.read().lower()
        problem_str = self.remove_comments_pddl(pddl_str=problem_str)
        problem_str = problem_str.strip()
        return problem_str

    def remove_comments_pddl(self, pddl_str: str) -> str:
        """

        :param pddl_str:
        :return:
        """
        lines = pddl_str.split('\n')
        cleaned_lines = []

        for l in lines:
            if ';' not in l:
                cleaned_lines.append(l)
            else:
                comment_start = l.index(';')
                cleaned_l = l[:comment_start]
                cleaned_lines.append(cleaned_l)

        cleaned_pddl_str = '\n'.join(cleaned_lines)

        return cleaned_pddl_str

    def get_file_paths(self):
        return self.domain_file_path, self.problem_file_path

    @cached_property
    def typed(self) -> bool:
        """Whether the domain is typed."""
        return self.domain.uses_typing

    @property
    def objects(self) -> Union[Set[Tuple[str, str]], Set[str]]:
        """The objects (not including constants) and their types."""
        if not self.typed:
            return {o.name for o in self.problem.objects}
        return {(o.name, str(o.var_type)) for o in self.problem.objects}

    @property
    def init(self) -> Set[Tuple[str, ...]]:
        """The initial atoms in string form."""
        return {_literal_to_tuple(l) for l in self.problem.initial_state}

    @property
    def goal(self) -> Set[Tuple[str, ...]]:
        """The goal in string form."""
        if isinstance(self.problem.goal, Literal):
            return {_literal_to_tuple(self.problem.goal)}
        else:
            return {_literal_to_tuple(l) for l in self.problem.goal.literals}

    @cached_property
    def size(self) -> int:
        """Crude estimate of task size."""
        return len(self.objects) + len(self.init) + len(self.goal)

    @cached_property
    def problem_name(self) -> str:
        """The problem name."""
        key = "(problem "
        s = self.problem_str.index(key)
        e = self.problem_str[s:].index(")")
        return self.problem_str[s + len(key) : s + e]

    @cached_property
    def domain_name(self) -> str:
        """The domain name."""
        key = "(domain "
        s = self.domain_str.index(key)
        e = self.domain_str[s:].index(")")
        return self.domain_str[s + len(key) : s + e]

    # TODO: if abbreviated, the problem_str should at least contain those objects and predicates for which the error occurs
    def get_abbreviated_problem_str(
        self,
        max_objects_per_type: int = 10,
        max_init_atoms_per_type: int = 10,
        max_goal_atoms_per_type: int = 10,
    ) -> str:
        """A shortened version of the problem string."""
        # Build objects.
        object_type_to_strs: DefaultDict[str, List[str]] = defaultdict(list)
        for obj in sorted(self.objects):
            if self.typed:
                assert isinstance(obj, tuple)
                obj_name, obj_type = obj
                obj_str = f"{obj_name} - {obj_type}"
            else:
                assert isinstance(obj, str)
                obj_type = "default"
                obj_str = obj
            object_type_to_strs[obj_type].append(obj_str)
        # Abbreviate.
        object_strs: List[str] = []
        for obj_type in sorted(object_type_to_strs):
            type_obj_strs = object_type_to_strs[obj_type]
            if len(type_obj_strs) > max_objects_per_type:
                type_obj_strs = type_obj_strs[:max_objects_per_type]
                type_obj_strs.append("...")
            object_strs.extend(type_obj_strs)
        objects_str = "\n    ".join(object_strs)
        # Build init.
        pred_to_init_strs: DefaultDict[str, List[str]] = defaultdict(list)
        for atom_tuple in sorted(self.init):
            pred = atom_tuple[0]
            atom_str = "(" + " ".join(atom_tuple) + ")"
            pred_to_init_strs[pred].append(atom_str)
        # Abbreviate.
        init_strs: List[str] = []
        for pred in sorted(pred_to_init_strs):
            pred_strs = pred_to_init_strs[pred]
            if len(pred_strs) > max_init_atoms_per_type:
                pred_strs = pred_strs[:max_init_atoms_per_type]
                pred_strs.append("...")
            init_strs.extend(pred_strs)
        init_str = "\n    ".join(init_strs)
        # Build goal.
        pred_to_goal_strs: DefaultDict[str, List[str]] = defaultdict(list)
        for atom_tuple in sorted(self.goal):
            pred = atom_tuple[0]
            atom_str = "(" + " ".join(atom_tuple) + ")"
            pred_to_goal_strs[pred].append(atom_str)
        # Abbreviate.
        goal_strs: List[str] = []
        for pred in sorted(pred_to_goal_strs):
            pred_strs = pred_to_goal_strs[pred]
            if len(pred_strs) > max_goal_atoms_per_type:
                pred_strs = pred_strs[:max_goal_atoms_per_type]
                pred_strs.append("...")
            goal_strs.extend(pred_strs)
        goal_str = "\n    ".join(goal_strs)

        return f"""(define (problem {self.problem_name} (:domain {self.domain_name})
  (:objects
    {objects_str}
  )
  (:init
    {init_str}
  )
  (:goal (and
    {goal_str}
    )
  )    
)
        """

    def action_has_valid_syntax(self, action: Any) -> bool:
        """Check if the action name and arity is correct and objects exist."""
        if not isinstance(action, str):
            return False
        if not (action.startswith("(") and action.endswith(")")):
            return False
        action = action[1:-1].strip()
        if " " not in action:
            name = action
            arg_names = []
        else:
            name, remainder = action.split(" ", 1)
            arg_names = remainder.split(" ")
        if name not in self.domain.operators:
            return False
        if len(arg_names) != len(self.domain.operators[name].params):
            return False
        return True

    @cached_property
    def actions_hint(self) -> str:
        """Write the action signatures."""
        action_str = self.get_hint(typed=False, operator_dict=self.actions)

        return action_str

    @cached_property
    def actions_hint_typed(self) -> str:
        action_str = self.get_hint(typed=True, operator_dict=self.actions)
        return action_str

    @cached_property
    def predicates_hint(self) -> str:
        """
        List of all available predicates
        :return:
        """
        pred_strs = self.get_hint(typed=False, operator_dict=self.predicates)
        return pred_strs

    @cached_property
    def predicates_hint_typed(self) -> str:
        pred_strs = self.get_hint(typed=True, operator_dict=self.predicates)
        return pred_strs

    def get_hint(self, typed: bool, operator_dict: Dict[str, OrderedDict]) -> str:

        op_strs: List[str] = []
        for op_name, params in operator_dict.items():
            param_str = " ".join(param_name for param_name in params.keys())
            act_str = f"({op_name.lower()} {param_str.lower()})"
            if typed:
                type_hint = [f'{obj.lower()} is of type {t.lower()}' for obj, t in params.items()]
                type_hint_str = ' '.join(type_hint)
                act_str += f'\twhere {type_hint_str}'
            op_strs.append(act_str)

        formatted = "\n- ".join(op_strs)
        formatted = f'- {formatted}'

        return formatted


def _literal_to_tuple(lit: Literal) -> Tuple[str, ...]:
    arg_strs = [v.name for v in lit.variables]
    return (lit.predicate.name,) + tuple(arg_strs)


class TaskData:

    def __init__(self,
                 task: Task,
                 llm_generated: bool = False):

        self.task = task
        self.task_name = self.task.task_name
        self.llm_generated_task = llm_generated

        # valid plan generated during the strategy generation phase
        self.generated_plan: Union[List[str], None] = None

        # NL descriptions
        self.nl_domain = None
        self.nl_problem = None
        self.nl_plan_optimal = None
        self.nl_plan_gen = None

        # Task specific strategy
        self.task_strategy = None

        # Summarized, i.e. domain specific strategy
        self.sum_strategy = None

        # whether task is solvable or not
        # 0 = False, 1 = True, 2 = not known
        self.solvable = task.solvable

    def set_generated_plan(self, plan: Union[List[str], None]):
        self.generated_plan = plan
        if plan:
            self.solvable = 1

    def set_nl_domain_description(self, domain_description: str):
        self.nl_domain = domain_description

    def set_nl_problem_description(self, problem_description: str):
        self.nl_problem = problem_description

    def set_nl_plan_description(self, plan_description: str, optimal: bool):
        if optimal:
            self.nl_plan_optimal = plan_description
        else:
            self.nl_plan_gen = plan_description

    def set_strategy_task(self, strategy: str):
        self.task_strategy = strategy

    def get_strategy_task(self):
        return self.task_strategy

    def set_strategy_domain(self, strategy: str):
        self.sum_strategy = strategy

    def get_strategy_domain(self):
        return self.sum_strategy

    def get_problem_str(self):
        return self.task.problem_str

    def get_problem_nl(self):
        return self.nl_problem

    def get_domain_str(self):
        return self.task.domain_str

    def get_domain_nl(self):
        return self.nl_domain

    def get_plan(self) -> List[str]:
        if self.generated_plan:
            return self.generated_plan
        else:
            return self.task.optimal_plan

    # TODO: think about this
    def get_plan_str(self) -> str:
        plan = self.get_plan()
        plan_str = ', '.join(plan)
        if plan_str == '':
            plan_str = 'Goal is already satisfied.'
        return plan_str

    def get_plan_nl(self):
        if self.nl_plan_gen:
            return self.nl_plan_gen
        else:
            return self.nl_plan_optimal


def update_domain_strategy(task_dict: Dict[str, TaskData],
                           summarized_strategy: str):

    for task_data in task_dict.values():
        task_data.set_strategy_domain(strategy=summarized_strategy)


def create_tasks_dict(tasks: List[Task]) -> Dict[str, TaskData]:
    tasks_dict = dict()
    for task in tasks:
        tasks_dict[task.task_name] = TaskData(task=task)
    return tasks_dict


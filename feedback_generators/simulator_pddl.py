import json
import os
import random
import re
import time
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Union, OrderedDict
from collections import defaultdict
from tarski.io import PDDLReader
from tarski.syntax import Atom, CompoundFormula, Predicate
from tarski.fstrips import Action
import atexit
from datetime import datetime
from utils.paths import TEMP_DIR


"""
Code from Stein et al. 2025, https://github.com/minecraft-saar/autoplanbench
Slightly adapted for more efficient processing
"""


def get_domain_name(domain_path):
    domain_name = None
    with open(domain_path, 'r') as f:
        for line in f.readlines():
            if ':requirements' in line:
                assert domain_name is not None
                break
            if '(domain ' in line:
                parts = line.split('(domain ')[-1]
                assert ')' in parts
                domain_name = parts.split(')')[0]
    assert domain_name is not None
    return domain_name


class RawPDDLEnvironment:

    def __init__(self, instance_file: str, domain_file: str):

        self.domain_file = domain_file
        self.instance_file = instance_file

        Path(TEMP_DIR).mkdir(exist_ok=True)

        domain_type_name = os.path.split(domain_file)[0]
        domain_type_name = os.path.split(domain_type_name)[-1]

        time_stamp = str(datetime.now())
        time_stamp = [char for char in time_stamp if char != ' ']
        time_stamp = ''.join(time_stamp)
        while True:
            self.tmp_file_int = random.randint(0, 1000000)
            self.tmp_file_int2 = random.randint(0, 1000000)
            tmp_instance_file = os.path.join(TEMP_DIR, f'tmp_instance_{domain_type_name}_{time_stamp}_{self.tmp_file_int}_{self.tmp_file_int2}.pddl')
            if not os.path.exists(tmp_instance_file):
                break

        temp_suffix = f'{domain_type_name}_{time_stamp}_{self.tmp_file_int}_{self.tmp_file_int2}'

        self.tmp_instance_file = tmp_instance_file
        self.tmp_action_file = os.path.join(TEMP_DIR, f'tmp_action_{temp_suffix}')

        domain_file_name = os.path.split(domain_file)[-1]
        instance_file_name = os.path.split(instance_file)[-1]
        self.lowercase_domain_file = os.path.join(TEMP_DIR, f'tmp_{temp_suffix}_{domain_file_name}')
        self.lower_case_instance_file = os.path.join(TEMP_DIR, f'tmp_{temp_suffix}_{instance_file_name}')
        self.problem = self.create_lowercase_problem()

        self.actions_pddl: dict = self.get_problem_actions()
        self.possible_objects = self.get_problem_objects()
        self.objects2tpyes, self.types2objects = self.get_object_types()
        self.static_preds = self.get_static_preds()

        self.facts_initial_state = [self.convert_pre2in(initial) for initial in list(self.problem.init.as_atoms())]
        self.facts_current_state = self.facts_initial_state.copy()
        # keys: 'pos_conditions', 'neg_conditions'
        self.conditions_goal_state: dict = self.process_goal_conditions()

        self.completed = False
        self.goal_feedback = ''
        self.last_val_response = ''

        self.problem_descript, self.goal_descript = self.split_problem_file(instance_file=instance_file)

        # Count how often step was already called
        self.current_step_id = 0

        atexit.register(self.remove_temp_files)

     # -------------------------- Setting things up -----------------------------

    def split_problem_file(self, instance_file):

        with open(instance_file, 'r') as pf:
            problem_text = pf.read()
        problem_text = problem_text.strip()
        pref, definition = problem_text.split('(:objects')
        problem_def, goal_def = definition.split('(:goal')

        problem_def = f'(:objects {problem_def}'
        problem_def = problem_def.strip()

        goal_def = f'(:goal {goal_def}'
        goal_def = goal_def.strip()
        if goal_def[-1] == ')':
            goal_def = goal_def[:-1]

        return problem_def, goal_def


    def create_lowercase_problem(self):
        with open(self.lowercase_domain_file, 'w') as new:
            with open(self.domain_file, 'r') as orig:
                orig_text = orig.read()
                new_text = orig_text.lower()
                new.write(new_text)

        with open(self.lower_case_instance_file, 'w') as new_inst:
            with open(self.instance_file, 'r') as orig_inst:
                orig_text = orig_inst.read()
                new_text = orig_text.lower()
                new_inst.write(new_text)

        problem = self.get_problem(self.lower_case_instance_file, self.lowercase_domain_file)
        return problem

    def remove_temp_files(self):
        if os.path.exists(self.tmp_instance_file):
            os.remove(self.tmp_instance_file)
        if os.path.exists(self.tmp_action_file):
            os.remove(self.tmp_action_file)
        if os.path.exists(self.lowercase_domain_file):
            os.remove(self.lowercase_domain_file)
        if os.path.exists(self.lower_case_instance_file):
            os.remove(self.lower_case_instance_file)

    def process_goal_conditions(self) -> Dict[str, list]:

        pos_goal_conditions = []
        neg_goal_conditions = []

        if isinstance(self.problem.goal, CompoundFormula):
            operator = self.problem.goal.connective
            if operator.name == 'And':
                for sub in self.problem.goal.subformulas:
                    if isinstance(sub, Atom):
                        pos_goal_conditions.append(self.convert_pre2in(sub))
                    elif sub.connective.name == 'Not':
                        assert len(sub.subformulas) == 1
                        pred_str = self.convert_pre2in(sub.subformulas[0])
                        neg_goal_conditions.append(pred_str)
            elif operator.name == 'Not':
                assert len(self.problem.goal.subformulas) == 1
                pred_str = self.convert_pre2in(self.problem.goal.subformulas[0])
                neg_goal_conditions.append(pred_str)
            else:
                raise ValueError

        elif isinstance(self.problem.goal, Atom):
            pos_goal_conditions.append(self.convert_pre2in(self.problem.goal))

        else:
            raise ValueError

        goal_conditions = {'pos_conditions': pos_goal_conditions,
                           'neg_conditions': neg_goal_conditions}

        return goal_conditions


    def convert_pre2in(self, action: Union[str, Atom, CompoundFormula]):
        """
        Converts actions and predicates in the format that the PDDLReader outputs into the format that VAL
        expects
        i.e. clear(b) -> (clear b), pick-up(b) -> (pick-up b), stack(b, c) -> (stack b c)
        :param action:
        :return:
        """
        action = str(action)
        action_name, action_args = action.split('(')
        new_action_str = f'({action_name} {action_args}'
        new_action_str = new_action_str.replace(',', ' ')
        new_action_str = new_action_str.replace(' )', ')')
        return new_action_str


    def convert_in2pre(self, action: str):
        """
        Converts actions and predicates in the format that VAL into the format that the PDDLReader outputs
        expects
        i.e. (clear b) -> clear(b); (pick-up b) -> pick-up(b), (stack b c) -> stack(b, c)
        :param action:
        :return:
        """
        action = action.replace('(', '').replace(')', '')
        components = action.split(' ')
        new_action_str = f'{components[0]}('
        for arg in components[1:]:
            new_action_str += f'{arg}, '
        new_action_str = new_action_str[:-2]
        new_action_str += ')'

        return new_action_str


    def create_tmp_instance(self):
        """

        :return:
        """
        lines_to_keep1 = []
        lines_to_keep2 = []
        with open(self.instance_file, 'r') as orig_file:
            before_init_state = True
            after_init_state = False
            for line in orig_file:
                stripped_line = line.strip()
                if stripped_line.startswith('(:init'):
                    before_init_state = False
                elif stripped_line.startswith('(:goal'):
                    after_init_state = True
                if before_init_state:
                    lines_to_keep1.append(line)
                elif after_init_state:
                    lines_to_keep2.append(line)
        with open(self.tmp_instance_file, 'w') as tmp_file:
            for line in lines_to_keep1:
                tmp_file.write(line)
            tmp_file.write('(:init\n')
            for current_fact in self.facts_current_state:
                tmp_file.write(f'{current_fact}\n')
            tmp_file.write(')\n')
            for line in lines_to_keep2:
                tmp_file.write(line)

    def check_completion(self) -> Tuple[str, bool, bool, list]:
        # TODO:
        positive_goal_conditions = self.conditions_goal_state['pos_conditions']
        negative_goal_conditions = self.conditions_goal_state['neg_conditions']

        should_be_false = []
        should_be_true = []

        reached_goal = True
        for pos_cond in positive_goal_conditions:
            if not pos_cond in self.facts_current_state:
                should_be_true.append(pos_cond)
                reached_goal = False

        for neg_cond in negative_goal_conditions:
            if neg_cond in self.facts_current_state:
                should_be_false.append(neg_cond)
                reached_goal = False

        if reached_goal:
            return '', True, reached_goal, []

        feedback = f'The generated plan does not reach the goal.\n'

        if len(should_be_false) > 0:
            feedback += f'The following needs to be false but is true after executing all actions: '
        for pred in should_be_false:
            feedback += f'{self.get_description_pred(pred)}, '
        if feedback[-2:] == ', ':
            feedback = feedback[:-2]
            feedback += '\n'

        if len(should_be_true) > 0:
            feedback += f'The following needs to be true but is false after executing all actions: '
        for pred in should_be_true:
            feedback += f'{self.get_description_pred(pred)}, '
        if feedback[-2:] == ', ':
            feedback = feedback[:-2]
            feedback += '\n'

        if not reached_goal:
            self.goal_feedback = feedback

        return feedback, True, reached_goal, []


    # -------------------------------- Validate Actions ------------------------
    # TODO
    def validate_complete_plan(self, plan_str: str) -> Tuple[bool, bool]:
        plan_str = plan_str.lower()

        plan = plan_str.split('\n')

        # ----- Functionalities to deal with inputs VAL cannot deal with ------
        for action_instr in plan:
            checked_action, mistakes = self.pre_check_action(action_instr=action_instr)
            if checked_action != action_instr:
                return False, self.completed

            if len(self.problem.language.sorts) > 1:
                type_checked_action = self.check_type_constraints(action_instr=action_instr)
                if type_checked_action != action_instr:
                    return False, self.completed

        # need to write it into a temporary file as a unary plan
        with open(self.tmp_action_file, 'w') as plan_file:
            plan_file.write(plan_str)

        # need an instance file that hast the current state as the initial state
        self.create_tmp_instance()

        # need run VAL validate -v self.domain_file self.instance_file plan
        val = os.environ.get('VAL')

        cmd = f'{val}/validate -v {self.domain_file} {self.tmp_instance_file} {self.tmp_action_file}'

        self.last_val_response = os.popen(cmd).read()

        reached_goal, executable, _, advice_goal, _ = self.parse_val_output(self.last_val_response)

        if not reached_goal and executable:
            self.goal_feedback = self.get_feedback_unsat(advice_goal)

        return executable, reached_goal


    def step(self, action_instr: str) -> Tuple[str, bool, bool, list]:
        """

        :param action_instr:
        :return: (feedback_message, executable, goal_reached, error_type)
        """
        self.current_step_id += 1
        action_instr = action_instr.lower()

        # ----- Functionalities to deal with inputs VAL cannot deal with for this domain ------
        checked_action, mistakes = self.pre_check_action(action_instr=action_instr)

        if checked_action != action_instr:
            return checked_action, False, self.completed, mistakes

        if len(self.problem.language.sorts) > 1:
            type_checked_action = self.check_type_constraints(action_instr=action_instr)
            if type_checked_action != action_instr:
                return type_checked_action, False, self.completed, ['type-mistake']

        # ----- Everything that can be done with VAL ----- #

        # need to write it into a temporary file as a unary plan
        with open(self.tmp_action_file, 'w') as plan_file:
            plan_file.write(action_instr)

        # need an instance file that hast the current state as the initial state
        self.create_tmp_instance()

        # need run VAL validate -v self.domain_file self.instance_file plan
        val = os.environ.get('VAL')

        cmd = f'{val}/validate -v {self.domain_file} {self.tmp_instance_file} {self.tmp_action_file}'
        self.last_val_response = os.popen(cmd).read()

        #cmd_tokens = [f'{val}/validate', '-v', self.domain_file, self.tmp_instance_file, self.tmp_action_file, '>', '/scratch/kstein/test_output.txt']
        #try:
        #    run_result = subprocess.run(args=cmd_tokens, check=True, capture_output=True)
        #    console_output = str(run_result.stdout)
        #except subprocess.CalledProcessError as e:
        #    output = e.output
        #    print(output)

        # store output somehow and parse it
        reached_goal, executable, effects, advice_goal, advice_precond = self.parse_val_output(self.last_val_response)

        mistakes = []
        if reached_goal:
            self.completed = True

        if executable:
            if not len(effects) > 0:
                print(self.last_val_response)
            assert len(effects) > 0
            self.update_current_state(effects)
            feedback = self.get_feedback_successful(action_instr)

        else:
            if not len(advice_precond) > 0:
                print(self.last_val_response)
            assert len(advice_precond) > 0
            feedback = self.get_feedback_unsat(advice_precond)
            mistakes.append('unsat-preconditions')

        if not reached_goal and executable:
            self.goal_feedback = self.get_feedback_unsat(advice_goal)

        return feedback, executable, reached_goal, mistakes


    def pre_check_action(self, action_instr: str):
        """
        Checks whether
        - the generated action is surrounded by brackets
        - whether the action is a valid action of the domain
        - whether the number of arguments matches
        - whether all arguments are available objects
        :param action_instr:
        :return:
        """
        feedback = ''
        mistakes = []
        assert len(action_instr) != 0, 'Empty plan (or line) was not processed correctly'

        if not action_instr[0] == '(' or not action_instr[-1] == ')':
            corrected_format = f'({action_instr})'
            feedback = f'The format of the action in the plan is wrong. Each action needs to be enclosed by brackets, i.e. the correct format would be {corrected_format}.\n'
            mistakes.append('action-formatting')

        pred_action_name, pred_objects = self.parse_pddl_tuple(action_instr, decode=False)

        # check whether action is part of the domain actions
        try:
            # check whether correct number of arguments for the action are provided
            action_dict = self.get_problem_actions()
            _, actual_objects = self.parse_pddl_tuple(action_dict[pred_action_name], decode=False)
            if len(pred_objects) != len(actual_objects):
                feedback += f'"{pred_action_name}" requires exactly {len(actual_objects)} objects as arguments but {len(pred_objects)} were given.\n'
                mistakes.append('wrong-number-parameters')
        except KeyError:
            feedback += f'"{pred_action_name}" does not match any possible actions.\n'
            mistakes.append('wrong-action')

        # check whether only observable objects are part of the prediction
        not_matching_objects = []
        current_problem_objects = self.get_problem_objects()
        for pred_obj in pred_objects:
            pred_obj_lower = pred_obj.lower()
            if pred_obj_lower not in current_problem_objects:
                not_matching_objects.append(pred_obj)

        if not_matching_objects:
            if len(not_matching_objects) == 1:
                feedback += f'{not_matching_objects[0]} is not an available object in this task. '
            else:
                feedback += f'{", ".join(not_matching_objects)} are not available objects in this task. '
            mistakes.append('undefined-objects')
        if feedback != '':
            return feedback, mistakes
        else:
            return action_instr, mistakes

    def get_feedback_successful(self, action: str) -> str:
        return f'The action "{action}" was successfully executed.'

    def get_feedback_unsat(self, advice: List[str]) -> str:

        feedback_type, failed_action, should_be_true, should_be_false, static_should_be_true, static_should_be_false = self.parse_feedback_unsat(advice=advice)

        if feedback_type == "goal":
            feedback = f'The generated plan does not reach the goal:\n'

            if len(should_be_false) > 0:
                feedback += f'The following needs to be false but is true after executing all actions:\n'
            for pred in should_be_false:
                feedback += f'- {self.get_description_pred(pred)}\n'

            if len(should_be_true) > 0:
                feedback += f'The following needs to be true but is false after executing all actions:\n'
            for pred in should_be_true:
                feedback += f'- {self.get_description_pred(pred)}\n'

        elif feedback_type == 'precondition':

            feedback = f'the preconditions of the action are not satisfied:\n'

            if len(static_should_be_false) > 0 or len(static_should_be_true) > 0:
                feedback += f'In this task instance\n'
                for pred in static_should_be_false:
                    feedback += f'- {self.get_description_pred(pred)}\n'
                for pred in static_should_be_true:
                    pred_description = self.get_description_pred(pred)
                    neg_pred_description = self.negate_pred_description(pred_description)
                    feedback += f'- {neg_pred_description}\n'

            if len(should_be_false) > 0 or len(should_be_true) > 0:
                feedback += f'At that specific step\n'
                for pred in should_be_false:
                    feedback += f'- {self.get_description_pred(pred)}\n'
                for pred in should_be_true:
                    pred_description = self.get_description_pred(pred)
                    neg_pred_description = self.negate_pred_description(pred_description)
                    feedback += f'- {neg_pred_description}\n'

        else:
            raise NotImplementedError

        return feedback


    def negate_pred_description(self, pred_description):

        neg_description = f'it is not the case that {pred_description}'
        return neg_description


    def parse_feedback_unsat(self, advice: List[str]) -> Tuple[str, str, list, list, list, list]:
        """
        input looks like ['The goal is not satisfied',
                          '(Set (on c b) to true)']
        or if more goals: ['The goal is not satisfied',
                            '(Follow each of:',
                            '(Set (on c b) to true)',
                            'and (Set (on a d) to true)',
                            ')']
        or if preconditions missing: ['(unstack c b) has an unsatisfied precondition at time 1',
                        '(Follow each of:',
                        '(Set (on c b) to true)',
                        'and (Set (clear c) to true)',
                        ')']
        :param advice:
        :return:
        """

        message = advice.pop(0)
        if 'precondition' in message:
            feedback_type = 'precondition'
            failed_action = re.findall(r'\(.*\)', message)[0]
        else:
            feedback_type = 'goal'
            failed_action = ''

        should_be_true = []
        should_be_false = []
        static_should_be_true = []
        static_should_be_false = []
        reg = r'Set \(.*\) to'
        for line in advice:
            line = line[1:] if line.startswith('(') else line
            line = line[:-1] if line.endswith(')') else line
            if line == '' or line.startswith('Follow'):
                continue

            fact = re.findall(reg, line)
            assert len(fact) == 1
            fact = fact[0]
            fact = fact.replace('Set ', '').replace(' to', '')
            pred_name = fact.split(' ')[0].replace('(', '')

            if 'true' in line:      # facts that are false but should be true
                if pred_name in self.static_preds:
                    static_should_be_true.append(fact)
                else:
                    should_be_true.append(fact)
            elif 'false' in line:   # facts that are true but should be false
                if pred_name in self.static_preds:
                    static_should_be_false.append(fact)
                else:
                    should_be_false.append(fact)

        return feedback_type, failed_action, should_be_true, should_be_false, static_should_be_true, static_should_be_false


    def parse_val_output(self, response: str) -> Tuple[bool, bool, list, list, list]:
        """
        Works only for plans consisting of 1 action
        :param response:
        :return:
        """
        goal_satisfied = False
        plan_executable = False

        advice_goal = []
        advice_precond = []
        effects = []

        reached_execution = False
        reached_unmet_pre = False
        reached_effect = False
        reached_advice = False

        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if 'Successful plans:' in line or 'Failed plans:' in line:
                break

            if reached_execution:
                if 'Plan failed because' in line:
                    reached_unmet_pre = True
                    plan_executable = False
                else:                               # then the plan is executable
                    reached_effect = True

            if reached_effect and line:
                if 'Deleting' in line or 'Adding' in line:
                    effects.append(line)
                elif 'executed successfully' in line:
                    plan_executable = True
                elif 'Plan valid' in line:
                    goal_satisfied = True   # plan is valid if plan is executable and goal is satisfied

            elif reached_effect and not line:
                reached_effect = False

            if reached_unmet_pre and not line:    # processed all unmet preconditions
                reached_unmet_pre = False

            if reached_advice and line:
                if not plan_executable:
                    advice_precond.append(line)
                elif not goal_satisfied:
                    advice_goal.append(line)

            if 'Plan Repair' in line:
                reached_advice = True

            if line.startswith('Checking next happening'):
                reached_execution = True

            if 'Bad plan description!' in line:
                advice_precond.append('The action cannot be executed. ')
                #self.problem.language.is_subtype()

        return goal_satisfied, plan_executable, effects, advice_goal, advice_precond


    def check_type_constraints(self, action_instr: str):
        """
        Check whether the types of the parameters match the action scheme
        :param action_instr:
        :return:
        """
        pred_action_name, pred_objects = self.parse_pddl_tuple(action_instr, decode=False)
        type_problems = []

        action_arg_types = self.problem.actions[pred_action_name].sort()
        predicted_arg_types = []
        predicted_obj_names = []
        constants_with_types = self.problem.language.constants()
        for pr_o in pred_objects:
            ordered_constants_names = [cons.name for cons in constants_with_types]
            obj_ind = ordered_constants_names.index(pr_o)
            constant = constants_with_types[obj_ind]
            predicted_arg_types.append(constant.sort)
            predicted_obj_names.append(constant.name)

        for arg_ind, (gold_type, pred_type) in enumerate(zip(action_arg_types, predicted_arg_types)):
            correct_type = False
            if gold_type == pred_type:
                correct_type = True
            else:
                type_hierarchy = self.problem.language.ancestor_sorts   # dict with key for each type, value is set of all ancestor sorts
                super_types = type_hierarchy[pred_type]
                if gold_type in super_types:
                    correct_type = True

            if not correct_type:
                type_problems.append(f'{predicted_obj_names[arg_ind]} has type {pred_type} but needs to be of type {gold_type}')

        if type_problems:
            feedback = f'The action {self.get_description_action(action_instr)} is not applicable because '
            feedback += ' and '.join(type_problems)
            return feedback
        else:
            return action_instr

    # ---------------------------------- Other functions ---------------------------------

    def parse_pddl_tuple(self, pddl_tup: str, decode=False) -> Tuple[str, list]:
        """

        :param pddl_tup: in the form as in a plan, e.g. (stack c b) or for predicates e.g. (ontable c)
        :param decode:
        :return: name of the predicate/action, names of the objects
        """
        pddl_tup = pddl_tup.replace('(', '')
        pddl_tup = pddl_tup.replace(')', '')
        pred_name, objs = pddl_tup.split()[0], pddl_tup.split()[1:]
        object_names = objs

        return pred_name, object_names

    def update_current_state(self, effects: list):
        """

        :param effects:
        :return:
        """
        for effect in effects:
            effect_type = effect.split(' ')[0]
            fact = effect.split(' ')[1:]
            fact = ' '.join(fact)
            if effect_type == 'Deleting':
                #assert fact in self.facts_current_state
                try:
                    self.facts_current_state.remove(fact)
                except ValueError:
                    continue
            elif effect_type == 'Adding':
                self.facts_current_state.append(fact)
            else:
                print(f'Warning: unknown type of effect action {effect_type}')

    def get_problem(self, instance, domain: str):
        """

        :param instance: path to the instance_file
        :param domain: path to the domain_file
        :return:
        """
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        problem = reader.parse_instance(instance)
        domain_name = get_domain_name(domain_path=domain)
        domain_name_problem = problem.domain_name
        assert domain_name == domain_name_problem
        return problem


    def get_problem_objects(self) -> list:
        """

        :return:
        """
        constants = list(self.problem.language.constants())
        object_constants = [str(c) for c in constants]
        return object_constants


    def get_problem_predicates(self) -> dict:
        """

        :return:
        """
        predicates = list(self.problem.language.predicates)
        predicates_str = [str(p) for p in predicates]

        only_preds = dict()
        for pr in predicates_str:
            predicate_name = pr.split('/')[0]
            arity = pr.split('/')[1]
            if predicate_name not in ['=', '!']:
                predicate_name = predicate_name.lower()
                only_preds[predicate_name] = predicate_name + arity * ' object'

        return only_preds

    def get_problem_actions(self) -> dict:
        """

        :return:
        """
        actions_problem: OrderedDict[str, Action] = self.problem.actions
        actions = dict()
        for action_name, action_temp in actions_problem.items():
            str_action = str(action_temp).lower()
            variables = re.findall(r'\?.*?: ', str_action)
            for v in variables:
                str_action = str_action.replace(v, '')

            action_name = action_name.lower()
            actions[action_name] = self.convert_pre2in(str_action)

        return actions

    def get_static_preds(self) -> list:
        """

        :return:
        """
        actions: OrderedDict[str, Action] = self.problem.actions
        predicates: List[Predicate] = self.problem.language.predicates
        non_static_preds = set()
        for a in actions.values():
            effects = a.effects
            for effect in effects:
                predicate_name = effect.atom.predicate.symbol
                non_static_preds.add(predicate_name)

        static_preds = []
        for pred in predicates:
            if not pred.name in non_static_preds:
                static_preds.append(pred.name)

        return static_preds

    def get_object_types(self) -> Tuple[Dict[str, str], Dict[str, list]]:

        object_type_dict = dict()
        type_object_dict = defaultdict(list)

        problem_constants = list(self.problem.language.constants())

        for const in problem_constants:
            object_name = str(const.name)
            object_type = str(const.sort.name)
            object_type_dict[object_name] = object_type
            type_object_dict[object_type].append(object_name)

        return object_type_dict, type_object_dict


    def get_description_initial_state(self):
        return self.problem_descript


    def get_description_goal_state(self):
        return self.goal_descript


    def get_description_action(self, action_str):
        return action_str

    def get_description_pred(self, pred_str):
        return pred_str

    def get_description_state_basic(self, state_facts: list, sep=''):
        pred_descriptions = state_facts
        pred_descriptions.sort()
        return pred_descriptions

    def get_description_state(self, state_facts: list, sep=''):

        all_types = list(self.types2objects.keys())
        if len(all_types) == 1:
            all_objects = self.possible_objects
            all_objects.sort()
            obj_descrip = ' '.join(all_objects)
            if not all_types[0] == 'object':
                # if not of the most general default type 'object' then add the type
                obj_descrip += f' - {all_types[0]}'
        else:
            for t in all_types:
                objects_of_t = self.types2objects[t]
                obj_descrip = ' '.join(objects_of_t)
                obj_descrip += f' - {t}\n'

        object_state_description = f'(:objects\n{obj_descrip}\n)'

        state_fact_description = f'(:init\n{self.get_description_state_basic(state_facts=state_facts, sep=sep)}\n)'

        state_description = f'{object_state_description}\n{state_fact_description}'
        return state_description




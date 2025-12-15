import os
import re
import shutil
import random
from datetime import datetime
from typing import Dict
from pathlib import Path
from collections import defaultdict
from tarski.io import PDDLReader
from tarski.syntax import Atom, CompoundFormula
from utils.domain_class import Domain
from utils.paths import TEMP_DIR


# TODO: not working for constants, quantifiers, conditionals

def create_ablated_dataset(data_dir: Path, new_data_dir: Path):

    domain_file = data_dir / 'domain.pddl'
    instance_dir = data_dir / 'problems_pddl'
    plan_dir = data_dir / 'plans_pddl'
    plan_dir_gbfs = data_dir / 'plans_gbfs'

    new_domain_file = new_data_dir / 'domain.pddl'
    new_inst_dir = new_data_dir / 'problems_pddl'
    new_plan_dir = new_data_dir / 'plans_pddl'
    new_plan_dir_gbfs = new_data_dir / 'plans_gbfs'
    new_inst_dir.mkdir(exist_ok=True, parents=True)
    new_plan_dir.mkdir(exist_ok=True, parents=True)
    new_plan_dir_gbfs.mkdir(exist_ok=True, parents=True)

    # parse PDDL
    domain = Domain(domain_file=domain_file)
    with open(domain_file, 'r') as f:
        content = f.read()
        requirements_list = re.findall(r'\(:requirements .*\)?', content)
        requirements = '' if len(requirements_list) == 0 else requirements_list[0]

    # create ablated domain file
    new_actions, new_predicates, new_types, action_mappings, predicate_mappings, type_mappings = create_ablated_domain(domain=domain)
    create_ablated_domain_file(new_actions=new_actions,
                               new_predicates=new_predicates,
                               new_types=new_types,
                               requirements=requirements,
                               new_file_path=new_domain_file)

    # modify all problem files
    for instance in os.listdir(instance_dir):
        problem_file = instance_dir / str(instance)
        plan_name = str(instance).replace('.pddl', '_plan.txt')
        plan_file = plan_dir / plan_name
        plan_file_gbfs = plan_dir_gbfs / plan_name
        inst_id = str(instance).replace('instance-', '').replace('.pddl', '')

        new_prob_file = new_inst_dir / str(instance)
        new_plan_file = new_plan_dir / plan_name
        new_plan_file_gbfs = new_plan_dir_gbfs / plan_name

        new_initial_facts, new_goal_facts, object_mappings, type_object_dict = create_ablated_problem(
            domain_file=domain_file,
            problem_file=problem_file,
            type_mappings=type_mappings,
            predicate_mappings=predicate_mappings)

        create_ablated_problem_file(inst_id=inst_id,
                                    new_init_state=new_initial_facts,
                                    new_goal_state=new_goal_facts,
                                    type_object_dict=type_object_dict,
                                    object_mappings=object_mappings,
                                    new_prob_file=new_prob_file)

        # optionally modify plan file if exists
        if os.path.exists(plan_file):
            create_ablated_plan_file(action_mappings=action_mappings,
                                     object_mappings=object_mappings,
                                     orig_plan_file=plan_file,
                                     new_plan_file=new_plan_file)

        if os.path.exists(plan_file_gbfs):
            create_ablated_plan_file(action_mappings=action_mappings,
                                     object_mappings=object_mappings,
                                     orig_plan_file=plan_file_gbfs,
                                     new_plan_file=new_plan_file_gbfs)


def create_ablated_plan_file(action_mappings,
                             object_mappings,
                             orig_plan_file,
                             new_plan_file):

    orig_plan = []
    new_plan = []
    with open(orig_plan_file, 'r') as f:
        for line in f.readlines():
            if line.startswith(';'):
                continue
            orig_plan.append(line.strip())

    for action in orig_plan:
        ac_no_brackets = action.replace('(', '').replace(')', '')
        parts = ac_no_brackets.split(' ')
        action_name = parts[0]
        args = parts[1:]

        new_action_name = action_mappings[action_name]
        new_action_parts = [new_action_name]
        for arg in args:
            new_arg = object_mappings[arg]
            new_action_parts.append(new_arg)

        action_str = ' '.join(new_action_parts)
        action_str = f'({action_str})'
        new_plan.append(action_str)

    with open(new_plan_file, 'w') as f:
        new_plan_str = '\n'.join(new_plan)
        f.write(new_plan_str)


def create_ablated_domain_file(new_actions,
                               new_predicates,
                               new_types,
                               requirements,
                               new_file_path,
                               domain_name: str = 'puzzle'):

    domain_str = f'(define (domain {domain_name})\n\n'
    if requirements:
        domain_str += f'\t{requirements}\n'

    if len(list(new_types.keys())):
        typing = True
    else:
        typing = False

    if typing:
        domain_str += '\t(:types\n'
        for type_name, sub_types in new_types.items():
            type_str = ' '.join(sub_types)
            type_str += f' - {type_name}'
            domain_str += f'\t\t{type_str}\n'
        domain_str += f'\t)\n\n'

    domain_str += '\t(:predicates\n'
    for pred, pred_data in new_predicates.items():
        pred_str = f'{pred}'
        for arg, arg_type in pred_data.items():
            if typing:
                pred_str += f' {arg} - {arg_type}'
            else:
                pred_str += f' {arg}'
        domain_str += f'\t\t({pred_str})\n'
    domain_str += f'\t)\n\n'

    for action, action_data in new_actions.items():
        domain_str += f'\t(:action {action}\n\t\t:parameters '
        param_str_list = []
        for param_name, param_type in action_data['parameters'].items():
            if typing:
                param_str_list.append(f'{param_name} - {param_type}')
            else:
                param_str_list.append(f'{param_name}')
        param_str = ' '.join(param_str_list)
        domain_str += f'({param_str})\n\t\t:precondition (and\n'

        for precond in action_data['pos_preconditions']:
            precond_str = ' '.join(list(precond))
            domain_str += f'\t\t\t({precond_str})\n'
        for precond in action_data['neg_preconditions']:
            precond_str = ' '.join(list(precond))
            domain_str += f'\t\t\t(not ({precond_str}))\n'

        domain_str += f'\t\t)\n\t\t:effect (and\n'
        for effect in action_data['add_effects']:
            effect_str = ' '.join(list(effect))
            domain_str += f'\t\t\t({effect_str})\n'
        for effect in action_data['del_effects']:
            effect_str = ' '.join(list(effect))
            domain_str += f'\t\t\t(not ({effect_str}))\n'
        domain_str += f'\t\t)\n\t)\n'

    domain_str += f')\n'

    with open(new_file_path, 'w') as f:
        f.write(domain_str)


def create_ablated_domain(domain: Domain):

    new_actions = dict()
    new_predicates = dict()
    new_types = dict()
    action_mappings = dict()
    predicate_mappings = dict()
    type_mappings = dict()

    current_id = 1
    current_type_id = 1
    current_predicate_id = 1

    for ty, type_data in domain.types.items():
        if ty == 'object':          # special type
            type_mappings[ty] = ty
        if ty not in type_mappings.keys():
            new_ty_name = f'type{current_type_id}'
            current_type_id += 1
            type_mappings[ty] = new_ty_name
        else:
            new_ty_name = type_mappings[ty]
        new_subtypes = []
        for sub_type in type_data:
            if sub_type not in type_mappings.keys():
                new_subty_name = f'type{current_type_id}'
                current_type_id += 1
                type_mappings[sub_type] = new_subty_name
            else:
                new_subty_name = type_mappings[sub_type]
            new_subtypes.append(new_subty_name)
        new_types[new_ty_name] = new_subtypes

    for pred, predicate_data in domain.predicates.items():
        new_pred_name = f'predicate{current_predicate_id}'
        current_predicate_id += 1
        predicate_mappings[pred] = new_pred_name

        new_parameter_dict = dict()
        current_param_id = 1
        for param_name, param_type in predicate_data.items():
            new_param_name = f'?arg{current_param_id}'
            current_param_id += 1
            new_param_type = type_mappings.get(param_type, 'object')
            new_parameter_dict[new_param_name] = new_param_type

        new_predicates[new_pred_name] = new_parameter_dict

    for action, action_data in domain.actions.items():

        new_action_name = f'action{current_id}'
        current_id += 1
        action_mappings[action] = new_action_name

        parameters = action_data['parameters']
        pos_preconditions = action_data['pos_preconditions']
        neg_preconditions = action_data['neg_preconditions']
        add_effects = action_data['add_effects']
        del_effects = action_data['del_effects']

        new_parameter_dict = dict()

        parameter_mappings = dict()
        current_param_id = 1
        for param, param_type in parameters.items():
            new_param_name = f'?arg{current_param_id}'
            current_param_id += 1
            parameter_mappings[param] = new_param_name

            new_param_type = type_mappings.get(param_type, 'object')
            new_parameter_dict[new_param_name] = new_param_type

        new_pos_precond = adapt_preconds_effects(parameter_mappings=parameter_mappings,
                                                 predicate_mappings=predicate_mappings,
                                                 prec_list=pos_preconditions)
        new_neg_precond = adapt_preconds_effects(parameter_mappings=parameter_mappings,
                                                 predicate_mappings=predicate_mappings,
                                                 prec_list=neg_preconditions)
        new_add_effects = adapt_preconds_effects(parameter_mappings=parameter_mappings,
                                                 predicate_mappings=predicate_mappings,
                                                 prec_list=add_effects)
        new_del_effects = adapt_preconds_effects(parameter_mappings=parameter_mappings,
                                                 predicate_mappings=predicate_mappings,
                                                 prec_list=del_effects)

        new_actions[new_action_name] = {'parameters': new_parameter_dict,
                                        'pos_preconditions': new_pos_precond,
                                        'neg_preconditions': new_neg_precond,
                                        'add_effects': new_add_effects,
                                        'del_effects': new_del_effects}

    return new_actions, new_predicates, new_types, action_mappings, predicate_mappings, type_mappings


def adapt_preconds_effects(parameter_mappings,
                           predicate_mappings,
                           prec_list):
    new_prec_list = []
    for precond in prec_list:
        pred = precond[0]
        params = precond[1:]
        new_precond = [predicate_mappings[pred]]
        for param in params:
            new_param = parameter_mappings[param]
            new_precond.append(new_param)
        new_prec_list.append(tuple(new_precond))

    return new_prec_list


def create_ablated_problem(domain_file, problem_file, type_mappings, predicate_mappings):

    lower_cased_domain_file = create_lowercase_file(domain_file)
    lower_cased_problem_file = create_lowercase_file(problem_file)
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(lower_cased_domain_file)
    problem = reader.parse_instance(lower_cased_problem_file)

    constants = list(problem.language.constants())
    type_object_dict = defaultdict(list)
    object_mappings = dict()
    for obj_id, const in enumerate(constants):
        object_name = str(const.name)
        object_type = str(const.sort.name)

        new_object_name = f'object{obj_id}'
        try:
            new_object_type = type_mappings[object_type]
            type_object_dict[new_object_type].append(new_object_name)
        except KeyError:
            pass
        object_mappings[object_name] = new_object_name

    facts_initial_state = [convert_pre2in(initial) for initial in list(problem.init.as_atoms())]
    goal_state = process_goal_conditions(problem)

    new_initial_facts = []
    new_goal_facts = []

    for fact in facts_initial_state:
        parts = fact.split(' ')
        pred_name = parts[0]
        new_pred_name = predicate_mappings[pred_name]
        new_fact = [new_pred_name]
        for arg in parts[1:]:
            new_obj = object_mappings[arg]
            new_fact.append(new_obj)
        new_initial_facts.append(new_fact)

    for cond, goal_conds in goal_state.items():
        for fact in goal_conds:
            parts = fact.split(' ')
            pred_name = parts[0]
            new_pred_name = predicate_mappings[pred_name]
            if cond.startswith('pos_'):
                new_fact = [new_pred_name]
            else:
                new_fact = ['not', new_pred_name]
            for arg in parts[1:]:
                new_obj = object_mappings[arg]
                new_fact.append(new_obj)
            new_goal_facts.append(new_fact)

    return new_initial_facts, new_goal_facts, object_mappings, type_object_dict


def convert_pre2in(action):
    action = str(action)
    action_name, action_args = action.split('(')
    new_action_str = f'{action_name} {action_args}'
    new_action_str = new_action_str.replace(',', ' ')
    new_action_str = new_action_str.replace(')', '')
    new_action_str = new_action_str.strip()
    return new_action_str


def process_goal_conditions(problem) -> Dict[str, list]:

    pos_goal_conditions = []
    neg_goal_conditions = []
    if isinstance(problem.goal, CompoundFormula):
        operator = problem.goal.connective
        if operator.name == 'And':
            for sub in problem.goal.subformulas:
                if isinstance(sub, Atom):
                    pos_goal_conditions.append(convert_pre2in(sub))
                elif sub.connective.name == 'Not':
                    assert len(sub.subformulas) == 1
                    pred_str = convert_pre2in(sub.subformulas[0])
                    neg_goal_conditions.append(pred_str)
        elif operator.name == 'Not':
            assert len(problem.goal.subformulas) == 1
            pred_str = convert_pre2in(problem.goal.subformulas[0])
            neg_goal_conditions.append(pred_str)
        else:
            raise ValueError
    elif isinstance(problem.goal, Atom):
        pos_goal_conditions.append(convert_pre2in(problem.goal))
    else:
        raise ValueError

    goal_conditions = {'pos_conditions': pos_goal_conditions,
                       'neg_conditions': neg_goal_conditions}

    return goal_conditions


def create_lowercase_file(orig_file):
    # Make sure temp dir exists
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    # Create temporary file
    domain_file_name = os.path.split(orig_file)[-1]
    time_stamp = str(datetime.now())
    time_stamp = [char for char in time_stamp if char != ' ']
    time_stamp = ''.join(time_stamp)
    tmp_file_int = random.randint(0, 1000000)
    tmp_file_int2 = random.randint(0, 1000000)
    lowercase_file = os.path.join(TEMP_DIR, f'tmp_{time_stamp}_{tmp_file_int}_{tmp_file_int2}_{domain_file_name}')
    with open(lowercase_file, 'w') as new:
        with open(orig_file, 'r') as orig:
            orig_text = orig.read()
            new_text = orig_text.lower()
            new.write(new_text)
    return lowercase_file


def create_ablated_problem_file(inst_id,
                                new_init_state,
                                new_goal_state,
                                type_object_dict,
                                object_mappings,
                                new_prob_file,
                                domain_name: str = 'puzzle'):
    problem_str = f'(define (problem {domain_name}-{inst_id})\n(:domain {domain_name})\n'

    obj_str = '(:objects\n\t'
    if len(list(type_object_dict.keys())):
        for obj_type, objects in type_object_dict.items():
            for obj in objects:
                obj_str += f'{obj} '
            obj_str += f'- {obj_type}\n\t'
    else:
        all_objects = list(object_mappings.values())
        obj_str += ' '.join(all_objects)
        obj_str += '\n'
    obj_str += ')'

    new_init_state.sort()
    new_goal_state.sort()

    init_str = ''
    for init_fact in new_init_state:
        init_fact_str = ' '.join(init_fact)
        init_str += f'\t({init_fact_str})\n'

    goal_str = ''
    for goal_fact in new_goal_state:
        if goal_fact[0] == 'not':
            goal_fact_str = ' '.join(goal_fact[1:])
            goal_str += f'\t(not ({goal_fact_str}))\n'
        else:
            goal_fact_str = ' '.join(goal_fact)
            goal_str += f'\t({goal_fact_str})\n'

    init_str = f'(:init\n{init_str})'

    goal_str = f'(:goal\n(and\n{goal_str}))'

    problem_str += f'{obj_str}\n\n{init_str}\n\n{goal_str}\n)'

    with open(new_prob_file, 'w') as f:
        f.write(problem_str)


if __name__ == '__main__':

    data_dirs = [
        #('silver', 'gripper'),
        ('silver', 'ferry'),
        #('additional', 'ferry'),
        #('additional', 'grippers'),
        #('additional', 'visitall'),
        ('silver', 'heavy'),
        ('silver', 'hiking'),
        ('silver', 'delivery'),
        ('silver', 'spanner'),
        ('silver', 'miconic'),
        #('additional', 'logistics'),
        #('additional', 'satellite')
    ]

    data_dirs = [
        #('icaps', 'ferry'),
        #('icaps', 'heavy'),
        #('icaps', 'hiking'),
        #('icaps', 'delivery'),
        #('icaps', 'spanner'),
        #('icaps', 'miconic'),
        ('icaps', 'gripper')
    ]

    for parent_dir, domain in data_dirs:
        data_dir = Path('../data') /parent_dir / domain

        # TODO
        new_data_dir = Path('../data') / f'{parent_dir}_ablated2' / domain

        new_data_dir.mkdir(exist_ok=True, parents=True)

        create_ablated_dataset(data_dir=data_dir, new_data_dir=new_data_dir)

        split_file = 'seed_splits.json'
        eval_split_file = 'seed_splits_eval.json'
        if os.path.exists(data_dir / split_file):
            shutil.copy(os.path.join(data_dir, split_file),
                        os.path.join(new_data_dir, split_file))

        if os.path.exists(data_dir / eval_split_file):
            shutil.copy(os.path.join(data_dir, eval_split_file),
                        os.path.join(new_data_dir, eval_split_file))

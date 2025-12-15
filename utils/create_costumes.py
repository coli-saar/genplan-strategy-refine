import os
import re
import json
import shutil
from pathlib import Path
from collections import defaultdict
from tarski.io import PDDLReader
from utils.domain_class import Domain
from utils.create_obfuscated import create_ablated_domain_file, create_ablated_problem_file, create_ablated_plan_file, adapt_preconds_effects, create_lowercase_file, convert_pre2in, process_goal_conditions
from utils.helper import run_validator
from utils.paths import VAL


# TODO: not working for constants, quantifiers, conditionals

def create_costumized_dataset(data_dir: Path,
                              new_data_dir: Path,
                              mappings_file):

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

    with open(mappings_file, 'r') as f:
        mappings = json.load(f)

    # create ablated domain file
    new_actions, new_predicates, new_types, action_mappings, predicate_mappings, type_mappings = create_costumized_domain(
        domain=domain,
        mappings=mappings)
    create_ablated_domain_file(new_actions=new_actions,
                               new_predicates=new_predicates,
                               new_types=new_types,
                               requirements=requirements,
                               new_file_path=new_domain_file,
                               domain_name=mappings['domain'])

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
            predicate_mappings=predicate_mappings,
            object_prefix_map=mappings['objects']
        )

        create_ablated_problem_file(inst_id=inst_id,
                                    new_init_state=new_initial_facts,
                                    new_goal_state=new_goal_facts,
                                    type_object_dict=type_object_dict,
                                    object_mappings=object_mappings,
                                    new_prob_file=new_prob_file,
                                    domain_name=mappings['domain']
        )

        # optionally modify plan file if exists
        if os.path.exists(plan_file):
            create_ablated_plan_file(action_mappings=action_mappings,
                                     object_mappings=object_mappings,
                                     orig_plan_file=plan_file,
                                     new_plan_file=new_plan_file)

            """cmd = f'{VAL}/validate -v {new_domain_file} {new_prob_file} {new_plan_file}'
            val_response = os.popen(cmd).read()
            if not 'Plan valid' in val_response:
                print(problem_file)
                print(val_response)
            assert 'Plan valid' in val_response"""

        if os.path.exists(plan_file_gbfs):
            create_ablated_plan_file(action_mappings=action_mappings,
                                     object_mappings=object_mappings,
                                     orig_plan_file=plan_file_gbfs,
                                     new_plan_file=new_plan_file_gbfs)

            """cmd = f'{VAL}/validate -v {new_domain_file} {new_prob_file} {new_plan_file_gbfs}'
            val_response = os.popen(cmd).read()
            if not 'Plan valid' in val_response:
                print(problem_file)
                print(val_response)
            assert 'Plan valid' in val_response"""


def create_costumized_domain(domain: Domain, mappings: dict):

    new_actions = dict()
    new_predicates = dict()
    new_types = dict()
    simple_action_mappings = dict()
    simple_predicate_mappings = dict()
    simple_type_mappings = dict()

    for ty, type_data in domain.types.items():
        new_ty_name = mappings['typing'][ty]
        new_subtypes = []
        for sub_type in type_data:
            new_subty_name = mappings['typing'][sub_type]
            if sub_type not in simple_type_mappings.keys():
                simple_type_mappings[sub_type] = new_subty_name

            new_subtypes.append(new_subty_name)
        new_types[new_ty_name] = new_subtypes

    for pred, predicate_data in domain.predicates.items():
        new_pred_name = mappings['predicates'][pred]['name']
        simple_predicate_mappings[pred] = new_pred_name

        new_parameter_dict = dict()
        for param_name, param_type in predicate_data.items():
            new_param_name = mappings['predicates'][pred]['args'][param_name]

            new_param_type = simple_type_mappings.get(param_type, 'object')
            new_parameter_dict[new_param_name] = new_param_type

        new_predicates[new_pred_name] = new_parameter_dict

    for action, action_data in domain.actions.items():

        new_action_name = mappings['actions'][action]['name']
        simple_action_mappings[action] = new_action_name

        parameters = action_data['parameters']
        pos_preconditions = action_data['pos_preconditions']
        neg_preconditions = action_data['neg_preconditions']
        add_effects = action_data['add_effects']
        del_effects = action_data['del_effects']

        new_parameter_dict = dict()
        parameter_mappings = dict()
        for param, param_type in parameters.items():
            new_param_name = mappings['actions'][action]['args'][param]
            parameter_mappings[param] = new_param_name

            new_param_type = simple_type_mappings.get(param_type, 'object')
            new_parameter_dict[new_param_name] = new_param_type

        new_pos_precond = adapt_preconds_effects(parameter_mappings=parameter_mappings,
                                                 predicate_mappings=simple_predicate_mappings,
                                                 prec_list=pos_preconditions)
        new_neg_precond = adapt_preconds_effects(parameter_mappings=parameter_mappings,
                                                 predicate_mappings=simple_predicate_mappings,
                                                 prec_list=neg_preconditions)
        new_add_effects = adapt_preconds_effects(parameter_mappings=parameter_mappings,
                                                 predicate_mappings=simple_predicate_mappings,
                                                 prec_list=add_effects)
        new_del_effects = adapt_preconds_effects(parameter_mappings=parameter_mappings,
                                                 predicate_mappings=simple_predicate_mappings,
                                                 prec_list=del_effects)

        new_actions[new_action_name] = {'parameters': new_parameter_dict,
                                        'pos_preconditions': new_pos_precond,
                                        'neg_preconditions': new_neg_precond,
                                        'add_effects': new_add_effects,
                                        'del_effects': new_del_effects}

    return new_actions, new_predicates, new_types, simple_action_mappings, simple_predicate_mappings, simple_type_mappings


def get_object_mappings(object_map, all_objects, type_mappings):
    type_object_dict = defaultdict(list)
    object_mappings = dict()

    for obj in all_objects:
        obj_name = str(obj.name)
        obj_type = str(obj.sort.name)

        # check whether the exact object name is in the mappings
        if obj_name in object_map.keys():
            new_obj_name = object_map[obj_name]
            #print(obj_name)
        else:
            new_parts = []
            parts = split_alpha_numeric(s=obj_name)
            if len(parts) == 0:
                new_obj_name = object_map[obj_name]
            else:
                for part in parts:
                    assert len(part) == 2
                    prefix = part[0]
                    suffix = part[1]

                    new_prefix = object_map[prefix]
                    name_part = f'{new_prefix}{suffix}'
                    new_parts.append(name_part)

                new_obj_name = '_'.join(new_parts)
        object_mappings[obj_name] = new_obj_name

        try:
            new_object_type = type_mappings[obj_type]
            type_object_dict[new_object_type].append(new_obj_name)
        except KeyError:
            pass

    return object_mappings, type_object_dict


def split_alpha_numeric(s: str):
    """
    Splits a string into pairs of:
      [alphabetical part, numeric (or _ + numeric) part]

    Examples:
        car0   -> [['car', '0']]
        l_3    -> [['l', '_3']]
        f3_b0  -> [['f', '3'], ['b', '0']]
        x0y0   -> [['x', '0'], ['y', '0']]
    """
    # Find all patterns like: letters followed by optional underscore and digits

    separator = '-' if '-' in s else '_'
    if separator == '_':
        reg1 = r'([A-Za-z]+)(?:_)?(\d+)'
    else:
        reg1 = r'([A-Za-z]+)(?:-)?(\d+)'
    parts = re.findall(reg1, s)

    # Reconstruct the second element (prefix underscore if present)
    result = []
    if separator == '_':
        reg2 = r'([A-Za-z]+)(?:(_)?(\d+))'
    else:
        reg2 = r'([A-Za-z]+)(?:(-)?(\d+))'
    for match in re.finditer(reg2, s):
        alpha = match.group(1)
        underscore = match.group(2) or ''
        digits = match.group(3) or ''
        result.append([alpha, underscore + digits])

    return result



def create_ablated_problem(domain_file,
                           problem_file,
                           type_mappings,
                           predicate_mappings,
                           object_prefix_map):

    lower_cased_domain_file = create_lowercase_file(domain_file)
    lower_cased_problem_file = create_lowercase_file(problem_file)
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(lower_cased_domain_file)
    problem = reader.parse_instance(lower_cased_problem_file)

    constants = list(problem.language.constants())
    print(problem_file)
    object_mappings, type_object_dict = get_object_mappings(object_map=object_prefix_map,
                                                            all_objects=constants,
                                                            type_mappings=type_mappings)

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


if __name__ == '__main__':

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
        new_data_dir = Path('../data') / f'{parent_dir}_costumes2' / f'{domain}_costume'

        new_data_dir.mkdir(exist_ok=True, parents=True)

        create_costumized_dataset(data_dir=data_dir,
                                  new_data_dir=new_data_dir,
                                  mappings_file=os.path.join(new_data_dir, 'mappings.json'))

        split_file = 'seed_splits.json'
        eval_split_file = 'seed_splits_eval.json'
        if os.path.exists(data_dir / split_file):
            shutil.copy(os.path.join(data_dir, split_file),
                        os.path.join(new_data_dir, split_file))

        if os.path.exists(data_dir / eval_split_file):
            shutil.copy(os.path.join(data_dir, eval_split_file),
                        os.path.join(new_data_dir, eval_split_file))

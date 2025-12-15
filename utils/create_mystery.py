import os
import json
import shutil
from pathlib import Path
from argparse import ArgumentParser


def create_costumed_version(orig_dir, new_dir, mappings_file):

    orig_dir = Path(orig_dir)
    new_dir = Path(new_dir)
    new_dir.mkdir(exist_ok=True, parents=True)

    with open(mappings_file, 'r') as f:
        mappings = json.load(f)

    if not os.path.exists(new_dir / 'domain.pddl'):
        modify_domain(orig_domain_path=orig_dir / 'domain.pddl',
                      new_domain_path=new_dir / 'domain.pddl',
                      mappings=mappings)

    orig_prob_dir = orig_dir / 'problems_pddl'
    new_prob_dir = new_dir / 'problems_pddl'
    new_prob_dir.mkdir(exist_ok=True, parents=True)
    modify_problems(orig_prob_dir=orig_prob_dir,
                    new_prob_dir=new_prob_dir,
                    mappings=mappings)

    orig_plan_dir = orig_dir / 'plans_pddl'
    new_plan_dir = new_dir / 'plans_pddl'
    new_plan_dir.mkdir(exist_ok=True, parents=True)
    modify_plans(orig_plan_dir= orig_plan_dir,
                 new_plan_dir=new_plan_dir,
                 mappings=mappings)

    orig_seed_split = orig_dir / 'seed_splits.json'
    new_seed_split = new_dir / 'seed_splits.json'

    shutil.copy(orig_seed_split, new_seed_split)


def modify_domain(orig_domain_path, new_domain_path, mappings: dict[str, dict]):

    with open(orig_domain_path, 'r') as f:
        domain_content = f.read()

    old_d_name = list(mappings['domain'].keys())[0]
    new_d_name = list(mappings['domain'].values())[0]
    domain_content = domain_content.replace(f'domain {old_d_name}', f'domain {new_d_name}')

    domain_content = replace_predicates_or_actions(
        file_content=domain_content,
        mappings=mappings['predicates'],
        dom=True
    )

    for old_action, new_action in mappings['actions'].items():
        if f':action {old_action} ' in domain_content:
            domain_content = domain_content.replace(f':action {old_action} ', f':action {new_action} ')
        elif f':action {old_action}\n' in domain_content:
            domain_content = domain_content.replace(f':action {old_action}\n', f':action {new_action}\n')
        else:
            raise ValueError

    with open(new_domain_path, 'w') as f:
        f.write(domain_content)


def modify_problems(orig_prob_dir: Path, new_prob_dir: Path, mappings: dict[str, dict]):

    for prob_file in os.listdir(orig_prob_dir):
        prob_path = orig_prob_dir / str(prob_file)

        with open(prob_path, 'r') as f:
            problem_content = f.read()

        old_d_name = list(mappings['domain'].keys())[0]
        new_d_name = list(mappings['domain'].values())[0]
        problem_content = problem_content.replace(f':domain {old_d_name}', f':domain {new_d_name}')

        problem_content = replace_predicates_or_actions(file_content=problem_content, mappings=mappings['predicates'])

        new_prob_path = new_prob_dir / str(prob_file)
        with open(new_prob_path, 'w') as f:
            f.write(problem_content)


def modify_plans(orig_plan_dir: Path, new_plan_dir: Path, mappings: dict[str, dict]):

    for plan_file in os.listdir(orig_plan_dir):
        plan_path = orig_plan_dir / str(plan_file)

        with open(plan_path, 'r') as f:
            plan_content = f.read()

        plan_content = replace_predicates_or_actions(file_content=plan_content,
                                                     mappings=mappings['actions'])

        new_plan_path = new_plan_dir / str(plan_file)
        with open(new_plan_path, 'w') as f:
            f.write(plan_content)


def replace_predicates_or_actions(file_content, mappings, dom: bool = False):

    for old_name, new_name in mappings.items():
        if f'({old_name} ' in file_content:
            file_content = file_content.replace(f'({old_name} ', f'({new_name} ')
        elif f'({old_name})' in file_content:
            file_content = file_content.replace(f'({old_name})', f'({new_name})')
        elif dom:
            raise ValueError

    return file_content


def create_mappings():
    pass


def get_object_mappings():
    pass




if __name__ == '__main__':

    orig_dir = '../../data/aaai/ferry'
    new_dir = '../../data/apb/ferry_obs'
    mappings_file = '../../data/apb/ferry_obs/mappings_costume.json'

    create_costumed_version(orig_dir=orig_dir,
                            new_dir=new_dir,
                            mappings_file=mappings_file)

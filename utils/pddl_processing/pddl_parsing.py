import re
from typing import List, Dict
from collections import OrderedDict
from tarski.io import PDDLReader
from tarski.syntax import Predicate, VariableBinding
from tarski.fstrips import Action


# TODO: lowercasing for the actions as well

def parse_ordered_predicates(domain_file, pddl_reader: PDDLReader) -> Dict[str, OrderedDict]:
    """
    Dict[str, OrderedDict]
                    one entry for each predicate;
                        key = predicate name, lower-cased
                        value = dictionary with all parameters and their types in the order as expected by the predicate
                        e.g. {"on": {'?x': 'object', '?y': 'object}, 'handempty': {}}
    :param domain_file:
    :param pddl_reader:
    :return:
    """

    predicates: List[Predicate] = pddl_reader.problem.language.predicates
    predicates = [pred for pred in predicates if not pred.builtin]
    predicate_var_names = get_predicate_variable_names(domain_file=domain_file)

    predicate_dict = dict()
    for pred in predicates:
        pred_signature = []
        predicate_name = pred.name
        predicate_vars = predicate_var_names[predicate_name]
        predicate_arg_sorts = list(pred.sort)
        predicate_arg_types = [s.name for s in predicate_arg_sorts]
        assert len(predicate_vars) == len(predicate_arg_types)
        for var_name, var_type in zip(predicate_vars, predicate_arg_types):
            pred_signature.append((var_name, var_type))
        predicate_dict[predicate_name] = OrderedDict(pred_signature)
    return predicate_dict


def get_predicate_variable_names(domain_file) -> Dict[str, List[str]]:
    """
    {'at': ['?x', '?y'],
     'on': ['?x', '?y'],
     ...}
    :param domain_file:
    :return: dictionary with one item for each predicate

    """
    file_tokens_cleaned = []
    with open(domain_file, 'r') as df:
        for line in df.readlines():
            line = line.strip()
            line = line.lower()
            if ';' in line:
                comment_start = line.index(';')
                line = line[:comment_start]
            if line:
                tokens = line.split()
                file_tokens_cleaned.extend(tokens)
    file_content_cleaned = ' '.join(file_tokens_cleaned)

    predicate_def = re.findall(r'\(:predicates .*?\(:', file_content_cleaned)[0]

    only_preds = predicate_def.replace(':predicates ', '')
    only_preds = only_preds.replace('(:', '')
    only_preds = only_preds.strip()

    preds_list = only_preds.split(') (')

    predicate_vars = dict()
    for pred in preds_list:
        if ')(' in pred:
            sub_preds = pred.split(')(')
            pred = sub_preds[0]
            preds_list.append(sub_preds[1])
        while pred.startswith('('):
            pred = pred[1:]
            pred = pred.strip()
        while pred.endswith(')'):
            pred = pred[:-1]
            pred = pred.strip()
        pred_parts = pred.split(' ')
        pred_name = pred_parts[0]
        pred_arg_names = []
        for part in pred_parts:
            if part.startswith('?'):
                pred_arg_names.append(part)

        predicate_vars[pred_name] = pred_arg_names

    return predicate_vars


def parse_actions(pddl_reader: PDDLReader) -> Dict[str, OrderedDict]:
    """
    one entry for each action
            key = action name; lower-cased
            value = dictionary with all parameters (OrderedDict)
            e.g. {'stack': {'?ob': 'object', '?underob': 'object'},
                  'drive': {'parameters': {'?t': '?truck', '?from': 'location', ...}
    :return:
    """
    actions: OrderedDict[str, Action] = pddl_reader.problem.actions
    actions_dict = dict()

    for a in actions.values():
        a_name: str = a.name
        parameter_variable: VariableBinding = a.parameters
        parameter_dict = parameter_variable.variables

        parameter_list_str_type = []
        for var_name, variable in parameter_dict.items():
            parameter_list_str_type.append((variable.symbol, variable.sort.name))
        parameter_dict_str_type = OrderedDict(parameter_list_str_type)

        actions_dict[a_name] = parameter_dict_str_type

    return actions_dict





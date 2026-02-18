from pathlib import Path
from copy import deepcopy
from typing import Union
from utils.tasks import Task
from utils.flags_scripts import FLAGS
from llm_models.create_and_init_model import create_and_init_model


def create_agent_description_gen(agent_config: dict,
                                 task_dict: dict,
                                 flags_dict: dict) -> dict:
    agent_param = deepcopy(agent_config['args'])
    agent_param['task_dict'] = task_dict
    agent_param['flags'] = flags_dict
    llm_conf_name = agent_config['llm_conf_name']
    agent_param['llm_model'] = create_and_init_model(llm_config=FLAGS.llm_configs[llm_conf_name],
                                                     system_prompt_path=agent_config['system_prompt'],
                                                     flags=flags_dict)

    return agent_param


def create_agent_strategy_gen(agent_config: dict,
                              prompt_task_dict: dict,
                              validation_task_dict: Union[dict, None],
                              flags_dict: dict,
                              llm_gen_dir: Path,
                              res_dir: Path) -> dict:

    agent_param = deepcopy(agent_config['args'])
    agent_param['prompt_task_dict'] = prompt_task_dict
    agent_param['flags'] = flags_dict

    llm_conf_name = agent_config['llm_conf_name']
    agent_param['llm_model'] = create_and_init_model(
        llm_config=FLAGS.llm_configs[llm_conf_name],
        system_prompt_path=agent_config['system_prompt'],
        flags=flags_dict
    )

    assert 'domain_strategy_prompt_file' in agent_param.keys() or 'domain_strategy_merge_prompt_file' in agent_param.keys()

    if 'plan_based_val_args' in agent_param.keys():
        plan_based_val_args = create_agent_strat_val_planbased(
            agent_config=agent_param['plan_based_val_args'],
            task_dict=validation_task_dict,
            flags_dict=flags_dict,
            llm_gen_dir=llm_gen_dir,
            res_dir=res_dir
        )
        plan_based_val_args['log_dir'] = agent_param['log_dir']
        plan_based_val_args['name'] = agent_param['plan_based_val_args']['name']
        agent_param['plan_based_val_args'] = plan_based_val_args

    if 'reflection_prompt_file' in agent_param.keys():
        assert 'plan_based_val_args' in agent_param.keys()
        debug_llm_conf_name = agent_config['debug_llm_conf_name']
        agent_param['llm_model_debug'] = create_and_init_model(
            llm_config=FLAGS.llm_configs[debug_llm_conf_name],
            system_prompt_path=agent_config['system_prompt_debug'],
            flags=flags_dict
        )

    return agent_param


def create_agent_strat_selfval(agent_config: dict,
                               prompt_task_dict: dict,
                               validation_task_dict: dict,
                               flags_dict: dict,
                               llm_gen_dir: Path,
                               res_dir: Path,
                               plan_gen_config: Union[None, dict] = None) -> dict:

    agent_param = deepcopy(agent_config['args'])
    agent_param['prompt_task_dict'] = prompt_task_dict
    agent_param['flags'] = flags_dict
    agent_param['llm_gen_dir'] = llm_gen_dir
    llm_conf_name = agent_config['llm_conf_name']
    agent_param['llm_model'] = create_and_init_model(llm_config=FLAGS.llm_configs[llm_conf_name],
                                                     system_prompt_path=agent_config['system_prompt'],
                                                     flags=flags_dict)

    # Create parameter for refinement agent
    refinement_agent_param = agent_param.pop('refine_agent')
    agent_param['param_refine_agent'] = create_agent_refinement(
        refinement_agent_param,
        flags_dict=flags_dict,
        plan_gen_config=plan_gen_config,
        llm_gen_dir=llm_gen_dir
    )
    agent_param['param_refine_agent']['log_dir'] = agent_param['log_dir']

    # Check if generation of plans is done by separate agent
    if 'plan_gen_prompt_file' in agent_param.keys():
        gen_plan_args = agent_param.pop('plan_gen_prompt_file')
    else:
        assert plan_gen_config is not None
        gen_plan_args = create_agent_plan_gen(plan_gen_config,
                                              flags_dict=flags_dict,
                                              llm_gen_dir=llm_gen_dir)
        gen_plan_args['log_dir'] = agent_param['log_dir']

    agent_param['plan_gen_args'] = gen_plan_args

    # add param for validation based on llm-generated plans
    if 'plan_based_val_args' in agent_param.keys():
        plan_based_val_args = create_agent_strat_val_planbased(
            agent_config=agent_param['plan_based_val_args'],
            task_dict=validation_task_dict,
            flags_dict=flags_dict,
            llm_gen_dir=llm_gen_dir,
            res_dir=res_dir
        )
        plan_based_val_args['log_dir'] = agent_param['log_dir']
        plan_based_val_args['name'] = agent_param['plan_based_val_args']['name']
        agent_param['plan_based_val_args'] = plan_based_val_args

    return agent_param


# Can be used on validation and on prompt tasks so this is fine
def create_agent_strat_val_planbased(agent_config: dict,
                                     task_dict: dict,
                                     flags_dict: dict,
                                     llm_gen_dir: Path,
                                     res_dir: Path):
    assert task_dict is not None

    agent_param = deepcopy(agent_config['args'])
    agent_param['task_dict'] = task_dict
    agent_param['flags'] = flags_dict
    agent_param['llm_gen_dir'] = llm_gen_dir
    agent_param['res_dir'] = res_dir
    llm_conf_name = agent_config['llm_conf_name']
    # This agent does not necessarily use its own LLM and hence does not really need a system prompt but could have ...
    if not 'system_prompt' in agent_config.keys():
        agent_param['llm_model'] = create_and_init_model(
            llm_config=FLAGS.llm_configs[llm_conf_name],
            system_prompt_path='agent_strategy_val_planbased/system_prompt.txt',
        flags=flags_dict)
    else:
        agent_param['llm_model'] = create_and_init_model(
            llm_config=FLAGS.llm_configs[llm_conf_name],
            system_prompt_path=agent_config['system_prompt'],
        flags=flags_dict)

    # Check if generation of plans is done by separate agent
    if 'plan_gen_prompt_file' in agent_param.keys():
        gen_plan_args = agent_param.pop('plan_gen_prompt_file')
    else:
        assert agent_config['args']['plan_gen_args'] is not None
        gen_plan_args = create_agent_plan_gen(agent_config['args']['plan_gen_args'],
                                              flags_dict=flags_dict,
                                              llm_gen_dir=llm_gen_dir)

    agent_param['plan_gen_args'] = gen_plan_args

    return agent_param


def create_agent_plan_gen(agent_config: dict,
                          flags_dict: dict,
                          llm_gen_dir: Path):

    agent_param = deepcopy(agent_config['args'])
    agent_param['name'] = agent_config['name']
    agent_param['flags'] = flags_dict
    agent_param['llm_gen_dir'] = llm_gen_dir
    llm_conf_name = agent_config['llm_conf_name']
    agent_param['llm_model'] = create_and_init_model(llm_config=FLAGS.llm_configs[llm_conf_name],
                                                     system_prompt_path=agent_config['system_prompt'],
                                                     flags=flags_dict)
    return agent_param


def create_agent_refinement(agent_config: dict,
                            flags_dict: dict,
                            llm_gen_dir: Path,
                            plan_gen_config: Union[None, dict] = None):
    """
    For creating an agent that is responsible for refining a strategy
    :param agent_config:
    :param flags_dict:
    :param llm_gen_dir:
    :param plan_gen_config:
    :return:
    """
    # all param but not task_dict
    agent_param = deepcopy(agent_config['args'])
    agent_param['flags'] = flags_dict
    agent_param['llm_gen_dir'] = llm_gen_dir
    llm_conf_name = agent_config['llm_conf_name']
    agent_param['llm_model'] = create_and_init_model(
        llm_config=FLAGS.llm_configs[llm_conf_name],
        system_prompt_path=agent_config['system_prompt'],
        flags=flags_dict
    )

    # Check if generation of plans is done by separate agent
    if 'plan_gen_prompt_file' in agent_param.keys():
        gen_plan_args = agent_param.pop('plan_gen_prompt_file')
        agent_param['plan_gen_args'] = gen_plan_args
    elif agent_config['name'] == 'AgentStrategyRefinePlanFeedback':
        # then no additional parameters needed
        pass
    else:
        assert plan_gen_config is not None
        gen_plan_args = create_agent_plan_gen(plan_gen_config,
                                              flags_dict=flags_dict,
                                              llm_gen_dir=llm_gen_dir)
        agent_param['plan_gen_args'] = gen_plan_args
        gen_plan_args['log_dir'] = agent_param['log_dir']

    return agent_param


def create_agent_codegen(agent_config: dict,
                         debug_task_names: list,
                         validation_task_dict: dict,
                         llm_gen_dir: Path,
                         res_dir: Path,
                         flags_dict: dict) -> dict:
    agent_param = deepcopy(agent_config['args'])
    agent_param['validation_task_dict'] = validation_task_dict
    agent_param['debug_task_names'] = debug_task_names
    agent_param['flags'] = flags_dict
    llm_conf_name = agent_config['llm_conf_name']
    agent_param['llm_model'] = create_and_init_model(llm_config=FLAGS.llm_configs[llm_conf_name],
                                                     system_prompt_path=agent_config['system_prompt'],
                                                     flags=flags_dict)

    agent_param['validator_param'] = FLAGS.validator['args']
    agent_param['validator_param']['name'] = FLAGS.validator['name']
    agent_param['validator_param']['flags'] = FLAGS

    if 'plan_based_val_args' in agent_param.keys():
        plan_based_val_args = create_agent_strat_val_planbased(
            agent_config=agent_param['plan_based_val_args'],
            task_dict=validation_task_dict,
            flags_dict=flags_dict,
            llm_gen_dir=llm_gen_dir,
            res_dir=res_dir
        )
        plan_based_val_args['log_dir'] = agent_param['log_dir']
        plan_based_val_args['name'] = agent_param['plan_based_val_args']['name']
        agent_param['plan_based_val_args'] = plan_based_val_args

    return agent_param

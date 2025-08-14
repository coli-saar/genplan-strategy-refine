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


# TODO: strategy gen with debug requires two LLMs
# Should work now but need to modify the config files to match this
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

    # TODO get information from name?
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


def create_agent_prob_suggestions(agent_config: dict,
                                  prompt_task_dict: dict,
                                  validation_task_dict,
                                  flags_dict: dict,
                                  benchmark_dir: Path,
                                  llm_gen_dir: Path,
                                  res_dir: Path,
                                  plan_gen_config: Union[None, dict] = None) -> dict:

    agent_param = deepcopy(agent_config['args'])
    agent_param['prompt_task_dict'] = prompt_task_dict
    agent_param['flags'] = flags_dict
    agent_param['llm_gen_dir'] = llm_gen_dir

    # Only the agents which are actually generating new problem instances have the LLM parameter
    if 'llm_conf_name' in agent_config.keys():
        llm_conf_name = agent_config['llm_conf_name']
        agent_param['llm_model'] = create_and_init_model(
            llm_config=FLAGS.llm_configs[llm_conf_name],
            system_prompt_path=agent_config['system_prompt'],
            flags=flags_dict
        )

        # example for valid pddl
        domain_file_path = benchmark_dir / 'domain.pddl'
        example_problem = benchmark_dir / 'pddl2nl_example' / 'instance-ex.pddl'
        example_task = Task(domain_file_path=str(domain_file_path),
                            problem_file_path=str(example_problem))
        example_task_str = example_task.problem_str

        # problem to pddl model (doesn't need log dir because not a separate agent)
        nl2pddl_model_conf_name = agent_param['problem_to_pddl_param'].pop('llm_conf_name')
        nl2pddl_model_conf = FLAGS.llm_configs[nl2pddl_model_conf_name]
        nl2pddl_system_prompt = agent_param['problem_to_pddl_param'].pop('system_prompt')
        agent_param['problem_to_pddl_param']['llm_model'] = create_and_init_model(
            llm_config=nl2pddl_model_conf,
            system_prompt_path=nl2pddl_system_prompt,
            flags=flags_dict
        )
        agent_param['problem_to_pddl_param']['example_problem_str'] = example_task_str

    # add param for refinement agent
    refinement_agent_type = agent_param['refine_agent']['name']
    if refinement_agent_type == 'AgentStrategyGen':
        agent_param['refine_agent_param'] = create_agent_strategy_gen(
            agent_param['refine_agent'],
            prompt_task_dict=prompt_task_dict,
            validation_task_dict=None,
            flags_dict=flags_dict,
            llm_gen_dir=llm_gen_dir,
            res_dir=res_dir
        )
    elif refinement_agent_type == 'AgentStrategyRefinement':
        agent_param['refine_agent_param'] = create_agent_refinement(
            agent_param['refine_agent'],
            flags_dict=flags_dict,
            plan_gen_config=plan_gen_config,
            llm_gen_dir=llm_gen_dir
        )
    else:
        raise ValueError

    agent_param.pop('refine_agent')
    agent_param['refine_agent_param']['log_dir'] = agent_param['log_dir']

    # add param for description agent
    if 'descript_agent' in agent_param.keys():
        agent_param['descript_agent_param'] = create_agent_description_gen(
            agent_config=agent_param['descript_agent'],
            task_dict=prompt_task_dict,
            flags_dict=flags_dict
        )
        agent_param.pop('descript_agent')
        agent_param['descript_agent_param']['log_dir'] = agent_param['log_dir']
    else:
        # (has log dir because from FLAGS directly)
        agent_param['descript_agent_param'] = create_agent_description_gen(
            agent_config=FLAGS.agents['description_generation'],
            task_dict=prompt_task_dict,
            flags_dict=flags_dict
        )

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

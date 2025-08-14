from typing import Union, List
from copy import deepcopy
from jinja2 import Template
from llm_models.create_llm import create_llm_model
from llm_models.llm_base_class import LLMModel
from utils.utils import create_prompt_template


def _create_and_init_model(config_dict: dict, flags: dict) -> LLMModel:

    model_params = config_dict['llm_parameter']
    assert model_params['max_history'] is None
    model_name = model_params.pop('model_type')
    llm_model = create_llm_model(model_type=model_name,
                                 model_param=model_params)
    system_prompt_path = config_dict.get('system_prompt', None)
    if system_prompt_path:
        system_prompt_template = create_prompt_template(prompt_file=system_prompt_path, flags=flags)
        system_prompt = system_prompt_template.render()
    else:
        system_prompt = ''
    llm_model.init_model(init_prompt=system_prompt, examples=[])

    return llm_model


def create_and_init_model(llm_config: dict,
                          flags: dict,
                          system_prompt_path: Union[str, None]=None
                          ) -> LLMModel:

    config = deepcopy(llm_config)
    if system_prompt_path is not None:
        config['system_prompt'] = system_prompt_path
    llm_model = _create_and_init_model(config_dict=config, flags=flags)

    return llm_model


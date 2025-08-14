import os
from pathlib import Path
from typing import Union

PROJ_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJ_DIR / Path('llm_caches')


def get_cache_dir(model_subdir_name: Union[str, None], exp_subdir_name: Union[str, None]):
    """

    :param model_subdir_name:
    :param exp_subdir_name:
    :return:
    """
    if model_subdir_name is None:
        assert exp_subdir_name is None
        cache_dir = None
    else:
        cache_dir = CACHE_DIR / Path(model_subdir_name)
        if exp_subdir_name is not None:
            cache_dir = cache_dir / Path(exp_subdir_name)
    if cache_dir is not None:
        Path(cache_dir).mkdir(exist_ok=True, parents=True)

    return cache_dir


def get_llm_type(llm_name: str):
    if llm_name.startswith('gpt-4') or llm_name.startswith('gpt-'):
        llm_type = 'openai_chat'
    elif 'llama' in llm_name:
        llm_type = 'llama_hf'
    else:
        raise ValueError(f'The model {llm_name} could not be mapped to a model type / class. Please specify the type of the model as "llm-type", e.g. openai_chat for chat models, openai_comp for text completion models.')
    return llm_type

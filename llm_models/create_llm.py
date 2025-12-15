import copy
from typing import Union

from llm_models.utils.helpers import get_cache_dir
from llm_models.llm_base_class import LLMModel
from llm_models.llm_classes.openai_models import OpenAIChatModel
from llm_models.llm_classes.openai_models_sampling import OpenAIChatModelSampling
from llm_models.llm_classes.ollama_models import OllamaModel
from llm_models.llm_classes.sglang_models import SGLangModel, SGLangThinking
from llm_models.llm_classes.deepseek import DeepSeekReasonModel

def create_llm_model(model_type: str,
                     model_param: dict,
                     cache_sub_dir: Union[None, str] = 'default') -> LLMModel:
    """
    Creates different kinds of llm models of the subclasses of LLMModel (in llm_models.py)
    :param model_type: the name of the model to use
    :param model_param: dictionary with the parameters for the model
                        required keys: 'model_path', 'max_tokens', 'temperature', 'max_history'
                        if vicuna model additionally: 'cuda_n', 'load_method'
                        optional_keys: "caching", "seed"
    :param cache_sub_dir:
    :return:
    """
    model_path = model_param['model_path']

    model_input_param = copy.deepcopy(model_param)

    if cache_sub_dir is None:
        cache_dir = get_cache_dir(None, None)
    else:
        cache_dir = get_cache_dir(model_subdir_name=f'{model_type}_{model_path}',
                                  exp_subdir_name=cache_sub_dir)

    model_input_param['cache_directory'] = cache_dir

    if model_type == 'openai_chat':
        if 'n' in model_input_param.keys():
            model = OpenAIChatModelSampling(**model_input_param)
        else:
            model = OpenAIChatModel(**model_input_param)

    elif model_type == 'ollama':
        model = OllamaModel(**model_input_param)
    elif model_type == 'sglang':
        model = SGLangModel(**model_input_param)
    elif model_type == 'sglang_think':
        model = SGLangThinking(**model_input_param)
    elif model_type == 'deepseek-reasoner':
        model = DeepSeekReasonModel(**model_input_param)
    else:
        raise NotImplementedError

    return model




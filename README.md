# genplan-strategy-refine

For generating generalized plans: see the scripts in the sh_scripts_generation folder
For evaluating the plans: see the scripts in the sh_scripts_evaluation folder


**Switching to other model**:
- need to adapt the configs in the configs/all_domains folder by exchanging the llm_model config part
- need to extend the requirements to include e.g. transformers 
- need to add the code required for running HF models to the llm_models folder 


Need a file 'set_env.py' in the utils folder 
```
import os


def set_env_vars():
    os.environ['OPENAI_API_KEY'] = 
```

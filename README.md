# genplan-strategy-refine

Code and data for the paper [Improved Generalized Planning with LLMs through Strategy Refinement and Reflection](https://www.arxiv.org/abs/2508.13876), to be presented at the [Workshop on Planning in the Era of LLMs](https://llmforplanning.github.io/) @ ICAPS25

### Prerequisites
Current version of the code is set up for OpenAI models and tested with OpenAI non-reasoning models. We will upload the files for using open-source models soon. 

Create a file 'set_env.py' in the utils folder 
```
import os


def set_env_vars():
    os.environ['OPENAI_API_KEY'] = 
```

Code tested with Python 3.10 on Linux

Requires a compiled version of the [plan validator VAL](https://github.com/KCL-Planning/VAL) and [fastdownward](https://www.fast-downward.org/latest/).<br>
Set the `FASTDOWNWARD` and `VAL` path variables in the ./utils/paths.py script accordingly (line 7-8)

## How to run

### Generating Generalized Plans
`python run_pipeline.py --env [dataset]-[domain] --config [config_path]`
-  env: dataset name (i.e. subfolder of the ./data directory) and name of the domain, separated by '-', e.g. 'silver-ferry'
- config_path: path to the configuration .json file

See also scripts in the sh_scripts_generation folder.<br>
To generate additional sh_scripts run: `python create_sh_scripts.py -d [dataset]-[domain] --conf [config_dir] --sh [sh_dir]`<br>
This will generate a file in the sh_dir directory with the commands for running the pipeline with all configuration files in the config_dir. 

**Configuration Files**<br>

- baseline.json: our baseline

Full framework:
- full_3_6.json: F3-6
- full_5_3.json: F5-4
- 
Ablations
- full_no_code_sr.json: no self-reflection during code debugging
- ful_no_multicode.json: only one initial program version
- full_no_strat_debug.json: no debugging on strategy 

### Evaluating Generalized Plans

`python run_evaluation.py --env [dataset]-[domain] --eval_env [dataset_ev]-[domain_ev] --config [config_path] --out [output file]`
- env: dataset name, needs to match the dataset name from generating the generalized plans
- eval_env: dataset name for the data to use for evaluation
- config_path: path to the evaluation configuration .json file
- output file: the name of the file where the results are saved; output directory is defined in the config file

**Note on env vs. eval_env**
- env determines the name of the subfolders in the output directory from which the generalized plans are read
- eval_env determines in which subfolders of the ./data directory the evaluation data can be found
- the domains should be the same but can come from different datasets, e.g. `--env silver-ferry --eval-env additional-ferry`

See also the scripts in the sh_scripts_evaluation folder

**Configuration Files**<br>

See ./eval_configs/eval_config_all.json
- experiments_output_folder: folder with the output files of the program generation, i.e. with the programs to test
- experiments_results_folder: folder with the corresponding result files from running the pipeline
- experiments_names: names of the approaches, i.e. sub-folders in the outputs and results folder that should be considered
- split_file: name of the .json file in the data directory which specifies which tasks to run the evaluation on, 
- eval_split_name: the key to extract from the json file
- val_split_file: the name of the .json file that specified the data splits for generating the programs


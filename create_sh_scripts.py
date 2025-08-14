import os
from pathlib import Path



if __name__ == '__main__':

    #domains = ['additional-ferry_test']
    domains = None

    if domains is None:
        domains = []
        add_domains = list(os.listdir('./data/additional'))
        silver_domains = (list(os.listdir('./data/silver')))
        for dom in add_domains:
            domains.append(f'additional-{dom}')
            domains.append(f'silver-{dom}')


    for domain in domains:

        # Sh scripts for generation
        config_dir = './configs/all_domains'        # TODO: change if running different config files
        sh_script_folder = './sh_scripts_generation'
        Path(sh_script_folder).mkdir(exist_ok=True)

        with open(os.path.join(sh_script_folder, f'{domain}_generation.sh'), 'w') as f:
            for config in os.listdir(config_dir):
                config_path = os.path.join(config_dir, config)
                f.write(f'python run_pipeline.py --env {domain} --config {config_path}\n')

        # Sh scripts for evaluation
        config_dir = './eval_configs'
        config_path = os.path.join(config_dir, 'eval_config_all.json')
        sh_script_folder = './sh_scripts_evaluation'
        Path(sh_script_folder).mkdir(exist_ok=True)

        with open(os.path.join(sh_script_folder, f'{domain}_evaluation.sh'), 'w') as f:
            f.write(f'python run_evaluation.py --env {domain} --eval_env {domain} --config {config_path} --out eval_results_data.json')


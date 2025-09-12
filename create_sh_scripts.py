import os
from pathlib import Path
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d', type=str, required=False, default='', help='dataset and domain name')
    parser.add_argument('--conf', required=False, default='./configs/all_domains', help='path to the config directory')
    parser.add_argument('--sh', required=False, default='./sh_scripts_generation', help='path to the folder for the sh scripts')
    args = parser.parse_args()

    domains = args.d
    if domains == '':
        domains = None      # for running all domains from paper
    config_dir = args.conf
    sh_script_folder = args.sh

    # domains = ['additional-ferry']     # for running on a specific dataset
    # config_dir = './configs/all_domains'  # path to the config directory
    # sh_script_folder = './sh_scripts_generation'    # path to the folder for the sh scripts


    if domains is None:
        domains = []
        add_domains = list(os.listdir('./data/additional'))
        silver_domains = (list(os.listdir('./data/silver')))
        for dom in add_domains:
            domains.append(f'additional-{dom}')
            domains.append(f'silver-{dom}')

    for domain in domains:

        # Sh scripts for generation

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


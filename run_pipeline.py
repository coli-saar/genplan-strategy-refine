import os
import sys
import json
import time
import traceback
from tqdm import tqdm
from pathlib import Path
from utils.paths import get_timestamp_for_log
from utils.flags_scripts import parse_flags_main_pipeline, FLAGS
from agentic_genplan.pipeline import run_pipeline
from utils.paths import create_log_dirname, get_res_file_name
from utils.helper import convert_path2str, create_ordered_results_data


if __name__ == '__main__':

    script_start = time.time()
    str_args = " ".join(sys.argv)
    # Parse command-line flags
    parse_flags_main_pipeline()
    FLAGS.__dict__.update({'time_stamp': get_timestamp_for_log()})

    for seed_value in tqdm(FLAGS.seeds):
        FLAGS.__dict__.update({'seed': seed_value})
        try:
            run_pipeline(seed_value=seed_value,
                         p_version=FLAGS.p_version)
        except Exception as e:
            error_message = str(e)
            experiment_id = FLAGS.experiment_id
            time_stamped_folder = create_log_dirname(flags=FLAGS)

            # Output directory for the LLM outputs
            domain_out_dir = os.path.join(experiment_id, FLAGS.env)
            seed_out_dir = Path(FLAGS.save_path) / domain_out_dir / time_stamped_folder / f'seed_{seed_value}'
            print(error_message)
            with open(os.path.join(seed_out_dir, "error_log.txt"), "a") as f:
                f.write("=== Exception caught ===\n")
                f.write(f"Type: {type(e).__name__}\n")
                f.write(f"Message: {str(e)}\n")
                f.write("Stacktrace:\n")
                f.write(traceback.format_exc())
                f.write("\n\n")

            """out_data = {"Error": {"type": type(e).__name__,
                                 "message": str(e)},
                       "config": FLAGS.__dict__}

            out_data = convert_path2str(out_data)
            out_data = create_ordered_results_data(out_data=out_data)
            res_out_dir = Path(FLAGS.results_dir) / domain_out_dir / time_stamped_folder / f'seed_{seed_value}'
            res_out_file = res_out_dir / f'{get_res_file_name(flags=FLAGS, version="final")}.json'

            with open(res_out_file, 'w') as f:
                json.dump(out_data, f, indent=2)"""

    script_time = time.time() - script_start
    print(f"\n\nMain script terminated in {script_time:.5f} seconds")


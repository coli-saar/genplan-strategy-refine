import sys
import time
from tqdm import tqdm
from utils.paths import get_timestamp_for_log
from utils.flags_scripts import parse_flags_main_pipeline, FLAGS
from agentic_genplan.pipeline import run_pipeline


if __name__ == '__main__':

    script_start = time.time()
    str_args = " ".join(sys.argv)
    # Parse command-line flags
    parse_flags_main_pipeline()
    FLAGS.__dict__.update({'time_stamp': get_timestamp_for_log()})

    for seed_value in tqdm(FLAGS.seeds):
        FLAGS.__dict__.update({'seed': seed_value})

        run_pipeline(seed_value=seed_value,
                     p_version=FLAGS.p_version)

    script_time = time.time() - script_start
    print(f"\n\nMain script terminated in {script_time:.5f} seconds")


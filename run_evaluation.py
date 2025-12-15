import sys
import time
from pathlib import Path
from utils.flags_scripts import parse_flags_eval_pipeline, FLAGS
from evaluation.evaluation_pipeline import run_all_evaluations_table, run_all_evaluations_json
from evaluation.evaluation_pipeline_full import run_all_evaluations_all_codes_json


if __name__ == '__main__':
    script_start = time.time()
    str_args = " ".join(sys.argv)
    # Parse command-line flags
    parse_flags_eval_pipeline()

    benchmark_name, domain_name = FLAGS.env.split('-')
    Path(f'./data_analysis_results/runs_with_empty_outputs').mkdir(exist_ok=True)

    if FLAGS.eval_all:
        run_all_evaluations_all_codes_json(benchmark_name=benchmark_name,
                                           domain_name=domain_name,
                                           output_file=FLAGS.out,
                                           flags=FLAGS)
    elif FLAGS.version == 'table':
        run_all_evaluations_table(benchmark_name=benchmark_name,
                                  domain_name=domain_name,
                                  output_file=FLAGS.out,
                                  flags=FLAGS,
                                  overwrite=FLAGS.overwrite)
    elif FLAGS.version == 'json':
        run_all_evaluations_json(benchmark_name=benchmark_name,
                                 domain_name=domain_name,
                                 output_file=FLAGS.out,
                                 flags=FLAGS)

    script_time = time.time() - script_start
    print(f"\n\nEvaluation terminated in {script_time:.5f} seconds")


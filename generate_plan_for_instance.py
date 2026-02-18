import argparse

from plan_generation.generate_plan import generate_plan


parser = argparse.ArgumentParser(
    description="Run a generated planning script on a problem instance"
)

parser.add_argument("domain_file", help="Path to the domain file for the relevant domain")
parser.add_argument("instance_file", help="Path to the problem instance file")
parser.add_argument("code_file", help="Path to the file with the generated python program")
parser.add_argument("output_path", help="Filepath to store the plan at")
parser.add_argument("log_path", help="Filepath to store the log notes at")
parser.add_argument("-t", "--timeout", type=int, default=45, help="Set a specific timeout, default is 45 (measured in seconds)")
parser.add_argument("-p", "--print", action="store_true", help="Set this flag to print the log file contents to console as well")

args = parser.parse_args()

# Run plan generation
generate_plan(
    args.domain_file,
    args.instance_file,
    args.code_file,
    args.output_path,
    args.log_path,
    args.timeout,
    args.print
)

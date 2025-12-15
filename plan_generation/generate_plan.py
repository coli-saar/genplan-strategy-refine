import time

from agents.agent_code_gen import postprocess_response
from utils.tasks import TaskSimple

from plan_generation.code_executor import GeneralizedPlan, CodeExecutor


def save_plan(plan, output_file):
    """Save generated plan to file, including the cost."""
    with open(output_file, "w", encoding="utf-8") as file_out:
        for line in plan:
            file_out.write(f"{line}\n")
        # State cost
        file_out.write(f"; cost = {len(plan)} (unit cost)\n")


def generate_plan(domain_file,
                  instance_file,
                  code_file_path,
                  output_file,
                  log_file,
                  timeout,
                  print_to_console):
    """Execute generated python code to generate a plan for a given instance."""

    # Get python code from file
    with open(code_file_path, 'r') as f:
        content = f.read()
    python_code = postprocess_response(model_response=content)
    if not python_code:
        print("Error: Python code empty")
        return

    # Get task
    task = TaskSimple(domain_file, instance_file, print_paths=False)

    # Get code executor
    generalized_plan = GeneralizedPlan(python_code)
    code_executor = CodeExecutor(gen_plan=generalized_plan, timeout=timeout)

    # Start time tracking
    generation_start = time.time()

    # Run planning program
    execution_results = code_executor.run_genplan_on_task(
        task=task,
        objects=task.objects,
        init_state=task.init,
        goal_state=task.goal
    )
    plan = execution_results["plan"]
    planning_runtime = execution_results["run_time"]
    error_msg = execution_results.get("error_msg", "")
    timeout_happened = execution_results.get("timeout_reached", False)

    # Save plan
    if plan:
        save_plan(plan, output_file)

    # Stop time tracking
    generation_end = time.time()
    overall_runtime = generation_end - generation_start

    # Print to console
    if print_to_console:
        print("-"*50)
        print(f"Plan generation for {instance_file}")
        print(f"Domain: {domain_file}")
        print(f"Python code: {code_file_path}\n")
        print(f"Objects: {task.objects}")
        print(f"Init: {task.init}")
        print(f"Goal: {task.goal}\n")
        if not error_msg:
            if not execution_results["plan"]:
                print(f"-> Generated plan is empty")
            else:
                print(f"-> Plan generated without error")
                print(f"Plan saved to {output_file}")
        else:
            print(f"-> Plan generation threw error.")
            for line in error_msg:
                print(line)
            if timeout_happened:
                print("-> This error is the the result of a timeout, usually the result of an infinite loop.")
            print()

        print(f"Program ran in {overall_runtime}")
        print(f"Planning ran in {planning_runtime}")
        print("-"*50)

    # Write log
    with open(log_file, "w", encoding="utf-8") as file_out:
        file_out.write(f"Plan generation for {instance_file}\n")
        file_out.write(f"Domain: {domain_file}\n")
        file_out.write(f"Python code: {code_file_path}\n\n")
        file_out.write(f"Objects: {task.objects}\n")
        file_out.write(f"Init: {task.init}\n")
        file_out.write(f"Goal: {task.goal}\n\n")
        if not error_msg:
            if not execution_results["plan"]:
                # Empty plan
                file_out.write(f"Generated plan is empty\n\n")
            else:
                # Plan generated
                file_out.write(f"Plan generated without error\n")
                file_out.write(f"Plan saved to {output_file}\n\n")
        else:
            # There was an error
            file_out.write(f"Plan generation threw error.\n")
            for line in error_msg:
                file_out.write(f"    {line}\n")
            if timeout_happened:
                file_out.write("-> This error is the the result of a timeout, usually the result of an infinite loop.\n")
            file_out.write("\n")

        # Runtimes
        file_out.write(f"Program ran in {overall_runtime}\n")
        file_out.write(f"Planning ran in {planning_runtime}\n")


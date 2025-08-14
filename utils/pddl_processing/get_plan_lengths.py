import os
from utils.utils import run_symbolic_planner_on_file


def get_plan_len(domain_dir,
                 domain_path,
                 instance_path,
                 plan_path,
                 instance_name,
                 run_planner: bool = True,
                 optimal: bool = False):

    if not optimal:
        unsolved_inst_file = os.path.join(domain_dir, 'unsolved_inst_gbfs.txt')
    else:
        unsolved_inst_file = os.path.join(domain_dir, 'unsolved_inst_opt.txt')
    unsolved_insts = []
    if os.path.exists(unsolved_inst_file):
        with open(unsolved_inst_file, 'r') as f:
            for line in f.readlines():
                inst = line.strip()
                if inst:
                    unsolved_insts.append(inst)

    if not os.path.exists(plan_path) and not instance_name in unsolved_insts:
        if not run_planner:
            unsolved_insts.append(instance_name)
        else:
            print(f'Running planner on {instance_path}')
            failed, _ = run_symbolic_planner_on_file(domain_file=domain_path,
                                                     instance_file=instance_path,
                                                     plan_file=plan_path,
                                                     optimal=False)
            if failed:
                unsolved_insts.append(instance_name)

            with open(unsolved_inst_file, 'w') as f:
                for inst in unsolved_insts:
                    f.write(f'{inst}\n')

    if os.path.exists(plan_path):
        plan = []
        cost = None
        with open(plan_path, 'r') as f:
            for line in f.readlines():
                l = line.strip()
                if not l.startswith(';'):
                    plan.append(l)
                elif l.startswith('; cost'):
                    cost_str = l.replace('; cost = ', '').replace(' (unit cost)', '')
                    cost = int(cost_str)

        if cost is not None:
            assert cost == len(plan)
        return cost

    else:
        return None


#!/usr/bin/env python3
import os
import sys
import random
import subprocess
import time
import pandas as pd
from tqdm import tqdm


# --- DIMACS parser ---
def parse_dimacs(fn):
    clauses, n = [], 0
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                n = int(line.split()[2])
            else:
                lits = list(map(int, line.split()))
                if lits[-1] == 0: lits = lits[:-1]
                clauses.append(lits)
    return n, clauses


# --- CNF Generation Functions ---
def generate_random_3sat_instance(num_vars, num_clauses, filename):
    """Generates a random 3-SAT instance and saves it in DIMACS format."""
    clauses = []
    for _ in range(num_clauses):
        clause = set()
        while len(clause) < 3:
            literal = random.randint(1, num_vars) * random.choice([1, -1])
            clause.add(literal)
        clauses.append(list(clause))

    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def generate_pigeonhole_instance(n, filename):
    """Generates a pigeonhole problem instance and saves it in DIMACS format.
    The pigeonhole principle with n+1 pigeons in n holes.
    """
    pigeons = n + 1  # n+1 pigeons
    holes = n  # n holes

    variables = {}  # Map (pigeon, hole) pairs to variable numbers
    var_count = 1  # Counter for variable numbering

    # Assign variable numbers to (pigeon, hole) pairs
    for p in range(1, pigeons + 1):
        for h in range(1, holes + 1):
            variables[(p, h)] = var_count
            var_count += 1

    clauses = []

    # Each pigeon must be in at least one hole
    for p in range(1, pigeons + 1):
        clauses.append([variables[(p, h)] for h in range(1, holes + 1)])

    # No hole can contain more than one pigeon
    for h in range(1, holes + 1):
        for p1 in range(1, pigeons + 1):
            for p2 in range(p1 + 1, pigeons + 1):
                clauses.append([-variables[(p1, h)], -variables[(p2, h)]])

    with open(filename, 'w') as f:
        f.write(f"p cnf {var_count - 1} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def create_sample_cnf_files(directory):
    """Generates one CNF file for each variable count."""
    os.makedirs(directory, exist_ok=True)

    # List to keep track of generated files
    generated_files = []

    # Generate one random instance for each variable count
    var_counts = [10, 15, 20, 35, 50]
    for var_count in var_counts:
        # Use 4.3 as ratio (around phase transition)
        num_clauses = int(var_count * 4.3)
        filename = os.path.join(directory, f"random_{var_count}vars.cnf")
        generate_random_3sat_instance(var_count, num_clauses, filename)
        generated_files.append(filename)
        print(f"Generated {filename}")

    # Generate 1 pigeonhole instance
    n = 5  # Will generate an instance with 6 pigeons in 5 holes
    filename = os.path.join(directory, f"pigeonhole.cnf")
    generate_pigeonhole_instance(n, filename)
    generated_files.append(filename)
    print(f"Generated {filename}")

    return generated_files


# --- Solver runner ---
def run_solver(instances, solver_script, algo, heur=None, timeout_map=None, log_file=None, progress=None):
    if timeout_map is None:
        timeout_map = {'resolution': 300, 'dp': 180, 'dpll': 120}

    timeout = timeout_map.get(algo, 120)  # Default timeout

    # Classify instances by type and size
    small_instances = [i for i in instances if ('10vars' in i or '15vars' in i)]
    medium_instances = [i for i in instances if ('20vars' in i)]
    large_instances = [i for i in instances if ('35vars' in i or '50vars' in i)]
    pigeonhole = [i for i in instances if 'pigeonhole' in i]

    # For resolution and dp, maybe skip large and pigeonhole instances
    if algo in ['resolution', 'dp']:
        if large_instances and input(f"Run {algo} on large instances? This could take a while (y/n): ").lower() != 'y':
            instances = [i for i in instances if i not in large_instances]
            print(f"Skipping large instances for {algo}")

        if pigeonhole and input(f"Run {algo} on pigeonhole instances? These are very hard (y/n): ").lower() != 'y':
            instances = [i for i in instances if i not in pigeonhole]
            print(f"Skipping pigeonhole instances for {algo}")

    records = []
    for i, inst in enumerate(instances):
        # Adjust timeout based on instance type
        instance_timeout = timeout
        if 'pigeonhole' in inst:
            instance_timeout = min(timeout * 2, 1800)  # Max 30 minutes for pigeonhole
        elif '50vars' in inst:
            instance_timeout = min(timeout * 1.5, 900)  # Max 15 minutes for 50 var instances

        cmd = [sys.executable, solver_script, '-a', algo]
        if algo == 'dpll' and heur:
            cmd += ['-H', heur]
        cmd.append(inst)

        print(f"Running: {' '.join(cmd)}")
        print(f"Timeout: {instance_timeout}s")

        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=instance_timeout)
            t = time.time() - start
            out = proc.stdout.decode().strip().splitlines()
            err = proc.stderr.decode().strip()

            if proc.returncode != 0 or err:
                res = f"ERROR: {err}" if err else "ERROR: Non-zero return code"
            else:
                res = out[0] if out else 'UNKNOWN'
        except subprocess.TimeoutExpired:
            t, res = instance_timeout, 'TIMEOUT'
        except Exception as e:
            t, res = 0.0, f'ERROR: {e}'

        record = {
            'instance': os.path.basename(inst),
            'algorithm': algo,
            'heuristic': heur or '',
            'result': res,
            'time_sec': round(t, 2)
        }
        records.append(record)

        # Log partial result to CSV if enabled
        if log_file:
            df = pd.DataFrame([record])
            df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

        if progress:
            progress.update(1)

        print(f"Result: {res}, Time: {round(t, 2)}s")

    return pd.DataFrame(records, columns=['instance', 'algorithm', 'heuristic', 'result', 'time_sec'])


# --- Main Benchmark Execution ---
def main():
    # Setup directory and solver script
    bench_dir = 'benchmarks'
    os.makedirs(bench_dir, exist_ok=True)
    solver = 'sat_solver.py'

    # Check if the solver script exists
    if not os.path.exists(solver):
        print(f"Error: Solver script '{solver}' not found. Please ensure it exists in the current directory.")
        sys.exit(1)

    # Get user input for timeout values
    print("\nSet timeout values for each algorithm (in seconds):")
    timeout_resolution = int(input("Resolution timeout [300]: ") or "300")
    timeout_dp = int(input("DP timeout [180]: ") or "180")
    timeout_dpll = int(input("DPLL timeout [120]: ") or "120")

    timeout_map = {
        'resolution': timeout_resolution,
        'dp': timeout_dp,
        'dpll': timeout_dpll
    }

    # Determine if we should generate new CNF files
    generate_new = input("\nGenerate new CNF files? (y/n) [y]: ").strip().lower() != 'n'

    # 1) Generate CNFs if requested or none exist
    instances = sorted(os.path.join(bench_dir, f)
                       for f in os.listdir(bench_dir) if f.endswith('.cnf'))

    if generate_new or not instances:
        print("\nGenerating CNF files...")
        generated_files = create_sample_cnf_files(bench_dir)
        instances = sorted(os.path.join(bench_dir, f)
                           for f in os.listdir(bench_dir) if f.endswith('.cnf'))

    print(f"\nFound {len(instances)} instances:")
    for inst in instances:
        print(f"  - {os.path.basename(inst)}")

    # Ask which algorithms to run
    print("\nSelect which algorithms to run:")
    run_resolution = input("Run Resolution? (y/n) [y]: ").strip().lower() != 'n'
    run_dp = input("Run DP? (y/n) [y]: ").strip().lower() != 'n'
    run_dpll = input("Run DPLL? (y/n) [y]: ").strip().lower() != 'n'

    if run_dpll:
        dpll_heuristics = []
        if input("Run DPLL with 'random' heuristic? (y/n) [y]: ").strip().lower() != 'n':
            dpll_heuristics.append('random')
        if input("Run DPLL with 'jw' heuristic? (y/n) [y]: ").strip().lower() != 'n':
            dpll_heuristics.append('jw')
        if input("Run DPLL with 'mom' heuristic? (y/n) [y]: ").strip().lower() != 'n':
            dpll_heuristics.append('mom')
    else:
        dpll_heuristics = []

    # Count total jobs
    total_jobs = 0
    if run_resolution: total_jobs += len(instances)
    if run_dp: total_jobs += len(instances)
    if run_dpll: total_jobs += len(instances) * len(dpll_heuristics)

    # 3) Run benchmarks
    all_dfs = []
    log_file = "results_log.csv"

    # Delete old log file if it exists
    if os.path.exists(log_file) and input("\nOverwrite existing log file? (y/n) [y]: ").strip().lower() != 'n':
        os.remove(log_file)

    try:
        with tqdm(total=total_jobs, desc="Running benchmarks") as progress:
            # Run resolution solver
            if run_resolution:
                print("\nRunning resolution solver...")
                df = run_solver(instances, solver, 'resolution', timeout_map=timeout_map,
                                log_file=log_file, progress=progress)
                all_dfs.append(df)

            # Run DP solver
            if run_dp:
                print("\nRunning DP solver...")
                df = run_solver(instances, solver, 'dp', timeout_map=timeout_map,
                                log_file=log_file, progress=progress)
                all_dfs.append(df)

            # Run DPLL solver with different heuristics
            if run_dpll:
                for h in dpll_heuristics:
                    print(f"\nRunning DPLL solver with {h} heuristic...")
                    df = run_solver(instances, solver, 'dpll', heur=h, timeout_map=timeout_map,
                                    log_file=log_file, progress=progress)
                    all_dfs.append(df)
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted!")

    # 4) Combine results
    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if df.empty:
        print("Error: No results collected! Check for errors above.")
        return

    # Save full results
    df.to_csv("full_results.csv", index=False)
    print("\nFull results saved to full_results.csv")

    # 5) Print summary
    print("\nFull results:")
    print(df)

    # Compute statistics for SAT and UNSAT instances
    sat_df = df[df['result'] == 'SAT']
    unsat_df = df[df['result'] == 'UNSAT']
    timeout_df = df[df['result'] == 'TIMEOUT']

    # Count instances by type
    total_instances = len(df['instance'].unique())
    sat_instances = len(sat_df['instance'].unique())
    unsat_instances = len(unsat_df['instance'].unique())
    timeout_instances = len(timeout_df['instance'].unique())

    print(f"\nInstance summary:")
    print(f"  Total unique instances: {total_instances}")
    print(f"  SAT instances: {sat_instances}")
    print(f"  UNSAT instances: {unsat_instances}")
    print(f"  Timed out instances: {timeout_instances}")

    # Average times
    avg_time_sat = sat_df.groupby(['algorithm', 'heuristic'])['time_sec'].mean().reset_index()
    avg_time_unsat = unsat_df.groupby(['algorithm', 'heuristic'])['time_sec'].mean().reset_index()

    # Success rates
    success_rate = df.groupby(['algorithm', 'heuristic'])['result'].apply(
        lambda x: ((x == 'SAT') | (x == 'UNSAT')).mean() * 100
    ).reset_index()

    # Create pivot tables for better display
    avg_time_sat_pivot = avg_time_sat.pivot(index='algorithm', columns='heuristic', values='time_sec').fillna(0)
    avg_time_unsat_pivot = avg_time_unsat.pivot(index='algorithm', columns='heuristic', values='time_sec').fillna(0)
    success_rate_pivot = success_rate.pivot(index='algorithm', columns='heuristic', values='result').fillna(0)

    print("\nAverage time (s) for SAT instances:")
    print(avg_time_sat_pivot.round(2))

    print("\nAverage time (s) for UNSAT instances:")
    print(avg_time_unsat_pivot.round(2))

    print("\nSuccess rate (%):")
    print(success_rate_pivot.round(1))

    # 6) Write LaTeX tables
    try:
        avg_time_sat_pivot.round(2).to_latex('avg_time_sat_table.tex',
                                             caption="Average solving time (s) for SAT instances",
                                             label="tab:avg-time-sat")

        avg_time_unsat_pivot.round(2).to_latex('avg_time_unsat_table.tex',
                                               caption="Average solving time (s) for UNSAT instances",
                                               label="tab:avg-time-unsat")

        success_rate_pivot.round(1).to_latex('success_rate_table.tex',
                                             caption="Success rate (\\%)",
                                             label="tab:success-rate")

        print("\nGenerated LaTeX tables: avg_time_sat_table.tex, avg_time_unsat_table.tex, success_rate_table.tex")
    except Exception as e:
        print(f"\nError generating LaTeX tables: {e}")


if __name__ == '__main__':
    main()
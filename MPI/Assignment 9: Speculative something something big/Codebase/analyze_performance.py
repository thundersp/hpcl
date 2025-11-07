#!/usr/bin/env python3

import subprocess
import statistics
import csv
import re


def run_program(executable, runs=5):
    """Run a program multiple times and extract timing information."""
    sequential_times = []
    parallel_times = []

    for i in range(runs):
        print(f"  Run {i+1}/{runs}...", end=" ", flush=True)
        result = subprocess.run(
            [f"./{executable}"], capture_output=True, text=True)
        output = result.stdout

        # Extract timing information based on the program
        if executable == "exploratory_nqueens":
            seq_match = re.search(r'Sequential time: ([\d.]+) ms', output)
            par_match = re.search(r'Parallel time: ([\d.]+) ms', output)
        else:  # speculative_branch
            seq_match = re.search(r'Sequential time: ([\d.]+) ms', output)
            par_match = re.search(r'Parallel time: ([\d.]+) ms', output)

        if seq_match and par_match:
            sequential_times.append(float(seq_match.group(1)))
            parallel_times.append(float(par_match.group(1)))
            print("done")
        else:
            print("failed to parse")

    return sequential_times, parallel_times


def calculate_stats(times):
    """Calculate statistics for a list of times."""
    return {
        'avg': statistics.mean(times),
        'min': min(times),
        'max': max(times),
        'median': statistics.median(times)
    }


def main():
    print("=== HPC Practical No. 9 - Performance Analysis (Python) ===")
    print("Running multiple iterations to collect statistics...")
    print()

    # System information
    cpu_cores = subprocess.run(
        ["nproc"], capture_output=True, text=True).stdout.strip()
    print(f"CPU cores: {cpu_cores}")
    print("Number of test runs: 5")
    print()

    results = {}

    # Test N-Queens (Exploratory Decomposition)
    print("=== Running N-Queens (Exploratory Decomposition) ===")
    seq_times, par_times = run_program("exploratory_nqueens")

    if seq_times and par_times:
        seq_stats = calculate_stats(seq_times)
        par_stats = calculate_stats(par_times)
        speedup = seq_stats['avg'] / par_stats['avg']

        results['N-Queens'] = {
            'sequential': seq_stats,
            'parallel': par_stats,
            'speedup': speedup,
            'wasted_computation': 0
        }

    print()

    # Test Speculative Branch Evaluation
    print("=== Running Branch Evaluation (Speculative Decomposition) ===")
    seq_times, par_times = run_program("speculative_branch")

    if seq_times and par_times:
        seq_stats = calculate_stats(seq_times)
        par_stats = calculate_stats(par_times)
        speedup = seq_stats['avg'] / par_stats['avg']

        results['Branch_Evaluation'] = {
            'sequential': seq_stats,
            'parallel': par_stats,
            'speedup': speedup,
            'wasted_computation': 50
        }

    print()

    # Display results
    print("=== Performance Summary ===")
    print()

    for problem, data in results.items():
        name = problem.replace('_', ' ')
        print(f"{name}:")
        print(
            f"  Sequential - Avg: {data['sequential']['avg']:.2f}ms, Min: {data['sequential']['min']:.2f}ms, Max: {data['sequential']['max']:.2f}ms")
        print(
            f"  Parallel   - Avg: {data['parallel']['avg']:.2f}ms, Min: {data['parallel']['min']:.2f}ms, Max: {data['parallel']['max']:.2f}ms")
        print(f"  Speedup: {data['speedup']:.2f}x")
        print(f"  Wasted Computation: ~{data['wasted_computation']}%")
        print()

    # Generate CSV
    print("=== Generating CSV Results ===")
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Problem', 'Sequential_Avg',
                      'Parallel_Avg', 'Speedup', 'Wasted_Computation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for problem, data in results.items():
            writer.writerow({
                'Problem': problem,
                'Sequential_Avg': f"{data['sequential']['avg']:.2f}",
                'Parallel_Avg': f"{data['parallel']['avg']:.2f}",
                'Speedup': f"{data['speedup']:.2f}",
                'Wasted_Computation': data['wasted_computation']
            })

    print("Results saved to results.csv")
    print()
    print("=== Analysis Complete ===")
    print("Use the results above for your technical report.")


if __name__ == "__main__":
    main()

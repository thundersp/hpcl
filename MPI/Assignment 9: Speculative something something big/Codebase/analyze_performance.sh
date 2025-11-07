#!/bin/bash

# Analysis script for HPC Practical No. 9
# Runs experiments multiple times and collects statistics

echo "=== HPC Practical No. 9 - Performance Analysis ==="
echo "Running multiple iterations to collect statistics..."
echo ""

# Check if programs are compiled
if [ ! -f "exploratory_nqueens" ] || [ ! -f "speculative_branch" ]; then
    echo "Programs not found. Compiling..."
    make all
    echo ""
fi

# Number of test runs
RUNS=5

echo "System Information:"
echo "CPU cores: $(nproc)"
echo "OpenMP threads: ${OMP_NUM_THREADS:-default ($(nproc))}"
echo "Number of test runs: $RUNS"
echo ""

# Function to extract timing from program output
extract_timing() {
    local program=$1
    local type=$2
    local temp_file=$(mktemp)
    
    echo "Running $program ($type timing)..."
    for i in $(seq 1 $RUNS); do
        echo -n "  Run $i/$RUNS... "
        ./$program > $temp_file 2>&1
        
        if [ "$program" = "exploratory_nqueens" ]; then
            if [ "$type" = "sequential" ]; then
                grep "Sequential time:" $temp_file | awk '{print $3}' | sed 's/ms//'
            else
                grep "Parallel time:" $temp_file | awk '{print $3}' | sed 's/ms//'
            fi
        else
            if [ "$type" = "sequential" ]; then
                grep "Sequential time:" $temp_file | awk '{print $3}' | sed 's/ms//'
            else
                grep "Parallel time:" $temp_file | awk '{print $3}' | sed 's/ms//'
            fi
        fi
        echo "done"
    done | tee ${program}_${type}_times.txt
    
    rm $temp_file
}

# Function to calculate statistics
calc_stats() {
    local file=$1
    local total=0
    local count=0
    local min=999999
    local max=0
    
    while read time; do
        total=$(echo "$total + $time" | bc -l)
        count=$((count + 1))
        if (( $(echo "$time < $min" | bc -l) )); then
            min=$time
        fi
        if (( $(echo "$time > $max" | bc -l) )); then
            max=$time
        fi
    done < $file
    
    avg=$(echo "scale=2; $total / $count" | bc -l)
    echo "$avg $min $max"
}

echo "=== Running N-Queens (Exploratory Decomposition) ==="
extract_timing "exploratory_nqueens" "sequential"
extract_timing "exploratory_nqueens" "parallel"

echo ""
echo "=== Running Branch Evaluation (Speculative Decomposition) ==="
extract_timing "speculative_branch" "sequential"
extract_timing "speculative_branch" "parallel"

echo ""
echo "=== Performance Summary ==="
echo ""

# Process N-Queens results
if [ -f "exploratory_nqueens_sequential_times.txt" ]; then
    nq_seq_stats=($(calc_stats "exploratory_nqueens_sequential_times.txt"))
    nq_par_stats=($(calc_stats "exploratory_nqueens_parallel_times.txt"))
    nq_speedup=$(echo "scale=2; ${nq_seq_stats[0]} / ${nq_par_stats[0]}" | bc -l)
    
    echo "N-Queens Problem (Exploratory Decomposition):"
    echo "  Sequential - Avg: ${nq_seq_stats[0]}ms, Min: ${nq_seq_stats[1]}ms, Max: ${nq_seq_stats[2]}ms"
    echo "  Parallel   - Avg: ${nq_par_stats[0]}ms, Min: ${nq_par_stats[1]}ms, Max: ${nq_par_stats[2]}ms"
    echo "  Speedup: ${nq_speedup}x"
    echo "  Wasted Computation: ~0% (exploratory)"
    echo ""
fi

# Process Speculative results
if [ -f "speculative_branch_sequential_times.txt" ]; then
    sb_seq_stats=($(calc_stats "speculative_branch_sequential_times.txt"))
    sb_par_stats=($(calc_stats "speculative_branch_parallel_times.txt"))
    sb_speedup=$(echo "scale=2; ${sb_seq_stats[0]} / ${sb_par_stats[0]}" | bc -l)
    
    echo "Branch Evaluation (Speculative Decomposition):"
    echo "  Sequential - Avg: ${sb_seq_stats[0]}ms, Min: ${sb_seq_stats[1]}ms, Max: ${sb_seq_stats[2]}ms"
    echo "  Parallel   - Avg: ${sb_par_stats[0]}ms, Min: ${sb_par_stats[1]}ms, Max: ${sb_par_stats[2]}ms"
    echo "  Speedup: ${sb_speedup}x"
    echo "  Wasted Computation: ~50% (speculative)"
    echo ""
fi

# Generate CSV for easy analysis
echo "=== Generating CSV Results ==="
echo "Problem,Sequential_Avg,Parallel_Avg,Speedup,Wasted_Computation" > results.csv
if [ -f "exploratory_nqueens_sequential_times.txt" ]; then
    echo "N-Queens,${nq_seq_stats[0]},${nq_par_stats[0]},${nq_speedup},0" >> results.csv
fi
if [ -f "speculative_branch_sequential_times.txt" ]; then
    echo "Branch_Evaluation,${sb_seq_stats[0]},${sb_par_stats[0]},${sb_speedup},50" >> results.csv
fi

echo "Results saved to results.csv"
echo ""

# Clean up temporary files
rm -f *_times.txt

echo "=== Analysis Complete ==="
echo "Use the results above for your technical report."
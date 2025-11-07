#!/bin/bash

echo "=== HPC Practical No. 9 - Compilation and Execution ==="
echo "Compiling programs with OpenMP support..."

gcc -fopenmp -O2 -o exploratory_nqueens exploratory_nqueens.c -lm
if [ $? -eq 0 ]; then
    echo "✓ Exploratory N-Queens compiled successfully"
else
    echo "✗ Failed to compile exploratory_nqueens.c"
    exit 1
fi

gcc -fopenmp -O2 -o speculative_branch speculative_branch.c -lm
if [ $? -eq 0 ]; then
    echo "✓ Speculative Branch Evaluation compiled successfully"
else
    echo "✗ Failed to compile speculative_branch.c"
    exit 1
fi

echo ""
echo "=== Running Programs ==="
echo ""

echo "1. Running Exploratory Decomposition (N-Queens):"
echo "================================================"
./exploratory_nqueens

echo ""
echo "2. Running Speculative Decomposition (Branch Evaluation):"
echo "=========================================================="
./speculative_branch

echo ""
echo "=== Execution Complete ==="
echo "Programs executed successfully!"
echo "Check the output above for timing results and analysis."
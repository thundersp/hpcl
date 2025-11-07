#!/bin/bash
# q1.sh - Run MPI convolution for multiple process counts

rm -f results1.csv
echo "Processes,Time(s)" > results1.csv

# Compile
mpicc -o q1 q1.c

# Run for 1–8 processes
for p in {1..8}
do
  echo "Running with $p processes..."
  mpirun -np $p ./q1 600 3 >> /dev/null
done

echo "Results stored in results1.csv"

#!/bin/bash
# Compile the MPI code
mpicc q2.c -o q2 -lm

# Remove old results
rm -f results2.csv
echo "Processes,Time" > results2.csv

# Run for 1–8 processes
for p in {1..8}
do
  echo "Running with $p processes..."
  mpirun -np $p ./q2 >> results2.csv
done

echo "All runs complete. Results stored in results2.csv"

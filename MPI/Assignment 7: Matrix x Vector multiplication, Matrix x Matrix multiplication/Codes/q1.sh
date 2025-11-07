#!/bin/bash
# Driver: compile, run for P=1..8 and write results1.csv
# Usage: ./run_q1.sh [N]

N=${1:-2000}
OUT=results1.csv

# compile
mpicc q1.c -O2 -o q1
if [ $? -ne 0 ]; then
  echo "Compilation failed"
  exit 1
fi

# header
echo "procs,time_seconds,N" > ${OUT}

for P in {1..8}; do
  echo "Running with ${P} process(es)..."
  # run once and append result (use --oversubscribe if running locally with more procs than cores)
  mpirun -np ${P} ./q1 ${N} >> ${OUT}
done

echo "Results written to ${OUT}"
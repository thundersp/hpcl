#!/bin/bash
# q2.sh - compile and run q2.c for P=1..8 and write results2.csv
# Usage: ./q2.sh [N]

N=${1:-1000}
OUT=results2.csv

mpicc q2.c -O2 -o q2
if [ $? -ne 0 ]; then
  echo "Compilation failed"
  exit 1
fi

echo "procs,time_seconds,N" > ${OUT}

for P in {1..8}; do
  echo "Running with ${P} process(es)..."
  mpirun -np ${P} ./q2 ${N} >> ${OUT}
done

echo "Results written to ${OUT}"
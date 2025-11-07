import sys
import pandas as pd
import matplotlib.pyplot as plt


if len(sys.argv) < 2:
    fname = "results1.csv"
else:
    fname = sys.argv[1]


df = pd.read_csv(fname)
# If multiple runs for same procs exist, take mean
summary = df.groupby("procs", as_index=False)["time_seconds"].mean()


plt.figure(figsize=(8, 5))
plt.plot(summary["procs"], summary["time_seconds"], marker="o")
plt.xlabel("Number of processes")
plt.ylabel("Time (s)")
plt.title("MPI Matrix-Vector Multiply Performance")
plt.grid(True)
plt.xticks(summary["procs"])
plt.tight_layout()
plt.savefig("results1.png")
print("Saved plot to results1.png")
plt.show()

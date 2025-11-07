import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

plt.figure(figsize=(9,6))

for program in df["Program"].unique():
    prog_df = df[df["Program"] == program]
    plt.plot(prog_df["Threads"], prog_df["Time"], marker='o', label=program)

plt.title("Execution Time vs Threads (OpenMP Programs)")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.legend()
plt.grid(True)
plt.xticks(sorted(df["Threads"].unique()))

plt.savefig("openmp_programs_time.png")
plt.show()

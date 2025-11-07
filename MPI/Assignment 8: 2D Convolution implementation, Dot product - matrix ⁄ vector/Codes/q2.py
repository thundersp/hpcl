import pandas as pd
import matplotlib.pyplot as plt

# Read results
data = pd.read_csv("results2.csv")

plt.figure(figsize=(8,5))
plt.plot(data["Processes"], data["Time"], marker='o', label="Execution Time (s)")
plt.title("MPI Matrix-Matrix Multiplication Performance")
plt.xlabel("Number of Processes")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results2.png", dpi=200)
plt.show()

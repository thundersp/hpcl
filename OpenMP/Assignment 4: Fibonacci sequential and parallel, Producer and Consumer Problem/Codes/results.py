import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("results.csv")

plt.figure(figsize=(8,5))

# Plot Producer-Consumer times
pc_df = df[df["Program"] == "ProducerConsumer"]
plt.plot(pc_df["Threads"], pc_df["Time"], marker='o', label="Producer-Consumer")

# Plot Fibonacci times
fib_df = df[df["Program"] == "Fibonacci"]
plt.plot(fib_df["Threads"], fib_df["Time"], marker='s', label="Fibonacci")

plt.title("Execution Time vs Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
plt.legend()
plt.xticks(sorted(df["Threads"].unique()))

plt.savefig("execution_time_graph1.png")
plt.show()

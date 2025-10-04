import pandas as pd
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('results1.csv')
serial_time = data['Time(s)'].iloc[0]
data['Speedup'] = serial_time / data['Time(s)']

# Plot time
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(data['Processes'], data['Time(s)'], marker='o')
plt.title('MPI Convolution Time')
plt.xlabel('Number of Processes')
plt.ylabel('Time (s)')

# Plot speedup
plt.subplot(1,2,2)
plt.plot(data['Processes'], data['Speedup'], marker='o')
plt.title('Speedup (serial_time / time)')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.savefig('results1.png')
plt.tight_layout()
plt.show()

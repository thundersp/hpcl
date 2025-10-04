import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    fname = 'results2.csv'
else:
    fname = sys.argv[1]

df = pd.read_csv(fname)
summary = df.groupby('procs', as_index=False)['time_seconds'].mean()
serial_time = summary.loc[summary['procs'] == 1, 'time_seconds'].values
if len(serial_time) == 0:
    print('No data point for 1 process found; cannot compute speedup. Plotting times only.')
    serial_time = None
else:
    serial_time = float(serial_time[0])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(summary['procs'], summary['time_seconds'], marker='o')
plt.xlabel('Number of processes')
plt.ylabel('Time (s)')
plt.title('MPI Matrix-Matrix Multiply Time')
plt.xticks(summary['procs'])
plt.grid(True)

plt.subplot(1,2,2)
if serial_time:
    summary['speedup'] = serial_time / summary['time_seconds']
    plt.plot(summary['procs'], summary['speedup'], marker='o')
    plt.xlabel('Number of processes')
    plt.ylabel('Speedup')
    plt.title('Speedup (serial_time / time)')
    plt.xticks(summary['procs'])
    plt.grid(True)
else:
    plt.text(0.5,0.5,'No serial run (P=1) found', horizontalalignment='center')

plt.tight_layout()
plt.savefig('results2.png')
print('Saved plot to results2.png')
plt.show()

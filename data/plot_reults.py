import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
data = pd.read_csv('performance_results.csv')

# Convert matrix size to log2 scale for plotting
data['Log2MatrixSize'] = np.log2(data['MatrixSize'])

# Set up the plotting style
sns.set_theme(style="whitegrid")

# Define custom color palette and dash styles
palette = sns.color_palette("hsv", len(data['NumProcs'].unique()))
dashes = [(1, 0), (5, 5)]  # Solid line for no multithreading, dashed line for multithreading

# Plot execution time
plt.figure(figsize=(14, 7))
sns.lineplot(data=data, x='Log2MatrixSize', y='ExecTimeDir', hue='NumProcs', style='Multithreading', palette=palette, dashes=dashes, markers=True)
plt.title('Execution Time for Dirichlet Boundary Condition')
plt.xlabel('Log2(Matrix Size)')
plt.ylabel('Execution Time (seconds)')
plt.legend(title='Number of Processes', loc='upper left')
plt.grid(True)
plt.savefig('exec_time_dirichlet.png')
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(data=data, x='Log2MatrixSize', y='ExecTimeRob', hue='NumProcs', style='Multithreading', palette=palette, dashes=dashes, markers=True)
plt.title('Execution Time for Robin Boundary Condition')
plt.xlabel('Log2(Matrix Size)')
plt.ylabel('Execution Time (seconds)')
plt.legend(title='Number of Processes', loc='upper left')
plt.grid(True)
plt.savefig('exec_time_robin.png')
plt.show()

# For L2 error, select one run with any number of processes and multithreading
l2_error_data = data[(data['NumProcs'] == 2) & (data['Multithreading'] == 'false')]

# Plot L2 error for Dirichlet boundary condition
plt.figure(figsize=(14, 7))
sns.lineplot(data=l2_error_data, x='Log2MatrixSize', y='ErrorDir', hue='MatrixSize', marker='o')
plt.title('L2 Error for Dirichlet Boundary Condition')
plt.xlabel('Log2(Matrix Size)')
plt.ylabel('L2 Error')
plt.legend(title='Matrix Size', loc='upper left')
plt.grid(True)
plt.savefig('l2_error_dirichlet.png')
plt.show()

# Plot L2 error for Robin boundary condition
plt.figure(figsize=(14, 7))
sns.lineplot(data=l2_error_data, x='Log2MatrixSize', y='ErrorRob', hue='MatrixSize', marker='o')
plt.title('L2 Error for Robin Boundary Condition')
plt.xlabel('Log2(Matrix Size)')
plt.ylabel('L2 Error')
plt.legend(title='Matrix Size', loc='upper left')
plt.grid(True)
plt.savefig('l2_error_robin.png')
plt.show()

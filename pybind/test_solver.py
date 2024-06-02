import solver
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tqdm.notebook import tqdm
import sys

def run_solver(n, max_iters, tol):
    solver.mpi_init()
    jacobi_solver = solver.JacobiSolver(n, m, max_iters, tol, solver.myFunc, solver.exactSolution)
    jacobi_solver.setBoundaryConditions()
    start_time = time.time()
    jacobi_solver.solve()
    end_time = time.time()
    l2_error = jacobi_solver.computeL2Error()
    solver.mpi_finalize()
    return end_time - start_time, jacobi_solver.current_iteration, jacobi_solver.current_residual, l2_error

def main():
    # Parameters
    matrix_sizes = [10, 20, 40, 80, 160, 320, 640, 1280]
    num_processes_list = [3, 4, 5, 6]
    max_iters = 10000
    tol = 1e-6

    # Initialize results dictionary
    results = {num: [] for num in num_processes_list}

    # Run the solver for each combination of matrix size and number of processes
    for num_processes in num_processes_list:
        for size in tqdm(matrix_sizes, desc=f'Running with {num_processes} processes'):
            result = subprocess.run(['mpirun', '-np', str(num_processes), 'python3', 'test_solver.py', str(size), str(size), str(max_iters), str(tol)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"Error running with {num_processes} processes and matrix size {size}x{size}: {result.stderr}")
            else:
                exec_time, current_iteration, current_residual, l2_error = map(float, result.stdout.strip().split())
                results[num_processes].append((size, exec_time, current_iteration, current_residual, l2_error))

    # Plot execution time
    plt.figure(figsize=(12, 8))
    for num_processes, times in results.items():
        sizes, exec_times, iterations, residuals, l2_errors = zip(*times)
        plt.plot(sizes, exec_times, label=f'{num_processes} Processes')

    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time of Jacobi Solver with Varying Processors and Matrix Sizes')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # Plot L2 error for 4 processes
    plt.figure(figsize=(12, 8))
    sizes, exec_times, iterations, residuals, l2_errors = zip(*results[4])
    plt.plot(sizes, l2_errors, label='L2 Error with 4 Processes')

    plt.xlabel('Matrix Size')
    plt.ylabel('L2 Error')
    plt.title('L2 Error of Jacobi Solver with 4 Processes')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 5:

        max_iters = int(sys.argv[3])
        tol = float(sys.argv[4])
        exec_time, current_iteration, current_residual, l2_error = run_solver(n, m, max_iters, tol)
        print(exec_time, current_iteration, current_residual, l2_error)
    else:
        main()

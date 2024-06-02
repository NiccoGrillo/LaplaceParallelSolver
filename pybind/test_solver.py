import solver
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

def run_solver(n, max_iters, tol):
    solver.mpi_init()
    jacobi_solver = solver.JacobiSolver(n, max_iters, tol, solver.myFunc, solver.exactSolution)
    start_time = time.time()
    jacobi_solver.solve()
    end_time = time.time()
    l2_error = jacobi_solver.computeL2Error()
    solver.mpi_finalize()
    return end_time - start_time, jacobi_solver.current_iteration, jacobi_solver.curr_avg_residual_ranks, l2_error

def main():
    # Parameters
    matrix_sizes = [20, 40, 80, 160, 320, 640]

    num_processes_list = [3, 4, 5, 6]
    max_iters = 1000000
    tol = 1e-5

    # Initialize results dictionary
    results = {num: [] for num in num_processes_list}

    # Run the solver for each combination of matrix size and number of processes
    for num_processes in num_processes_list:
        for size in matrix_sizes:
            print(f'Running with {num_processes} processes and matrix size {size}x{size}')
            result = subprocess.run(['mpirun', '-np', str(num_processes), 'python3', 'test_solver.py', str(size), str(max_iters), str(tol)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"Error running with {num_processes} processes and matrix size {size}x{size}: {result.stderr}")
            else:
                output = result.stdout.strip().split("\n")
                # Filter lines containing the marker "RESULT:"
                result_lines = [line for line in output if line.startswith("RESULT:")]
                
                # Ensure we have at least one result line
                if len(result_lines) == 0:
                    print(f"Unexpected output format: {result.stdout}")
                    continue

                # Use only the first result line
                result_data = result_lines[0].replace("RESULT:", "").strip().split()
                
                try:
                    exec_time = float(result_data[0])
                    current_iteration = float(result_data[1])
                    current_residual = float(result_data[2])
                    l2_error = float(result_data[3])
                except ValueError as e:
                    print(f"Error converting output to float: {e}, Output: {result_data}")
                    continue

                results[num_processes].append((size, exec_time, current_iteration, current_residual, l2_error))

    # Plot execution time
    plt.figure(figsize=(12, 8))
    for num_processes, times in results.items():
        if times:  # Ensure there are results to plot
            sizes, exec_times, iterations, residuals, l2_errors = zip(*times)
            plt.plot(sizes, exec_times, label=f'{num_processes} Processes')

    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time of Jacobi Solver with Varying Processors and Matrix Sizes\n tolerance: {tol}, max_iters: {max_iters}')
    plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    plt.grid(True)  
    plt.savefig('execution_time.png')  # Save the plot to a file

    # Plot L2 error for 4 processes
    if 4 in results and results[4]:  # Ensure there are results for 4 processes
        plt.figure(figsize=(12, 8))
        sizes, exec_times, iterations, residuals, l2_errors = zip(*results[4])
        plt.plot(sizes, l2_errors, label='L2 Error with 4 Processes')

        plt.xlabel('Matrix Size')
        plt.ylabel('L2 Error')
        plt.title('L2 Error of Jacobi Solver with 4 Processes')
        plt.legend()
        plt.grid(True)
        plt.savefig('l2_error.png')  # Save the plot to a file

if __name__ == "__main__":
    if len(sys.argv) == 4:
        size = int(sys.argv[1])
        max_iters = int(sys.argv[2])
        tol = float(sys.argv[3])
        exec_time, current_iteration, curr_avg_residual_ranks, l2_error = run_solver(size, max_iters, tol)
        print(f"RESULT: {exec_time} {current_iteration} {curr_avg_residual_ranks} {l2_error}")
    else:
        main()

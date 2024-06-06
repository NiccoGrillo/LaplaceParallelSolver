import solver
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import time

def myFunc(x, y):
    return 8 * np.pi * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def exactSolution(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def zerofunc(x, y):
    return 0.0
def g_func(x, y):
    # return np.cos(2 * np.py * x) + np.sin(2 * np.pi * y)
    return 0

def run_solver(n, max_iters, tol, func, exact_sol, bc_type="dirichlet", alpha=1.0, beta=1.0):
    solver.mpi_init()
    try:
        if bc_type == "dirichlet":
            boundary_condition = solver.DirichletBoundaryCondition(zerofunc)
            jacobi_solver = solver.JacobiSolverDirichlet(n, max_iters, tol, func, exact_sol, boundary_condition, False)
        elif bc_type == "robin":
            # g_func = lambda x, y: 0 # Adjust this as needed
            boundary_condition = solver.RobinBoundaryCondition(alpha, beta, g_func)
            jacobi_solver = solver.JacobiSolverRobin(n, max_iters, tol, func, exact_sol, boundary_condition, False)
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")
        exec_time = jacobi_solver.solve()
        l2_error = jacobi_solver.computeL2Error()
    finally:
        solver.mpi_finalize()
    return exec_time, jacobi_solver.current_iteration, jacobi_solver.curr_avg_residual_ranks, l2_error

def main():
    # Parameters
    # Parameters
    matrix_sizes = [64, 128, 256, 512]
    num_processes_list = [2, 3, 4, 5, 6]
    max_iters = 1000000
    tol = 1e-8
    bc_types = ["dirichlet", "robin"]
    alpha = 1.0
    beta = 1.0

    # Initialize results dictionary
    results = {(num, bc): [] for num in num_processes_list for bc in bc_types}

    # Run the solver for each combination of matrix size, number of processes, and multithreading flag
    for num_processes in num_processes_list:
        for size in matrix_sizes:
            for bc_type in bc_types:
                print(f'Running with {num_processes} processes, matrix size {size}x{size}, boundary condition: {bc_type}')
                result = subprocess.run(['mpirun', '-np', str(num_processes), 'python3', 'test_solver.py', str(size), str(max_iters), str(tol), bc_type, str(alpha), str(beta), str(False)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    print(f"Error running with {num_processes} processes, matrix size {size}x{size}, boundary condition: {bc_type}")
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

                    results[(num_processes, bc_type)].append((size, exec_time, current_iteration, current_residual, l2_error))

    # Plot execution time for Dirichlet
    plt.figure(figsize=(12, 8))
    for (num_processes, bc_type), times in results.items():
        if bc_type == "dirichlet" and times:  # Ensure there are results to plot
            sizes, exec_times, iterations, residuals, l2_errors = zip(*times)
            plt.plot(np.log2(sizes), exec_times, label=f'{num_processes} Processes')

    plt.xlabel('Log2(Matrix Size)')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time of Jacobi Solver with Dirichlet Boundary Condition\n tolerance: {tol}, max_iters: {max_iters}')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_time_dirichlet_pybind.png')  # Save the plot to a file

    # Plot execution time for Robin
    plt.figure(figsize=(12, 8))
    for (num_processes, bc_type), times in results.items():
        if bc_type == "robin" and times:  # Ensure there are results to plot
            sizes, exec_times, iterations, residuals, l2_errors = zip(*times)
            plt.plot(np.log2(sizes), exec_times, label=f'{num_processes} Processes')

    plt.xlabel('Log2(Matrix Size)')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time of Jacobi Solver with Robin Boundary Condition\n tolerance: {tol}, max_iters: {max_iters}')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_time_robin_pybind.png')  # Save the plot to a file

    # Plot L2 error
    plt.figure(figsize=(12, 8))
    for (num_processes, bc_type), times in results.items():
        if num_processes == 4 and times:  # Ensure there are results for 4 processes
            sizes, exec_times, iterations, residuals, l2_errors = zip(*times)
            plt.plot(np.log2(sizes), l2_errors, label=f'L2 Error {bc_type}')

    plt.xlabel('Log2(Matrix Size)')
    plt.ylabel('L2 Error')
    plt.title('L2 Error of Jacobi Solver with 4 Processes and Different Boundary Conditions')
    plt.legend()
    plt.grid(True)
    plt.savefig('l2_error_rob_dir_pybind.png')  # Save the plot to a file

if __name__ == "__main__":
    if len(sys.argv) >= 8:
        size = int(sys.argv[1])
        max_iters = int(sys.argv[2])
        tol = float(sys.argv[3])
        bc_type = sys.argv[4]
        alpha = float(sys.argv[5])
        beta = float(sys.argv[6])
        exec_time, current_iteration, curr_avg_residual_ranks, l2_error = run_solver(size, max_iters, tol, myFunc, exactSolution, bc_type, alpha, beta)
        print(f"RESULT: {exec_time} {current_iteration} {curr_avg_residual_ranks} {l2_error}")
    else:
        main()

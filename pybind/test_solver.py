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

def run_solver(n, max_iters, tol, func, exact_sol, bc_type="dirichlet", alpha=1.0, beta=1.0, use_multithreading=True):
    solver.mpi_init()
    try:
        if bc_type == "dirichlet":
            boundary_condition = solver.DirichletBoundaryCondition()
            jacobi_solver = solver.JacobiSolverDirichlet(n, max_iters, tol, func, exact_sol, boundary_condition, use_multithreading)
        elif bc_type == "robin":
            g_func = lambda x, y: 0.0  # Adjust this as needed
            boundary_condition = solver.RobinBoundaryCondition(alpha, beta, g_func)
            jacobi_solver = solver.JacobiSolverRobin(n, max_iters, tol, func, exact_sol, boundary_condition, use_multithreading)
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

        print("Starting solver")
        start_time = time.time()
        exec_time = jacobi_solver.solve()
        end_time = time.time()
        l2_error = jacobi_solver.computeL2Error()
        print(f"Solver finished in {end_time - start_time} seconds")
    finally:
        solver.mpi_finalize()
    return exec_time, jacobi_solver.current_iteration, jacobi_solver.curr_avg_residual_ranks, l2_error

def main():
    # Parameters
    matrix_sizes = [20, 40, 80, 160, 320, 640]
    num_processes_list = [3, 4, 5, 6]
    max_iters = 1000000
    tol = 1e-5
    bc_type = "dirichlet"  # or "robin"
    alpha = 1.0
    beta = 1.0

    # Initialize results dictionary
    results = {(num, mt): [] for num in num_processes_list for mt in [True, False]}

    # Run the solver for each combination of matrix size, number of processes, and multithreading flag
    for num_processes in num_processes_list:
        for size in matrix_sizes:
            for use_multithreading in [True, False]:
                print(f'Running with {num_processes} processes, matrix size {size}x{size}, multithreading {use_multithreading}')
                result = subprocess.run(['mpirun', '-np', str(num_processes), 'python3', 'test_solver.py', str(size), str(max_iters), str(tol), bc_type, str(alpha), str(beta), str(use_multithreading)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    print(f"Error running with {num_processes} processes, matrix size {size}x{size}, multithreading {use_multithreading}: {result.stderr}")
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

                    results[(num_processes, use_multithreading)].append((size, exec_time, current_iteration, current_residual, l2_error))

    # Plot execution time
    plt.figure(figsize=(12, 8))
    for (num_processes, use_multithreading), times in results.items():
        if times:  # Ensure there are results to plot
            sizes, exec_times, iterations, residuals, l2_errors = zip(*times)
            line_style = '-' if use_multithreading else '--'
            plt.plot(sizes, exec_times, line_style, label=f'{num_processes} Processes {"with" if use_multithreading else "without"} MT')

    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time of Jacobi Solver with Varying Processors, Matrix Sizes, and Multithreading\n tolerance: {tol}, max_iters: {max_iters}')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_time.png')  # Save the plot to a file

    # Plot L2 error for 4 processes
    plt.figure(figsize=(12, 8))
    for use_multithreading in [True, False]:
        if (4, use_multithreading) in results and results[(4, use_multithreading)]:  # Ensure there are results for 4 processes
            sizes, exec_times, iterations, residuals, l2_errors = zip(*results[(4, use_multithreading)])
            line_style = '-' if use_multithreading else '--'
            plt.plot(sizes, l2_errors, line_style, label=f'L2 Error with 4 Processes {"with" if use_multithreading else "without"} MT')

    plt.xlabel('Matrix Size')
    plt.ylabel('L2 Error')
    plt.title('L2 Error of Jacobi Solver with 4 Processes and Multithreading')
    plt.legend()
    plt.grid(True)
    plt.savefig('l2_error.png')  # Save the plot to a file

if __name__ == "__main__":
    if len(sys.argv) >= 8:
        size = int(sys.argv[1])
        max_iters = int(sys.argv[2])
        tol = float(sys.argv[3])
        bc_type = sys.argv[4]
        alpha = float(sys.argv[5])
        beta = float(sys.argv[6])
        use_multithreading = sys.argv[7].lower() == 'true'
        exec_time, current_iteration, curr_avg_residual_ranks, l2_error = run_solver(size, max_iters, tol, myFunc, exactSolution, bc_type, alpha, beta, use_multithreading)
        print(f"RESULT: {exec_time} {current_iteration} {curr_avg_residual_ranks} {l2_error}")
    else:
        main()

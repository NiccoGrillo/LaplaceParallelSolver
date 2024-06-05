#include "JacobiSolver.hpp"
#include "DirichletBoundaryCondition.hpp"
#include "RobinBoundaryCondition.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    int num = 320;
    int max_iters = 10000;
    double tol = 1e-12;
    bool use_multithreading = true;
    if (rank == 0) {
        std::cout << "Number of grid points: " << num << std::endl;
        std::cout << "Maximum number of iterations: " << max_iters << std::endl;
        std::cout << "Tolerance: " << tol << std::endl;
        std::cout << "Using multithreading: " << (use_multithreading ? "true" : "false") << "\n\n" << std::endl;
    }

    auto func = [](double x, double y) { return 8 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y); };
    auto exact_sol = [](double x, double y) { return sin(2 * M_PI * x) * sin(2 * M_PI * y); };

    // For Dirichlet boundary condition
    if(rank ==0)
        std::cout << "Dirichlet boundary condition" << std::endl;
    DirichletBoundaryCondition dirichlet_bc;
    JacobiSolver<DirichletBoundaryCondition> solverDirichlet(num, max_iters, tol, func, exact_sol, dirichlet_bc, use_multithreading);
    double exec_time_dir = solverDirichlet.solve();
    solverDirichlet.saveSolution("solution_dirichlet.vtk", true, false);
    double l2_error_dir = solverDirichlet.computeL2Error();

    if (rank == 0) {
        std::cout << "Reached solution in sec: " << exec_time_dir << std::endl;
        std::cout << "L2 Error: " << l2_error_dir << std::endl;
    }

    // For Robin boundary condition
    if (rank == 0)
        std::cout << "\n\nRobin boundary condition" << std::endl;
    double alpha = 1.0; // example value
    double beta = 1.0;  // example value
    auto g_func = [](double x, double y) { return cos(2 * M_PI * x) + sin(2 * M_PI * y); }; // Example boundary function
    RobinBoundaryCondition robin_bc(alpha, beta, g_func);
    JacobiSolver<RobinBoundaryCondition> solverRobin(num, max_iters, tol, func, exact_sol, robin_bc, use_multithreading);
    double exec_time_rob = solverRobin.solve();
    solverRobin.saveSolution("solution_robin.vtk", true, false);
    double l2_error_robin = solverRobin.computeL2Error();

    if (rank == 0) {
        std::cout << "Reached solution in sec: " << exec_time_rob << std::endl;
        std::cout << "L2 Error: " << l2_error_robin << std::endl;
    }    

    MPI_Finalize();
    return 0;
}

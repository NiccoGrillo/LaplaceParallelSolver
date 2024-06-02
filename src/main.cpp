#include <mpi.h>
#include <iostream>
#include <cmath>
#include "JacobiSolver.hpp"
// #include "writeVTK.hpp"

double myFunc(double x, double y) {
    return 8 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y);
}

double exactSolution(double x, double y) {
    return sin(2 * M_PI * x) * sin(2 * M_PI * y);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Running with " << size << " processes" << std::endl;
    }

    int n = 200;  // total number of rows
    int max_iters = 200000;
    double tol = 1e-13;

    JacobiSolver solver(n, max_iters, tol, myFunc, exactSolution);

    // solver.iterJacobi();
    solver.solve();

    double l2_error = solver.computeL2Error();

    if (rank == 0) {
        std::cout << "L2 Error: " << l2_error << std::endl;
    }
         
    solver.saveSolution("solution.vtk", true, false); // save also the exact solution and do not print the solution

    MPI_Finalize();
    return 0;
}
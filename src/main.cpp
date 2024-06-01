#include <mpi.h>
#include <iostream>
#include "JacobiSolver.hpp"

double myFunc(double x, double y) {
    // return 8 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y); //M_PI is pi greek letter
    return x + y;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Running with " << size << " processes" << std::endl;
        std::cout << "Before iteration" << std::endl;
    }

    int n = 14;  // total number of rows
    int m = 5;   // total number of columns


    JacobiSolver solver(n, m, myFunc);


    //test jacobi iteration
    solver.solve();

    // solver.printLocalMatrixF();
    solver.printLocalMatrixU();
    

    MPI_Finalize();
    return 0;
}

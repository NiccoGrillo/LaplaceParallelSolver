#ifndef JACOBISOLVER_HPP
#define JACOBISOLVER_HPP

#include <Eigen/Dense>
#include <mpi.h>
#include <functional>
#include <iostream>
// #include "writeVTK.hpp"
class JacobiSolver {
public:
    JacobiSolver(int num, int max_iters, double tol, std::function<double(double, double)> func, std::function<double(double, double)> exact_sol);

    void setBoundaryConditions();
    double iterJacobi();
    void solve();
    void printLocalMatrixF() const;
    void printLocalMatrixU() const;
    void saveSolution(const std::string& filename, bool save_also_exact_solution,  bool print = false) const;
    double computeL2Error();

    int current_iteration;
    double curr_avg_residual_ranks;

private:
    int n; // number of rows and columns
    int max_iterations;
    double tolerance;
    double h;
    int rank, size;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_F;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_U;
    std::function<double(double, double)> exact_solution;

    void initializeMatrix(std::function<double(double, double)> func);
    std::tuple<int, int, int> localToGlobal( int other_rank = -1, bool real_rows = false, bool real_local = false) const;
};

#endif // JACOBISOLVER_HPP

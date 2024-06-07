#ifndef JACOBISOLVER_HPP
#define JACOBISOLVER_HPP

#include <Eigen/Dense>
#include <mpi.h>
#include <omp.h>
#include <functional>
#include <iostream>
#include "BoundaryCondition.hpp"
#include "writeVTK.hpp"
#include "chrono.hpp"


template <typename BoundaryConditionType>
class JacobiSolver {
public:
    JacobiSolver(int num, int max_iters, double tol, std::function<double(double, double)> func, std::function<double(double, double)> exact_sol, BoundaryConditionType bc, bool use_multithreading = false);

    double solve();
    double computeL2Error();
    void printLocalMatrixF() const;
    void printLocalMatrixU() const;
    void saveSolution(const std::string& filename, bool save_also_exact_solution, bool print = false) const;

    int current_iteration;
    double curr_avg_residual_ranks;    
    bool use_multithreading = false;

private:
    int n; // number of rows and columns
    int max_iterations;
    double tolerance;
    double h;
    int rank, size;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_F;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_U;
    std::function<double(double, double)> exact_solution;
    BoundaryConditionType boundary_condition;

    void initializeMatrix(std::function<double(double, double)> func);
    double iterJacobi();
    std::tuple<int, int, int> localToGlobal(int other_rank = -1, bool real_rows = false, bool real_local = false) const;
};

#include "JacobiSolver.tpp"

#endif // JACOBISOLVER_HPP

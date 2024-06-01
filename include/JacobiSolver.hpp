#ifndef JACOBISOLVER_HPP
#define JACOBISOLVER_HPP

#include <Eigen/Dense>
#include <mpi.h>
#include <functional>
#include <iostream>

class JacobiSolver {
public:
    JacobiSolver(int rows, int cols, std::function<double(double, double)> func);
    void setBoundaryConditions();
    void printLocalMatrixF() const;
    void printLocalMatrixU() const;
    void solve();
    double iterJacobi();

private:
    int n, m; //n -> number of rows, m -> number of columns
    int rank, size;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_matrix;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_U;

    void initializeMatrix(std::function<double(double, double)> func);

    //new method: iterJacobi

    //new method: updateMatrix
};

#endif // JACOBISOLVER_HPP

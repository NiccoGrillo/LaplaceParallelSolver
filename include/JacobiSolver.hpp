#ifndef JACOBISOLVER_HPP
#define JACOBISOLVER_HPP

#include <Eigen/Dense>
#include <mpi.h>
#include <functional>
#include <iostream>

class JacobiSolver {
public:
    JacobiSolver(int rows, int cols, std::function<double(double, double)> func);
    void printLocalMatrixF() const;
    void printLocalMatrixU() const;
    void print_temp() const{
        std::cout << "Rank: " << rank << std::endl;
        std::cout << local_matrix << std::endl;
    };
    void solve();

    void iterJacobi();

private:
    int n, m; //n -> number of rows, m -> number of columns
    int rank, size;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_matrix;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> local_U;

    void initializeMatrix(std::function<double(double, double)> func);
    // void distributeMatrix();

    //new method: iterJacobi
    
    //new method: computeResidual
    double computeResidual();
    //new method: updateMatrix
    void updateMatrix();
};

#endif // JACOBISOLVER_HPP

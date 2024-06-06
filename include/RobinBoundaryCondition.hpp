#ifndef ROBINBOUNDARYCONDITION_HPP
#define ROBINBOUNDARYCONDITION_HPP

#include "BoundaryCondition.hpp"
#include <Eigen/Dense>
#include <functional>

class RobinBoundaryCondition : public BoundaryCondition {
public:
    RobinBoundaryCondition(double alpha, double beta, std::function<double(double, double)> g_func)
        : alpha(alpha), beta(beta), g_func(g_func) {}

    void setBoundaryConditions(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& local_U, int rank, int size, int n, double h, int start_row) override {
        applyBoundaryConditionsAtIteration(local_U, rank, size, n, h, start_row);
    }

    void applyBoundaryConditionsAtIteration(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                             Eigen::RowMajor>& local_U, int rank, int size, int n,
                                             double h, int start_row) override {
        //using euler's method to approximate the boundary conditions
        if (rank == 0) {
            for (int j = 0; j < local_U.cols(); ++j) {
                double x = j * h;
                local_U(0, j) = (g_func(x, 1)*h + beta * local_U(1, j)) / (alpha + beta);
            }
        }
        if (rank == size - 1) {
            for (int j = 0; j < local_U.cols(); ++j) {
                double x = j * h;
                local_U(local_U.rows() - 1, j) = (g_func(x, 0)*h + beta * local_U(local_U.rows() - 2, j)) / (alpha + beta);
            }
        }
        for (int i = 0; i < local_U.rows(); ++i) {
            double y = (start_row + i) * h;
            local_U(i, 0) = (g_func(0,1- y)*h + beta * local_U(i, 1)) / (alpha + beta);
            local_U(i, n - 1) = (g_func(1,1- y)*h + beta * local_U(i, local_U.cols() - 2)) / (alpha + beta);
        }
    }

private:
    double alpha;
    double beta;
    std::function<double(double, double)> g_func;
};

#endif // ROBINBOUNDARYCONDITION_HPP

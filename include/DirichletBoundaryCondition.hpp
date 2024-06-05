#ifndef DIRICHLETBOUNDARYCONDITION_HPP
#define DIRICHLETBOUNDARYCONDITION_HPP

#include "BoundaryCondition.hpp"
#include <Eigen/Dense>

class DirichletBoundaryCondition : public BoundaryCondition {
public:
    void setBoundaryConditions(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& local_U, int rank, int size, int n, double h) override {
        if (rank == 0) {
            local_U.row(0).setZero();
        }
        if (rank == size - 1) {
            local_U.row(local_U.rows() - 1).setZero();
        }
        local_U.col(0).setZero();
        local_U.col(n - 1).setZero();
    }

    void applyBoundaryConditionsAtIteration(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& local_U, int rank, int size, int n, double h) override {
        // No additional boundary conditions to apply at each iteration for Dirichlet
    }
};

#endif // DIRICHLETBOUNDARYCONDITION_HPP

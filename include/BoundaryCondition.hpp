#ifndef BOUNDARYCONDITION_HPP
#define BOUNDARYCONDITION_HPP

#include <Eigen/Dense>

class BoundaryCondition {
public:
    virtual ~BoundaryCondition() = default;
    virtual void setBoundaryConditions(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& local_U, int rank, int size, int n, double h) = 0;
    virtual void applyBoundaryConditionsAtIteration(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& local_U, int rank, int size, int n, double h) = 0;
};

#endif // BOUNDARYCONDITION_HPP

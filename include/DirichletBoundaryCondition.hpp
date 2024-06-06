#ifndef DIRICHLETBOUNDARYCONDITION_HPP
#define DIRICHLETBOUNDARYCONDITION_HPP

#include "BoundaryCondition.hpp"
#include <Eigen/Dense>

class DirichletBoundaryCondition : public BoundaryCondition {
public:
    DirichletBoundaryCondition(std::function<double(double, double)> g_func):
        g_func(g_func) {}
    void setBoundaryConditions(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& local_U,
                                 int rank, int size, int n, double h, int start_row) override{   

        if (rank == 0) {
            for(int i = 0; i < n; ++i) {
                double x = i * h;
                local_U(0, i) = g_func(x, 1);
            }
        }
        if (rank == size - 1) {
            for(int i = 0; i < n; ++i) {
                double x = i * h;
                local_U(local_U.rows()-1, i) = g_func(x, 0);
            }
        }
        //let's now apply g_func to the boundary
        for (int i = 0; i < local_U.rows(); ++i) {
            double x = (start_row + i) * h;
            local_U(i, 0) = g_func(0, 1 - x);
            local_U(i, n - 1) = g_func(1, 1 - x);
        }
    }
    void applyBoundaryConditionsAtIteration(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                             Eigen::RowMajor>& local_U, int rank, int size, int n,
                                             double h, int start_row) override {
                                                //Dirichlet boundary condition does not change at each iteration -> simply do not apply anything
                                             }
private:
std::function<double(double, double)> g_func;
};

#endif // DIRICHLETBOUNDARYCONDITION_HPP

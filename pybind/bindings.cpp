#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <mpi.h>
#include <functional>
#include "../include/JacobiSolver.hpp"
#include "../include/DirichletBoundaryCondition.hpp"
#include "../include/RobinBoundaryCondition.hpp"

namespace py = pybind11;

PYBIND11_MODULE(solver, m) {
    py::class_<DirichletBoundaryCondition>(m, "DirichletBoundaryCondition")
        .def(py::init<>());

    py::class_<RobinBoundaryCondition>(m, "RobinBoundaryCondition")
        .def(py::init<double, double, std::function<double(double, double)>>());

    py::class_<JacobiSolver<DirichletBoundaryCondition>>(m, "JacobiSolverDirichlet")
        .def(py::init<int, int, double, std::function<double(double, double)>, std::function<double(double, double)>, DirichletBoundaryCondition, bool>())
        .def("solve", &JacobiSolver<DirichletBoundaryCondition>::solve)
        .def("computeL2Error", &JacobiSolver<DirichletBoundaryCondition>::computeL2Error)
        .def("printLocalMatrixF", &JacobiSolver<DirichletBoundaryCondition>::printLocalMatrixF)
        .def("printLocalMatrixU", &JacobiSolver<DirichletBoundaryCondition>::printLocalMatrixU)
        .def("saveSolution", &JacobiSolver<DirichletBoundaryCondition>::saveSolution)
        .def_readonly("current_iteration", &JacobiSolver<DirichletBoundaryCondition>::current_iteration)
        .def_readonly("curr_avg_residual_ranks", &JacobiSolver<DirichletBoundaryCondition>::curr_avg_residual_ranks);

    py::class_<JacobiSolver<RobinBoundaryCondition>>(m, "JacobiSolverRobin")
        .def(py::init<int, int, double, std::function<double(double, double)>, std::function<double(double, double)>, RobinBoundaryCondition, bool>())
        .def("solve", &JacobiSolver<RobinBoundaryCondition>::solve)
        .def("computeL2Error", &JacobiSolver<RobinBoundaryCondition>::computeL2Error)
        .def("printLocalMatrixF", &JacobiSolver<RobinBoundaryCondition>::printLocalMatrixF)
        .def("printLocalMatrixU", &JacobiSolver<RobinBoundaryCondition>::printLocalMatrixU)
        .def("saveSolution", &JacobiSolver<RobinBoundaryCondition>::saveSolution)
        .def_readonly("current_iteration", &JacobiSolver<RobinBoundaryCondition>::current_iteration)
        .def_readonly("curr_avg_residual_ranks", &JacobiSolver<RobinBoundaryCondition>::curr_avg_residual_ranks);

    m.def("mpi_init", []() {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    });

    m.def("mpi_finalize", []() {
        MPI_Finalize();
    });
}

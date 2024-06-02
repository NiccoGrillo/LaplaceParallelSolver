#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>  // Include this header for std::function conversions
#include <mpi.h>
#include <functional>
#include "../include/JacobiSolver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(solver, m) {
    py::class_<JacobiSolver>(m, "JacobiSolver")
        .def(py::init<int, int, double, std::function<double(double, double)>, std::function<double(double, double)>>())
        .def("setBoundaryConditions", &JacobiSolver::setBoundaryConditions)
        .def("iterJacobi", &JacobiSolver::iterJacobi)
        .def("solve", &JacobiSolver::solve)
        .def("printLocalMatrixF", &JacobiSolver::printLocalMatrixF)
        .def("printLocalMatrixU", &JacobiSolver::printLocalMatrixU)
        .def("computeL2Error", &JacobiSolver::computeL2Error)
        .def_readonly("current_iteration", &JacobiSolver::current_iteration)
        .def_readonly("curr_avg_residual_ranks", &JacobiSolver::curr_avg_residual_ranks);

    m.def("myFunc", [](double x, double y) {
        return 8 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y);
    });

    m.def("exactSolution", [](double x, double y) {
        return sin(2 * M_PI * x) * sin(2 * M_PI * y);
    });

    m.def("mpi_init", []() {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    });

    m.def("mpi_finalize", []() {
        MPI_Finalize();
    });
}
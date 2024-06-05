// #include "JacobiSolver.hpp"
#include "BoundaryCondition.hpp"
#include "DirichletBoundaryCondition.hpp"
#include "RobinBoundaryCondition.hpp"

template <typename BoundaryConditionType>
JacobiSolver<BoundaryConditionType>::JacobiSolver(int num, int max_iters, double tol, std::function<double(double, double)> func, std::function<double(double, double)> exact_sol, BoundaryConditionType bc, bool use_multithreading)
    : n(num), max_iterations(max_iters), tolerance(tol), exact_solution(exact_sol), boundary_condition(bc), current_iteration(0), curr_avg_residual_ranks(0.0), use_multithreading(use_multithreading) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    h = 1.0 / (n - 1);

    initializeMatrix(func);
    local_U = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(local_F.rows(), n);
    boundary_condition.setBoundaryConditions(local_U, rank, size, n, h);
}

template <typename BoundaryConditionType>
void JacobiSolver<BoundaryConditionType>::initializeMatrix(std::function<double(double, double)> func) {
    auto [start_row, end_row, num_rows] = localToGlobal();
    local_F.resize(num_rows, n);
    for (int i = 0; i < local_F.rows(); ++i) {
        for (int j = 0; j < n; ++j) {
            local_F(i, j) = func((start_row + i) * h, 1 - j * h);
        }
    }
}

template <typename BoundaryConditionType>
double JacobiSolver<BoundaryConditionType>::iterJacobi() {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_new_buf = local_U;
    double res_sum_squared = 0.0;

    if (use_multithreading) {
        #pragma omp parallel for reduction(+:res_sum_squared) collapse(2)
        for (int i = 1; i < local_U.rows() - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                U_new_buf(i, j) = 0.25 * (local_U(i - 1, j) + local_U(i + 1, j) + local_U(i, j - 1) + local_U(i, j + 1) - h * h * local_F(i, j));
                res_sum_squared += std::pow(local_U(i, j) - U_new_buf(i, j), 2);
            }
        }
    } else {
        for (int i = 1; i < local_U.rows() - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                U_new_buf(i, j) = 0.25 * (local_U(i - 1, j) + local_U(i + 1, j) + local_U(i, j - 1) + local_U(i, j + 1) - h * h * local_F(i, j));
                res_sum_squared += std::pow(local_U(i, j) - U_new_buf(i, j), 2);
            }
        }
    }

    local_U = U_new_buf;
    double res_k = std::sqrt(res_sum_squared * h);

    // Communicate the edge rows with the neighbors ranks
    for(int r = 0; r < size; ++r){


        if (rank == r) {
            if (rank != size - 1) {
                MPI_Request send_request;
                MPI_Isend(local_U.row(local_U.rows() - 2).data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_request);
                
                Eigen::VectorXd new_last_row(n);
                MPI_Request recv_request;
                MPI_Irecv(new_last_row.data(), n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_request);
                
                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
                local_U.row(local_U.rows() - 1) = new_last_row;
                
                MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            }
        } else if (rank == r + 1) {
            Eigen::VectorXd new_first_row(n);
            MPI_Request recv_request;
            MPI_Irecv(new_first_row.data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_request);
            
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            local_U.row(0) = new_first_row;

            MPI_Request send_request;
            MPI_Isend(local_U.row(1).data(), n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_request);
            
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        }
    }

    return res_k;
}

template <typename BoundaryConditionType>
void JacobiSolver<BoundaryConditionType>::solve() {
    current_iteration = 0;
    double res_k = 0.0;
    int converged_res = 0;

    do {
        current_iteration++;
        res_k = iterJacobi();
        
        // Apply boundary conditions at each iteration only for Robin
        if constexpr (std::is_same_v<BoundaryConditionType, RobinBoundaryCondition>) {
            boundary_condition.applyBoundaryConditionsAtIteration(local_U, rank, size, n, h);
        }
        
        int res_k_true = res_k < tolerance ? 1 : 0;
        MPI_Allreduce(&res_k, &curr_avg_residual_ranks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&res_k_true, &converged_res, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

    } while (!converged_res && current_iteration < max_iterations);

    if (rank == 0) {
        if (current_iteration >= max_iterations) {
            std::cout << "Stopped at iteration _"<< current_iteration <<"_ because the maximum number of iterations was reached with a final average residual among the processes of: " << curr_avg_residual_ranks << std::endl;
        } else {
            std::cout << "Stopped at iteration _"<< current_iteration <<"_ because the residual was less than the tolerance with a final average residual among the processes of: " << curr_avg_residual_ranks << std::endl;
        }
    }
}

template <typename BoundaryConditionType>
double JacobiSolver<BoundaryConditionType>::computeL2Error() {
    double local_l2_error = 0.0;

    int start_orig_row = std::get<0>(localToGlobal());

    for (int i = 1; i < local_F.rows() - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            double x = (start_orig_row + i) * h;
            double y = j * h;
            double exact_val = exact_solution(x, y);
            local_l2_error += std::pow(local_U(i, j) - exact_val, 2);
        }
    }

    double global_l2_error = 0.0;
    MPI_Allreduce(&local_l2_error, &global_l2_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return std::sqrt(global_l2_error * h);
}

template <typename BoundaryConditionType>
std::tuple<int, int, int> JacobiSolver<BoundaryConditionType>::localToGlobal(int other_rank, bool real_rows, bool real_local) const {
    int rows_per_process = n / size;
    int r = rank;

    if (other_rank != -1) {
        r = other_rank;
    }

    int start_row, end_row, num_rows, additional_row = 0;
    int real_start_row, real_end_row, real_num_rows;
    int real_local_start_row, real_local_end_row, real_local_num_rows;

    int rest = n % size;

    if (rest > r) {
        start_row = r * (rows_per_process + 1) - 1;
        real_start_row = r * (rows_per_process + 1);
        additional_row++;
    } else {
        start_row = rest * (rows_per_process + 1) + (r - rest) * rows_per_process - 1;
        real_start_row = rest * (rows_per_process + 1) + (r - rest) * rows_per_process;
    }

    if (r != 0 && r != size - 1) {
        end_row = start_row + rows_per_process + 1 + additional_row;
        real_end_row = start_row + additional_row + rows_per_process;
        real_local_start_row = 1;
    } else if (r == 0) {
        start_row = 0;
        real_start_row = 0;
        real_local_start_row = 0;
        end_row = start_row + rows_per_process + additional_row;
        real_end_row = end_row - 1;
    } else {
        end_row = n - 1;
        real_end_row = n - 1;
        real_local_start_row = 1;
    }

    num_rows = end_row - start_row + 1;
    real_num_rows = real_end_row - real_start_row + 1;
    real_local_num_rows = real_num_rows;
    real_local_end_row = real_local_start_row + real_local_num_rows - 1;

    if (real_rows) {
        return std::make_tuple(real_start_row, real_end_row, real_num_rows);
    } else if (real_local) {
        return std::make_tuple(real_local_start_row, real_local_end_row, real_local_num_rows);
    } else {
        return std::make_tuple(start_row, end_row, num_rows);
    }
}

template <typename BoundaryConditionType>
void JacobiSolver<BoundaryConditionType>::printLocalMatrixF() const {
    if (rank == 0) {
        std::cout << "Process " << rank << " local matrix F:" << std::endl;
        std::cout << local_F << std::endl;
        
        if (size > 1) {
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(nullptr, 0, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::cout << "Process " << rank << " local matrix F:" << std::endl;
        std::cout << local_F << std::endl;

        if (rank < size - 1) {
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

template <typename BoundaryConditionType>
void JacobiSolver<BoundaryConditionType>::printLocalMatrixU() const {
    if (rank == 0) {
        std::cout << "Process " << rank << " local matrix U:" << std::endl;
        std::cout << local_U << std::endl;
        
        if (size > 1) {
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(nullptr, 0, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::cout << "Process " << rank << " local matrix U:" << std::endl;
        std::cout << local_U << std::endl;

        if (rank < size - 1) {
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

template <typename BoundaryConditionType>
void JacobiSolver<BoundaryConditionType>::saveSolution(const std::string& filename, bool save_also_exact_solution, bool print) const {
    std::string new_filename = "../data/" + filename;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> full_U;
    if (rank == 0) {
        full_U.resize(n, n);

        auto [start_row, end_row, real_num_rows] = localToGlobal(0, true);
        for (int i = 0; i < real_num_rows; ++i) {
            full_U.row(start_row + i) = local_U.row(i);   
        }

        for (int r = 1; r < size; ++r) {
            auto [real_start_row, real_end_row, real_num_rows] = localToGlobal(r, true);

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp_matrix(real_num_rows, n);
            MPI_Recv(temp_matrix.data(), real_num_rows * n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            full_U.block(real_start_row, 0, real_num_rows, n) = temp_matrix;
        }
    } else {
        auto [real_start_row, real_end_row, real_num_rows] = localToGlobal(-1, true);
        auto [real_local_start_row, real_local_end_row, real_local_num_rows] = localToGlobal(-1, false, true);
        MPI_Send(local_U.block(real_local_start_row, 0, real_num_rows, n).data(), real_num_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0 && print) {
        std::cout << "Full matrix U:" << std::endl;
        std::cout << full_U << std::endl;
    }

    if (rank == 0) {
        std::vector<std::vector<double>> vec(full_U.rows(), std::vector<double>(full_U.cols()));

        for (int i = 0; i < full_U.rows(); ++i)
            for (int j = 0; j < full_U.cols(); ++j)
                vec[i][j] = full_U(i, j);
        
        generateVTKFile(new_filename, vec, n - 1, n - 1, h, h);
    }

    if (rank == 0 && save_also_exact_solution) {
        std::vector<std::vector<double>> vecc(full_U.rows(), std::vector<double>(full_U.cols()));
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_corr(n, n);

        for (int i = 0; i < full_U.rows(); ++i)
            for (int j = 0; j < full_U.cols(); ++j)
                vecc[i][j] = exact_solution(i * h, j * h);

        generateVTKFile("../data/exact_solution.vtk", vecc, n - 1, n - 1, h, h);
    }
}

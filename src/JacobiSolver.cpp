#include "JacobiSolver.hpp"
#include <iostream>

JacobiSolver::JacobiSolver(int num, int max_iters, double tol, std::function<double(double, double)> func, std::function<double(double, double)> exact_sol)
    : n(num), max_iterations(max_iters), tolerance(tol), exact_solution(exact_sol), current_iteration(0), current_residual(0.0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    h = 1.0 / (n - 1);

    initializeMatrix(func);
    local_U = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(local_F.rows(), n);
    setBoundaryConditions();
}

void JacobiSolver::initializeMatrix(std::function<double(double, double)> func) {
    int rows_per_process = n / size;

    //this fills the f matrix with the computed values
    if (rank == 0) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> full_matrix(n, n);
        int count_temp = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j){
                full_matrix(i, j) = func(static_cast<double>(i) / (n - 1), static_cast<double>(j) / (n - 1));

            }
        }

        // Share the full matrix row information with all processes
        for (int r = 0; r < size; ++r) {
            int start_row, end_row, num_rows, additional_row = 0;
            int rest = n % size;

            if (rest > r) {
                start_row = r * (rows_per_process + 1) - 1;
                additional_row++;
            } else {
                start_row = rest * (rows_per_process + 1) + (r - rest) * rows_per_process - 1;
            }


            if (r != 0 and r != size - 1) {
                // start_row = r*rows_per_process - 1;
                end_row = start_row + rows_per_process + 1 + additional_row;

            }
            else if (r == 0){
                start_row = 0;
                end_row = start_row + rows_per_process + additional_row;
            }
            else{ //at last rank we cannot have additional rows (otherwise rest is 0)
                end_row = n - 1;
            }
            num_rows = end_row - start_row + 1;

            if (r != 0) {
                MPI_Send(&num_rows, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(full_matrix.data() + start_row * n, num_rows *n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            } else {
                local_F = full_matrix.block(start_row, 0, num_rows, n);
            }
        }

    } else {
        int num_rows;
        MPI_Recv(&num_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_F.resize(num_rows, n);
        MPI_Recv(local_F.data(), num_rows * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void JacobiSolver::setBoundaryConditions() {
    if (rank == 0) {
        local_U.row(0).setZero();
    }
    if (rank == size - 1) {
        local_U.row(local_U.rows() - 1).setZero();
    }
    local_U.col(0).setZero();
    local_U.col(local_U.cols() - 1).setZero();
}

double JacobiSolver::iterJacobi(){
    //for each block of the local U compute the jacobi
    //compute the new matrix U 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_new_buf(local_U.rows(), n);
    double res_sum_squared = 0.0;
    
    #pragma omp parallel for reduction(+:res_sum_squared) collapse(2)
    for (int i = 1; i < local_F.rows() - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            U_new_buf(i, j) = 0.25 * (local_U(i-1, j) + local_U(i+1, j) + local_U(i, j-1) + local_U(i, j+1) - h*h*local_F(i, j));
            res_sum_squared += std::pow(local_U(i, j) - U_new_buf(i, j), 2);
            //compute residual

        }
    }

    local_U = U_new_buf;
    double res_k = std::sqrt(res_sum_squared * h);

    // communicate the edge rows with the neighbors ranks
    for(int r = 0; r < size; ++r){
        int rows_per_process = n / size;
        int start_row;
        int num_rows = rows_per_process;
        int rest = n % size;
        //send the last row of the previous rank to the next rank
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

    //now the local_U is ready for the next iteration

    //now return the residual
    return res_k;

}

void JacobiSolver::solve() {
    current_iteration = 0;
    double res_k = 0.0;

    do {
        current_iteration++;
        res_k = iterJacobi();
        MPI_Allreduce(&res_k, &current_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Iteration " << current_iteration << ", Residual: " << current_residual << std::endl;
        }
    } while (current_residual > tolerance && current_iteration < max_iterations);
}

double JacobiSolver::computeL2Error() {
    double local_l2_error = 0.0;

    int start_orig_row = localToGlobal(0);

    for (int i = 1; i < local_F.rows() - 1; ++i) {
        for (int j = 1; j < n-1; ++j) {
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

int JacobiSolver::localToGlobal(int local_row) const {
    int rows_per_process = n / size;


    // Share the full matrix row information with all processes
    int start_row, end_row, num_rows, additional_row = 0;
    int rest = n % size;

    if (rest > rank) {
        start_row = rank * (rows_per_process + 1) - 1;
        additional_row++;
    } else {
        start_row = rest * (rows_per_process + 1) + (rank - rest) * rows_per_process - 1;
    }

    if (rank != 0 and rank != size - 1) {
        // start_row = r*rows_per_process - 1;
        end_row = start_row + rows_per_process + 1 + additional_row;

    }
    else if (rank == 0){
        start_row = 0;
        end_row = start_row + rows_per_process + additional_row;
    }
    else{ //at last rank we cannot have additional rows (otherwise rest is 0)
        end_row = n - 1;
    }
    num_rows = end_row - start_row + 1;

    return start_row + local_row;
}
//prints
void JacobiSolver::printLocalMatrixF() const {
    if (rank == 0) {
        // Rank 0 starts the printing process
        std::cout << "Process " << rank << " local matrix F:" << std::endl;
        std::cout << local_F << std::endl;
        
        if (size > 1) {
            // Signal the next process to start printing
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    } else {
        // Other ranks wait for the signal to print
        MPI_Recv(nullptr, 0, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::cout << "Process " << rank << " local matrix F:" << std::endl;
        std::cout << local_F << std::endl;

        if (rank < size - 1) {
            // Signal the next process to start printing
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    // Ensure all processes complete their printing before proceeding
    MPI_Barrier(MPI_COMM_WORLD);
}

void JacobiSolver::printLocalMatrixU() const {
    if (rank == 0) {
        // Rank 0 starts the printing process
        std::cout << "Process " << rank << " local matrix U:" << std::endl;
        std::cout << local_U << std::endl;
        
        if (size > 1) {
            // Signal the next process to start printing
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    } else {
        // Other ranks wait for the signal to print
        MPI_Recv(nullptr, 0, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::cout << "Process " << rank << " local matrix U:" << std::endl;
        std::cout << local_U << std::endl;

        if (rank < size - 1) {
            // Signal the next process to start printing
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    // Ensure all processes complete their printing before proceeding
    MPI_Barrier(MPI_COMM_WORLD);
}
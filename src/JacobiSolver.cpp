#include "JacobiSolver.hpp"
#include <iostream>

JacobiSolver::JacobiSolver(int rows, int cols, std::function<double(double, double)> func)
    : n(rows), m(cols) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    initializeMatrix(func);

    //now we can initialiaze the local_U matrix with the ssame size of the local_matrix
    local_U = Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(local_matrix.rows(), m);
}

void JacobiSolver::initializeMatrix(std::function<double(double, double)> func) {
    int rows_per_process = n / size;

    //this fills the f matrix with the computed values
    if (rank == 0) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> full_matrix(n, m);
        int count_temp = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j){
                // full_matrix(i, j) = func(static_cast<double>(i) / (n - 1), static_cast<double>(j) / (m - 1));
                full_matrix(i, j) = count_temp++; //for testing purposes

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
                MPI_Send(full_matrix.data() + start_row * m, num_rows * m, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            } else {
                local_matrix = full_matrix.block(start_row, 0, num_rows, m);
            }
        }

    } else {
        int num_rows;
        MPI_Recv(&num_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_matrix.resize(num_rows, m);
        MPI_Recv(local_matrix.data(), num_rows * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U_new_buf(local_U.rows(), m);
    double h = 1.0 / (n - 1);
    double res_sum_squared = 0.0;

    for (int i = (rank >= 0 ? 1 : 0); i < local_matrix.rows() - (rank <= size - 1 ? 1 : 0); ++i) {
        for (int j = 1; j < m - 1; ++j) {
            U_new_buf(i, j) = 0.25 * (local_U(i-1, j) + local_U(i+1, j) + local_U(i, j-1) + local_U(i, j+1) - h*h*local_matrix(i, j));
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
                MPI_Isend(local_U.row(local_U.rows() - 2).data(), m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_request);
                
                Eigen::VectorXd new_last_row(m);
                MPI_Request recv_request;
                MPI_Irecv(new_last_row.data(), m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_request);
                
                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
                local_U.row(local_U.rows() - 1) = new_last_row;
                
                MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            }
        } else if (rank == r + 1) {
            Eigen::VectorXd new_first_row(m);
            MPI_Request recv_request;
            MPI_Irecv(new_first_row.data(), m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_request);
            
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            local_U.row(0) = new_first_row;

            MPI_Request send_request;
            MPI_Isend(local_U.row(1).data(), m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_request);
            
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        }
    }

    //now the local_U is ready for the next iteration

    //now return the residual
    return res_k;

}

void JacobiSolver::solve() {
    double tol = 1e-6;
    int max_iter = 10000;
    double residual;
    int iter = 0;

    double res_k = 0.0;

    do {
        iter++;
        res_k = iterJacobi();
        //aggregate the residuals
        MPI_Allreduce(&res_k, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Iteration " << iter << ", Residual: " << residual << std::endl;
        }
    } while (residual > tol && iter < max_iter);
}

//prints
void JacobiSolver::printLocalMatrixF() const {
    if (rank == 0) {
        // Rank 0 starts the printing process
        std::cout << "Process " << rank << " local matrix F:" << std::endl;
        std::cout << local_matrix << std::endl;
        
        if (size > 1) {
            // Signal the next process to start printing
            MPI_Send(nullptr, 0, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
        }
    } else {
        // Other ranks wait for the signal to print
        MPI_Recv(nullptr, 0, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::cout << "Process " << rank << " local matrix F:" << std::endl;
        std::cout << local_matrix << std::endl;

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
# Define the matrix sizes, number of processes, and multithreading options
matrix_sizes=(128 256 512 1024)
num_procs=(2 3 4 5 6)
multithreading_options=(false true)

# Output CSV file
output_file="../data/performance_results.csv"

# Write the CSV header
echo "MatrixSize,NumProcs,Multithreading,ExecTimeDir,ErrorDir,ExecTimeRob,ErrorRob" >> $output_file

# Run tests
for size in "${matrix_sizes[@]}"; do
    for procs in "${num_procs[@]}"; do
        for mt in "${multithreading_options[@]}"; do
            # Run the C++ program with mpirun
            mpirun -np $procs ./solver $size $mt > output.log
            
            # Extract the results from the output log
            exec_time_dir=$(grep "Reached solution in sec" output.log | head -n 1 | awk '{print $5}')
            error_dir=$(grep "L2 Error" output.log | head -n 1 | awk '{print $3}')
            exec_time_rob=$(grep "Reached solution in sec" output.log | tail -n 1 | awk '{print $5}')
            error_rob=$(grep "L2 Error" output.log | tail -n 1 | awk '{print $3}')
            
            # Write the results to the CSV file
            echo "$size,$procs,$mt,$exec_time_dir,$error_dir,$exec_time_rob,$error_rob" >> $output_file
        done
    done
done
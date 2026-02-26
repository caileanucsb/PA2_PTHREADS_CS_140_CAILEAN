Last name of Student 1:
First name of Student 1:
Email of Student 1:
Last name of Student 2:
First name of Student 2:
Email of Student 2:

See the description of this assignment  for detailed reporting requirements 


Part B

Q2.a List parallel code that uses at most two barrier calls inside the while loop
void work_block(long my_rank) {
  int blocksize = (matrix_dim + thread_count - 1) / thread_count;
  int start = my_rank*blocksize;
  int end = start + blocksize;
  if(end > matrix_dim) end = matrix_dim;
    double local_error = 0.0;

    for (int k = 0; k < no_iterations; k++) {
        local_error = 0.0;
        for (int i = start; i < end; i++) {
            mv_compute(i);
            local_error = fmax(local_error, fabs(vector_x[i] - vector_y[i]));
        }
        local_errors[my_rank] = local_error;
        pthread_barrier_wait(&mybarrier);
        for (int i = start; i < end; i++) {
            vector_x[i] = vector_y[i];
        }
        if (my_rank == 0) {
            global_error = 0.0;
            for (int l = 0; l < thread_count; l++) {
                global_error = fmax(global_error, local_errors[l]);
            }
        }
        pthread_barrier_wait(&mybarrier);
        if (global_error <= ERROR_THRESHOLD) {
            break;
        }
    }
}
void work_blockcyclic(long my_rank) {
    double local_error = 0.0;
    int start = my_rank * cyclic_blocksize;
    int incr = thread_count * cyclic_blocksize;
    for (int k = 0; k < no_iterations; k++) {
        local_error = 0.0;
        for (int block_start = start; block_start < matrix_dim; block_start += incr) {
            int start = block_start;
            int end = start + cyclic_blocksize;
            if (end > matrix_dim) end = matrix_dim;
            for (int i = start; i < end; i++) {
                mv_compute(i);
                local_error = fmax(local_error, fabs(vector_y[i] - vector_x[i]));
            }
        }
        local_errors[my_rank] = local_error;
        pthread_barrier_wait(&mybarrier);
        for (int block_start = start; block_start < matrix_dim; block_start += incr) {
            int start = block_start;
            int end = start + cyclic_blocksize;
            if (end > matrix_dim) end = matrix_dim;
            for (int i = start; i < end; i++) {
                vector_x[i] = vector_y[i];
            }
        }
        if (my_rank == 0) {
            global_error = 0.0;
            for (int l = 0; l < thread_count; l++) {
                global_error = fmax(global_error, local_errors[l]);
            }
        }
        pthread_barrier_wait(&mybarrier);
        if (global_error < ERROR_THRESHOLD) {
          break;
        }
    }
}


Q2.b Report parallel time, speedup, and efficiency for  the upper triangular test matrix case when n=4096 and t=1024. 
Use 2 threads and 4  threads (1 thread per core) under blocking mapping, and block cyclic mapping with block size 1 and block size 16.    
Write a short explanation on why one mapping method is significantly faster than or similar to another.

###EVALUATION DONE ON EXPANSE
Test 12 (Upper block mapping):
1 thread = 0.114755 s (baseline), 2 threads = 0.091434 s → speedup = 1.26×, efficiency = 63%, 4 threads = 0.058924 s → speedup = 1.95×, efficiency = 48.8%.
This test shows good scaling as threads increase, but efficiency drops at higher thread counts due to memory bandwidth limits, scheduling overhead, and cache contention, which prevent linear scaling.

Test 13 (Upper block cyclic, r=1):
1 thread = 0.127596 s (baseline), 2 threads = 0.061241 s → speedup ≈ 1.98×, efficiency ≈ 99%, 4 threads = 0.063129 s → speedup = 2.02×, efficiency = 50.5%.
This configuration provides the best performance and load balance, achieving near-ideal scaling at 2 threads due to good cache locality and evenly distributed work, but shows diminishing returns at 4 threads as memory and synchronization overhead dominate.

Test 14 (Upper block cyclic, r=16):
1 thread = 0.102096 s (baseline), 2 threads = 0.067226 s → speedup = 1.52×, efficiency = 76%, 4 threads = 0.080097 s → speedup = 1.27×, efficiency = 31.8%.
Larger block sizes reduce scalability by increasing cache pressure and reducing load balance quality, causing performance to degrade at higher thread counts.

Overall, the results show that moderate parallelism provides strong performance gains, but scaling beyond two threads is limited by memory bandwidth saturation, cache contention, synchronization overhead, and reduced arithmetic intensity per thread, making linear scaling unattainable for these workloads.
-----------------------------------------------------------------
Part C

1. Report what code changes you made for blasmm.c. 

The BLAS2 matrix–matrix multiplication was implemented using repeated calls to cblas_dgemv(). Since MKL uses column-major storage, the computation is done column-wise, meaning each column of C is computed independently as A multiplied by the corresponding column of B. Pointers to B(:,j) and C(:,j) are created using &B[j*K] and &C_dgemv[j*M], and cblas_dgemv is called with CblasColMajor, CblasNoTrans, M, K, alpha = 1.0, beta = 0.0, and unit strides. No other structural changes were made.


2. Conduct a latency and GFLOPS comparison of the above 3 when matrix dimension N varies as 50, 200, 800, and 1600. 
Run the code in one thread and 8 threads on an AMD CPU server of Expanse.
List the latency and GFLOPs of  each method in each setting.  

Maximum number of threads allowed for MKL: 1
--- Matrix Multiplication Performance Comparison ---
Matrix size: N=50 x N=50

MKL DGEMM : Time 0.001412 sec. GFLOPS 0.17. 0.06x
MKL DGEMV Loop: Time 0.000041 sec. GFLOPS 6.12. 1.93x
Naive 3 loops : Time 0.000079 sec. GFLOPS 3.17. 1.00x

Mid-point verification looks OK: DGEMM=10.4441, DGEMV=10.4441, Naive=10.4441

--- Matrix Multiplication Performance Comparison ---
Matrix size: N=200 x N=200

MKL DGEMM : Time 0.000892 sec. GFLOPS 17.93. 3.61x
MKL DGEMV Loop: Time 0.001275 sec. GFLOPS 12.54. 2.53x
Naive 3 loops : Time 0.003248 sec. GFLOPS 4.93. 1.00x

Mid-point verification looks OK: DGEMM=45.1042, DGEMV=45.1042, Naive=45.1042

--- Matrix Multiplication Performance Comparison ---
Matrix size: N=800 x N=800

MKL DGEMM : Time 0.027184 sec. GFLOPS 37.68. 13.22x
MKL DGEMV Loop: Time 0.084103 sec. GFLOPS 12.19. 4.27x
Naive 3 loops : Time 0.365902 sec. GFLOPS 2.80. 1.00x

Mid-point verification looks OK: DGEMM=201.3727, DGEMV=201.3727, Naive=201.3727

--- Matrix Multiplication Performance Comparison ---
Matrix size: N=1600 x N=1600

MKL DGEMM : Time 0.198244 sec. GFLOPS 41.29. 41.61x
MKL DGEMV Loop: Time 1.621933 sec. GFLOPS 5.06. 5.11x
Naive 3 loops : Time 8.394115 sec. GFLOPS 0.97. 1.00x

Mid-point verification looks OK: DGEMM=405.6966, DGEMV=405.6966, Naive=405.6966


Maximum number of threads allowed for MKL: 8
--- Matrix Multiplication Performance Comparison ---
Matrix size: N=50 x N=50

MKL DGEMM : Time 0.048173 sec. GFLOPS 0.01. 0.00x
MKL DGEMV Loop: Time 0.000047 sec. GFLOPS 5.31. 1.43x
Naive 3 loops : Time 0.000068 sec. GFLOPS 3.66. 1.00x

Mid-point verification looks OK: DGEMM=10.4441, DGEMV=10.4441, Naive=10.4441

--- Matrix Multiplication Performance Comparison ---
Matrix size: N=200 x N=200

MKL DGEMM : Time 0.000571 sec. GFLOPS 28.01. 2.44x
MKL DGEMV Loop: Time 0.001742 sec. GFLOPS 9.19. 0.80x
Naive 3 loops : Time 0.001402 sec. GFLOPS 11.40. 1.00x

Mid-point verification looks OK: DGEMM=45.1042, DGEMV=45.1042, Naive=45.1042

--- Matrix Multiplication Performance Comparison ---
Matrix size: N=800 x N=800

MKL DGEMM : Time 0.012891 sec. GFLOPS 79.42. 7.18x
MKL DGEMV Loop: Time 0.116902 sec. GFLOPS 8.76. 0.79x
Naive 3 loops : Time 0.094117 sec. GFLOPS 10.88. 1.00x

Mid-point verification looks OK: DGEMM=201.3727, DGEMV=201.3727, Naive=201.3727

--- Matrix Multiplication Performance Comparison ---
Matrix size: N=1600 x N=1600

MKL DGEMM : Time 0.058614 sec. GFLOPS 139.71. 25.88x
MKL DGEMV Loop: Time 1.168204 sec. GFLOPS 7.01. 1.30x
Naive 3 loops : Time 1.542881 sec. GFLOPS 5.31. 1.00x

Mid-point verification looks OK: DGEMM=405.6966, DGEMV=405.6966, Naive=405.6966

Explain why when N varies from small to large,  Method 1 with GEMM starts to outperform others. 
The three methods (MKL GEMM, repeated GEMV, and naive triple loop) were tested for N = 50, 200, 800, 1600 using 1 and 8 threads on the AMD Expanse server. For small matrices (N = 50), performance differences are minor and multithreading can be slower due to overhead. As N increases, GEMM clearly outperforms the other methods in both latency and GFLOPS, with the gap becoming large at N = 1600, especially with 8 threads. GEMV shows moderate improvement, while the naive implementation performs worst and scales poorly.

Explanation:
GEMM outperforms the other methods for large matrices because it has higher arithmetic intensity and much better cache reuse. MKL’s GEMM uses blocking, vectorization, and optimized memory access, enabling efficient multi-core scaling. In contrast, GEMV is more memory-bound, and the naive triple loop lacks cache optimizations, resulting in poor locality and limited scalability

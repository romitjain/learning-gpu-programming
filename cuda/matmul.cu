// nvcc -o matmul.o matmul.cu && ./matmul.o

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
A simple kernel to perform matrix multiplication
for square matrices.

I have created 2D arrays in C then flattened them
and sent to the kernel.
After the computation, I am again converting from
flattened 1D array to 2D array before printing the result.

This is more for an educational purpose for me to understand
pointers and 2D arrays in C in a better way!

There can be certain improvements I would want to do:
1. Can we manipulate 2D arrays directly in CUDA?
2. Can this solution be more general? How to make it work with any 2 size of matrix
given they are compatible.
*/

__global__ void matMul(float *A_d, float *B_d, float *C_d, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    C_d[i] = 0;

    int a_row_itr = (int)i/n;
    int b_col_itr = (int)i%n;

    if (i < n*n) {
        for (int k=0; k<n; k ++) {
            int a_index = (k*n)+a_row_itr;
            int b_index = (k*n)+b_col_itr;

            C_d[i] += A_d[a_index] * B_d[b_index];
        }
    }
}

void matMulCPU(float *A_h, float *B_h, float *C_h, int n) {
    for(int i = 0; i < n*n; i ++) {
        C_h[i] = 0;

        int a_row_itr = (int)i/n;
        int b_col_itr = (int)i%n;

        printf("%d\tA row iterator: %d, B column iterator: %d\n", i, a_row_itr, b_col_itr);

        for (int k=0; k<n; k ++) {
            int a_index = (k*n)+a_row_itr;
            int b_index = (k*n)+b_col_itr;

            printf("%d\tA index: %d, B index: %d\n", k, a_index, b_index);

            C_h[i] += A_h[a_index] * B_h[b_index];
        }

        printf("----\n");
    }
}

int main() {
    int N = 10;
    size_t size = N*sizeof(float);

    // Allocate a 2D matrix in host
    float **A_h = (float**)malloc(size);
    float **B_h = (float**)malloc(size);
    float **C_h = (float**)malloc(size);

    for (int i=0; i<N; i++) {
        //Allocating memory for a single row
        A_h[i] = (float*)malloc(size);
        B_h[i] = (float*)malloc(size);

        // Some values to fill the arrays
        for (int j=0; j<N; j++) {
            A_h[i][j] = j + i;
            B_h[i][j] = i + 1;
        }
    }

    // Allocating memory for flattened array in host
    float *flat_A = (float*)malloc(size*N);
    float *flat_B = (float*)malloc(size*N);
    float *flat_C = (float*)malloc(size*N);

    printf("A\n");
    int k = 0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%0.1f ", A_h[i][j]);
            flat_A[k] = A_h[i][j];
            k += 1;
        }
        printf("\n");
    }

    k = 0;
    printf("B\n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%0.1f ", B_h[i][j]);
            flat_B[k] = B_h[i][j];
            k += 1;
        }
        printf("\n");
    }

    // Allocating memory for flattened array in device
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**) &A_d, size*N);
    cudaMalloc((void**) &B_d, size*N);
    cudaMalloc((void**) &C_d, size*N);

    cudaMemcpy(A_d, flat_A, size*N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, flat_B, size*N, cudaMemcpyHostToDevice);

    dim3 threads_per_block(N*N);
    dim3 number_of_blocks(1);

    matMul<<<number_of_blocks, threads_per_block>>>(A_d, B_d, C_d, N);

    // Copy the results back to 1D array
    cudaMemcpy(flat_C, C_d, N*size, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    // CPU version
    // matMulCPU(flat_A, flat_B, flat_C, N);

    k = 0;
    printf("C\n");
    for (int i=0; i<N; i++) {
        C_h[i] = (float*)malloc(size);
        for (int j=0; j<N; j++) {
            // Copy from 1D array to 2D array in host
            C_h[i][j] = flat_C[k];
            k += 1;
            printf("%0.1f ", C_h[i][j]);
        }
        printf("\n");
    }

    // Getting error in this block for freeing memory, need to debug
    // for (int i = 0; i < N; i++) {
    //     free(A_h[i]);
    //     free(B_h[i]);
    //     free(C_h[i]);
    // }
    // free(A_h); free(B_h); free(C_h);
    // free(flat_A); free(flat_B); free(flat_C);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);

    return 0;
}

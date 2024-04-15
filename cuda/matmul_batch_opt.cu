// nvcc -o matmul_batch_opt.o matmul_batch_opt.cu && ./matmul_batch_opt.o

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

/*
This kernel should be faster implementation than matmul_batch.cu.

How?
- How to make use of shared memory to reduce HBM reads?
- How to make the most of the hardware architecture?

It takes a batch of data: (B x N x Din)
and a weight matrix: (Din X Dout)
and outputs: (B x N x Dout)
*/


__global__ void matMul(
    float* X,
    float* W,
    float* OO,
    int B,
    int N,
    int D_in,
    int D_out
) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int width = blockDim.x;
    int height = blockDim.y;

    int offset = width*ty + tx;

    if (offset < N*N) {
        float sum = 0.0;

        for (int i = 0; i < D_in; i++) {
            sum += X[width*ty + i] * W[tx + height*i];
        }

        OO[offset] = sum;
    }

}

__global__ void matMulTiled(
    float* X,
    float* W,
    float* OO,
    int B,
    int N,
    int D_in,
    int D_out,
    int SMALL_N
) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int width = blockDim.x;
    int height = blockDim.y;

    int offset = width*ty + tx;

    if (offset < N*N) {
        __shared__ float sum;

        for (int i = 0; i < SMALL_N; i++) {
            sum += X[width*ty + i] * W[tx + height*i];
        }

        OO[offset] = sum;
    }

}

void display_3dmat(char* name, float* M, int A, int B, int C) {
    // A x B x C
    printf("\nMatrix: %s\n", name);

    for (int i=0; i<A; i++) {
        for (int j=0; j<B; j++) {
            for (int k=0; k<C; k++) {
                // skip batches, skip rows, skip columns
                int offset = i*B*C + j*C + k;
                printf("%0.1f, ", M[offset]);
            }
            printf("\n");
        }
        printf("--\n");
    }
}

void display_2dmat(char* name, float* M, int A, int B) {
    // A x B
    printf("\nMatrix: %s\n", name);

    for (int i=0; i<A; i++) {
        for (int j=0; j<B; j++) {
            int offset = i*B + j;
            printf("%0.1f, ", M[offset]);
        }
        printf("\n");
    }
}


void matMulCPU(float *A_h, float *B_h, float *C_h, int n) {
    for(int i = 0; i < n*n; i ++) {
        C_h[i] = 0;

        int a_row_itr = (int)i/n;
        int b_col_itr = (int)i%n;

        for (int k=0; k<n; k ++) {
            int a_index = k+n*a_row_itr;
            int b_index = k*n+b_col_itr;

            C_h[i] += A_h[a_index] * B_h[b_index];
        }
    }
}

inline int cdiv(int num, int den) {
    return (num + den - 1)/den;
}

int main() {
    int B = 1;
    int N = 32;
    int D_in = 32;
    int D_out = 32;
    // N*D_out needs to be < 1024

    srand(42);

    float *X = (float*)malloc(B*N*D_in*sizeof(float));
    float *W = (float*)malloc(D_in*D_out*sizeof(float));
    float *O = (float*)malloc(B*N*D_out*sizeof(float));
    float *OC = (float*)malloc(B*N*D_out*sizeof(float));

    // Allocating with random numbers
    for (int i=0; i<B*N*D_in; i++) {
        X[i] = rand() % 10;
    }
    for (int i=0; i<D_in*D_out; i++) {
        W[i] = rand() % 10;
    }

    struct timeval start_cpu, end_cpu;

    gettimeofday(&start_cpu, NULL);
    matMulCPU(X, W, OC, N);
    gettimeofday(&end_cpu, NULL);
    float micros = end_cpu.tv_usec - start_cpu.tv_usec;
    printf("CPU version: %0.6f seconds\n", micros/1000);

    // Allocating memory for flattened array in device
    float *d_X, *d_W, *d_O;

    cudaMalloc((void**) &d_X, B*N*D_in*sizeof(float));
    cudaMalloc((void**) &d_W, D_in*D_out*sizeof(float));
    cudaMalloc((void**) &d_O, B*N*D_out*sizeof(float));

    cudaMemcpy(d_X, X, B*N*D_in*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, D_in*D_out*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads_per_block(N, N);
    dim3 number_of_blocks(B);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    matMul<<<number_of_blocks, threads_per_block>>>(
        d_X,
        d_W,
        d_O,
        B,
        N,
        D_in,
        D_out
    );

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nGPU version: %0.6fms\n", milliseconds);

    // Copy the results back to 1D array
    cudaMemcpy(O, d_O, B*N*D_out*sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    for (int i = 0;i<N*N; i++) {
        if (OC[i] != O[i]) {
            printf("Values not matching, %0.1f %0.1f\n", OC[i], O[i]);
        }
    }

    // display_3dmat("X", X, B, N, D_in);
    // display_2dmat("W", W, D_in, D_out);
    // display_3dmat("O", O, B, N, D_out);
    // display_3dmat("OC", OC, B, N, D_out);

    free(X); free(W); free(O); free(OC);
    cudaFree(d_X); cudaFree(d_W); cudaFree(d_O);

    return 0;
}

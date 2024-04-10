// nvcc -o matmul_batch.o matmul_batch.cu && ./matmul_batch.o

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
This kernel should be more general than matmul.cu

It takes a batch of data: (B x N x Din)
and a weight matrix: (Din X Dout)

and produces: (B x N x Dout)

First, we flatten the input arrays
Call the kernel with appropriate grid and block size
Compute the data for the output matrix

This however is still slow because for every element
of the output matrix, we have to read multiple data from
HBM
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

    int batch = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    int out_offset = N*D_out*batch + row*D_out + col;

    if ((batch < B) && (col < D_out) && (row < N)) {
        float sum = 0.0f;
        for (int i = 0; i < D_in; i++) {
            sum += X[N * D_in * batch + row * D_in + i] * W[i * D_out + col];
        }
        OO[out_offset] = sum;
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
                printf("%0.1f ", M[offset]);
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
            printf("%0.1f ", M[offset]);
        }
        printf("\n");
    }
}

int main() {
    int B = 2;
    int N = 5;
    int D_in = 4;
    int D_out = 8;

    srand(42);

    float *X = (float*)malloc(B*N*D_in*sizeof(float));
    float *W = (float*)malloc(D_in*D_out*sizeof(float));
    float *O = (float*)malloc(B*N*D_out*sizeof(float));

    // Allocating with random numbers
    for (int i=0; i<B*N*D_in; i++) {
        X[i] = rand() % 10;
    }

    for (int i=0; i<D_in*D_out; i++) {
        W[i] = rand() % 10;
    }

    // Allocating memory for flattened array in device
    float *d_X, *d_W, *d_O;

    cudaMalloc((void**) &d_X, B*N*D_in*sizeof(float));
    cudaMalloc((void**) &d_W, D_in*D_out*sizeof(float));
    cudaMalloc((void**) &d_O, B*N*D_out*sizeof(float));

    cudaMemcpy(d_X, X, B*N*D_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, D_in*D_out, cudaMemcpyHostToDevice);

    /*
    each block should ideally process
    a single batch
    inside each block should ideally process
    a chunk of X and complete W
    */

    dim3 threads_per_block(D_out, N);
    dim3 number_of_blocks(B);

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

    // Copy the results back to 1D array
    cudaMemcpy(O, d_O, B*N*D_out, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    display_3dmat("X", X, B, N, D_in);
    display_2dmat("W", W, D_in, D_out);
    display_3dmat("O", O, B, N, D_out);

    free(X); free(W); free(O);
    cudaFree(d_X); cudaFree(d_W); cudaFree(d_O);

    return 0;
}

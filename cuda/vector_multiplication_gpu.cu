#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> //Doubt: Where does it find this file?

/*
A simple kernel to multiply two vectors using pre decided number of threads
Doubt: How to find the maximum number of threads available?
*/

__global__ void vecMultiply(float *d_A, float *d_B, float *d_C, float *d_D, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N) {
        d_C[i] = d_A[i] * d_B[i];
        d_D[i] = threadIdx.x;
    }
}

int main() {
    int N = 1000;

    float *A = (float *)malloc(N* sizeof(float));
    float *B = (float *)malloc(N* sizeof(float));
    float *C = (float *)malloc(N* sizeof(float));
    float *D = (float *)malloc(N* sizeof(float));

    float *d_A; float *d_B; float *d_C; float *d_D;

    int size = N * sizeof(float);

    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);
    cudaMalloc((void **) &d_D, size);

    for (int i=0; i<N; i++) {
        // A[i] = i+1;
        B[i] = i-1;
    }

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int block_size = 500;
    int num_blocks = N/block_size;

    vecMultiply <<<num_blocks, block_size>>> (d_A, d_B, d_C, d_D, N);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    for (int i = 0; i < N; i++) {
        printf("%0.1f\t%0.1f\n", D[i], C[i]);
    }

    free(A); free(B); free(C);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;

}
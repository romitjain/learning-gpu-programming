#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vecAddKernel(float *A_h, float *B_h, float *C_h, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n) {
        C_h[i] = A_h[i] + B_h[i];
    }

}

int main() {
    int N = 100;

    // Allocating floats in CPU memory
    float *A_h = (float *)malloc(N * sizeof(float));
    float *B_h = (float *)malloc(N * sizeof(float));
    float *C_h = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        A_h[i] = i*1;
        B_h[i] = i*2;
    }

    float *A_d, *B_d, *C_d;

    float size = N*sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    int gridsize(1);
    int blocksize(100);

    vecAddKernel<<<gridsize, blocksize>>>(A_d, B_d, C_d, N);

    cudaMemcpy(C_h, C_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    printf("here\n\n");

    for (int i = 0; i < N; i++) {
        printf("%0.1f\n", C_h[i]);
    }

    free(A_h);
    free(B_h);
    free(C_h);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}

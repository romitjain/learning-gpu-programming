/*
nvcc -std=c++17 -O3 -arch=native \
     -I$LIBTORCH/include \
     -I$LIBTORCH/include/torch/csrc/api/include \
     -diag-suppress=2464 \
     -o rms.o rms.cu ../utils.c
*/

#include "../utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

/*
The most straightforward implementation of RMS Layernorm in CUDA.
RMSNorm

Formula:
    y = x/RMS(x) * gamma
    RMS = sqrt(mean(x^2) + eps)
*/
__global__ void singleWarpRMS(
    float *data,
    float *out,
    int B,
    int N,
    int V,
    float eps,
    float gamma)
{
    int row_num = blockDim.y * blockIdx.y + threadIdx.y;

    if (row_num >= N) return;

    int batch_num = blockIdx.x;
    int row_offset = batch_num * N * V + V * row_num;
    
    int tx = threadIdx.x;
    float sumVal = 0.0;

    // Calculate the RMS first
    // Do summation for each thread
    for (int i=tx; i<V;i += blockDim.x) {
        float val = data[row_offset + i];
        sumVal += val * val;
    }
    // Then for each thread combine the values
    // Since we only have a single warp, we can just do a xor sync once
    for (int offset=16; offset>0;offset/=2) {
        sumVal = sumVal + __shfl_xor_sync(0xffffffff, sumVal, offset, 32);
    }

    sumVal /= V;
    sumVal += eps;
    sumVal = sqrtf(sumVal);

    for (int i=tx; i<V; i += blockDim.x) {
        out[row_offset + i] = (data[row_offset + i]/sumVal) * gamma;
    }
}

void launch_rms(
    torch:: Tensor data,
    torch::Tensor out,
    float eps,
    float gamma,
    int version
) {
    const int B = data.size(0);
    const int N = data.size(1);
    const int V = data.size(2);

    if (version == 1)
    {
        const int THREADS_X = 32;
        const int THREADS_Y = 8;

        dim3 block(THREADS_X, THREADS_Y, 1);
        dim3 grid(B, (N + THREADS_Y - 1) / THREADS_Y, 1);

        singleWarpRMS<<<grid, block>>>(
            data.data_ptr<float>(),
            out.data_ptr<float>(),
            B, N, V, eps, gamma);
    }
}

int main() {
    int B = 1;
    int N = 10;
    int V = 45;
    float eps = 1e-5;
    float gamma = 1;

    srand(420);

    size_t mat_size = B*N*V;

    float *out = (float *)malloc(mat_size * sizeof(float));
    float *data = (float *)malloc(mat_size * sizeof(float));

    fill_random(data, mat_size, 5);
    display_3dmat("data", data, B, N, V);

    // Device variable 
    float *d_data, *d_out;

    cudaMalloc((void **) &d_data, mat_size * sizeof(float));
    cudaMalloc((void **) &d_out, mat_size * sizeof(float));

    cudaMemcpy(d_data, data, mat_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(32, 8, 1);
    dim3 grid(B, (N + 7) / 8, 1);

    singleWarpRMS<<<grid, block_size>>>(
        d_data, d_out, B, N, V, eps, gamma);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(out, d_out, mat_size * sizeof(float), cudaMemcpyDeviceToHost);

    display_3dmat("out", out, B, N, V);

    free (out);
    free(data);
    cudaFree (d_out);
    cudaFree(d_data);

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_forward", &launch_rms, "RMSNorm (CUDA)");
}
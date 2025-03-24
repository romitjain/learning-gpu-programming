// nvcc -o softmax.o softmax.cu utils.c -diag-suppress 2464 && ./softmax.o

#include "../utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <torch/extension.h>

/*
Batched softmax
B x N x V
*/

#define cudaErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void naivesoftMax(
    float *data,
    float *out,
    int B,
    int N,
    int V)
{

    if (threadIdx.y >= N) return;

    int batch = blockIdx.x;
    int row_offset = batch * N * V + V * threadIdx.y;
    int out_offset = batch * N * V + V * threadIdx.y + threadIdx.x;

    int ty = threadIdx.y;

    if (ty < N)
    {
        float row_local_max = 0;

        for (int i = 0; i < V; i++)
        {
            row_local_max = fmaxf(data[row_offset + i], row_local_max);
        }

        out[out_offset] = row_local_max;
    }
}

__global__ void softMax(
    float* data,
    float* out,
    int B,
    int N,
    int V
) 
{
    if (threadIdx.y >= N)
        return;

    int batch = blockIdx.x;
    int row_offset = batch * N * V + V * threadIdx.y;

    // Manually fixing the stride
    int tx = threadIdx.x;
    float maxVal = -FLT_MAX;

    // Each thread loops the array
    // strided by warp size and finds max
    for (int i = tx; i < V; i+=blockDim.x)
    {
        maxVal = fmaxf(data[row_offset + i], maxVal);
    }

    // Now we have a warp which has max val
    // do register level reduction
    for (int offset = 16; offset > 0; offset /= 2)
    {
        maxVal = fmaxf(maxVal, __shfl_xor_sync(0xffffffff, maxVal, offset, 32));
    }

    float sumVal = 0.0;
    // Same for finding the denominator
    for (int i = tx; i < V; i += blockDim.x)
    {
        sumVal = sumVal + __expf(data[row_offset + i] - maxVal);
    }
    for (int offset = 16; offset > 0; offset /=2)
    {
        sumVal = sumVal + __shfl_xor_sync(0xffffffff, sumVal, offset, 32);
    }

    out[row_offset + tx] = __expf(data[row_offset + tx] - maxVal)/sumVal;
}

void launch_softmax(torch::Tensor data, torch::Tensor out)
{
    const int B = data.size(0);
    const int N = data.size(1);
    const int V = data.size(2);

    const int threads = 32;
    dim3 block(threads, N, 1);
    dim3 grid(B, 1, 1);

    softMax<<<grid, block>>>(
        data.data_ptr<float>(),
        out.data_ptr<float>(),
        B, N, V
    );
}

int main() {

    int B = 1;
    int N = 1;
    int V = 32;

    srand(69);

    size_t mat_size = B*N*V;

    float *out = (float *)malloc(mat_size * sizeof(float));
    float *data = (float *)malloc(mat_size * sizeof(float));

    fill_random(data, mat_size, 5);
    display_3dmat("data", data, B, N, V);

    float *d_data, *d_out;

    cudaMalloc((void **) &d_data, mat_size * sizeof(float));
    cudaMalloc((void **) &d_out, mat_size * sizeof(float));

    cudaMemcpy(d_data, data, mat_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, mat_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(32, N, 1);
    dim3 grid(B, 1, 1);

    softMax<<<grid, block_size>>>(
        d_data,
        d_out,
        B,
        N,
        V
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(out, d_out, mat_size*sizeof(float), cudaMemcpyDeviceToHost);

    display_3dmat("out", out, B, N, V);

    free (out); free(data);
    cudaFree(d_out);
    cudaFree(d_data);

    return 0;
}
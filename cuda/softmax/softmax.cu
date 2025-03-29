// nvcc -o softmax.o softmax.cu utils.c -diag-suppress 2464 && ./softmax.o

#include "../utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>

/*
Batched softmax
B x N x V

Only 4 tricks are used to reach PyTorch performance:
1. Use warp level reduction to find the max value in a warp.
2. Then use block level reduction to find the global max value
3. Use float4 values to load 4 floats at a time and store 4 floats at a time
4. Use pragmas to unroll the loops
5. Suggested by GPT: Use cta.sync() and __restrict__ but it doesn't give any performance boost
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

__global__ void singleWarpSoftMax(
    float* data,
    float* out,
    int B,
    int N,
    int V
) 
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= N)
        return;

    int batch = blockIdx.x;
    int row_offset = batch * N * V + V * row;

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

    for (int i = tx; i < V; i += blockDim.x)
    {
    	out[row_offset + i] = __expf(data[row_offset + i] - maxVal)/sumVal;
    }
}

__global__ void multiWarpSoftMax(
    float *__restrict__ data,
    float *__restrict__ out,
    int B,
    int N,
    int V)
{
    namespace cg = cooperative_groups;
    cg::thread_block cta = cg::this_thread_block();

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= N) return;

    int batch = blockIdx.x;
    int row_offset = batch * N * V + V * row;
    
    // This is the shared memory where each warp will store the max value
    int num_warps = (blockDim.x + 31) / 32;
    extern __shared__ float sdata[];
    float *warp_max = sdata;
    float *warp_sum = &sdata[num_warps];

    int tx = threadIdx.x;

    // Step 1: This value is stored in register
    // Each thread loops the array
    // strided by warp size and finds max
    // while making sure that memory access is coalesced
    float local_max = -FLT_MAX;
#pragma unroll 8
    for (int i = tx; i < V/4; i += blockDim.x)
    {
        float4 val = reinterpret_cast<float4 *>(&data[row_offset + i * 4])[0];
        local_max = fmaxf(val.x, local_max);
        local_max = fmaxf(val.y, local_max);
        local_max = fmaxf(val.z, local_max);
        local_max = fmaxf(val.w, local_max);
    }

    // Step 2:Now we have a warp which has max val - do register level reduction
    // This happens in register cache: https://developer.nvidia.com/blog/register-cache-warp-cuda/
#pragma unroll 8
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset, 32));
    }

    // Step 3: Store the max value in shared memory, we only need to store one value per warp
    // Hence mod 32 check
    if (tx % 32 == 0)
    {
        warp_max[tx / 32] = local_max;
    }

    cta.sync();

    // Step 4: Reduce the max value across the shared memory
    // I'll do it with a single warp, but then I will first have to load the values
    // from the shared memory to the register
    float global_max = -FLT_MAX;
    if (tx < num_warps)
    {
        global_max = warp_max[tx];
    }
    if (tx < 32)
    {
#pragma unroll 8
        for (int offset = 16; offset > 0; offset /= 2)
        {
            global_max = fmaxf(global_max, __shfl_xor_sync(0xffffffff, global_max, offset, 32));
        }
    }
    if (tx == 0)
    {
        warp_max[0] = global_max;
    }
    cta.sync();
    global_max = warp_max[0];

    // Same for finding the denominator
    float local_sum = 0.0;
#pragma unroll 8
    for (int i = tx; i < V/4; i += blockDim.x)
    {
        float4 val = reinterpret_cast<float4 *>(&data[row_offset + i * 4])[0];
        local_sum = local_sum + __expf(val.x - global_max);
        local_sum = local_sum + __expf(val.y - global_max);
        local_sum = local_sum + __expf(val.z - global_max);
        local_sum = local_sum + __expf(val.w - global_max);
    }
#pragma unroll 8
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_sum = local_sum + __shfl_xor_sync(0xffffffff, local_sum, offset, 32);
    }
    if (tx % 32 == 0)
    {
        warp_sum[tx / 32] = local_sum;
    }
    cta.sync();

    float global_sum = 0.0;
    if (tx < num_warps)
    {
        global_sum = warp_sum[tx];
    }
    if (tx < 32)
    {
#pragma unroll 8
        for (int offset = 16; offset > 0; offset /= 2)
        {
            global_sum = global_sum + __shfl_xor_sync(0xffffffff, global_sum, offset, 32);
        }
    }
    if (tx == 0)
    {
        warp_sum[0] = global_sum;
    }
    cta.sync();
    global_sum = warp_sum[0];

#pragma unroll 8
    for (int i = tx; i < V/4; i += blockDim.x)
    {
        float4 val = reinterpret_cast<float4 *>(&data[row_offset + i * 4])[0];
        val.x = __expf(val.x - global_max) / global_sum;
        val.y = __expf(val.y - global_max) / global_sum;
        val.z = __expf(val.z - global_max) / global_sum;
        val.w = __expf(val.w - global_max) / global_sum;
        reinterpret_cast<float4 *>(&out[row_offset + i * 4])[0] = val;
    }
}


void launch_softmax(torch::Tensor data, torch::Tensor out, int version)
{
    const int B = data.size(0);
    const int N = data.size(1);
    const int V = data.size(2);

    if (version == 0)
    {
        dim3 block(32, 1, 1);
        dim3 grid(B, N, 1);

        naivesoftMax<<<grid, block>>>(
            data.data_ptr<float>(),
            out.data_ptr<float>(),
            B, N, V
        );
    }
    else if (version == 1)
    {
        const int THREADS_X = 32;
        const int THREADS_Y = 8;

        dim3 block(THREADS_X, THREADS_Y, 1);
        dim3 grid(B, (N + THREADS_Y - 1) / THREADS_Y, 1);

        singleWarpSoftMax<<<grid, block>>>(
            data.data_ptr<float>(),
            out.data_ptr<float>(),
            B, N, V
        );
    }
    else if (version == 2)
    {
        int THREADS_X = 32;

        if (V > 512 && V <= 2048)
        {
            THREADS_X *= 4;
        }
        else if (V > 2048)
        {
            THREADS_X *= 32;
        }

        int THREADS_Y = 8;
        if (V > 512) {
            THREADS_Y = ceil(1024 / THREADS_X);
        }
        dim3 block(THREADS_X, THREADS_Y, 1);
        dim3 grid(B, (N + THREADS_Y - 1) / THREADS_Y, 1);

        multiWarpSoftMax<<<grid, block>>>(
            data.data_ptr<float>(),
            out.data_ptr<float>(),
            B, N, V
        );
    }

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

    singleWarpSoftMax<<<grid, block_size>>>(
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

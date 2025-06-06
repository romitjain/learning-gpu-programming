/*
nvcc -std=c++17 -O3 -arch=native \
     -I$LIBTORCH/include \
     -I$LIBTORCH/include/torch/csrc/api/include \
     -I$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
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

/*
So how do I optimize it further?

One obvious way is to use multiple warps
But then there will be two reductions?
One within warps and then one for all warps

GPT suggested this code:

Latency of __shfl_xor_sync tree	• Use warp_reduce_sum<float, \/*unroll=\/
    false > (x)helpers from CUTLASS / TensorRT - LLM,
    which emit optimised PTX.
• If device ≥ SM80, try __reduce_add_sync(mask, x)(hardware shuffle + reduce).


Each warps accumulates in fp32 But stores in fp16
- should shed some weight of the movement

Use rsqrtf(mean)
*/

__global__ void multiWarpRMS(float *data, float *out, int B, int N, int V, float eps, float gamma)
{
    int row_num = blockDim.y * blockIdx.y + threadIdx.y;

    if (row_num >= N) return;

    int batch_num = blockIdx.x;
    int row_offset = batch_num * N * V + V * row_num;
    
    int tx = threadIdx.x;
    float sumVal = 0.0;

    extern __shared__ float sdata[];
    float *warp_sum = sdata;
    int num_warps = (blockDim.x + 31)/32;
    int lane = tx & 31;
    int warp_id = tx >> 5;  // Faster than tx/32

    bool aligned = (((row_offset | V) & 3) == 0); // 16‑byte aligned row **and** V multiple of 4

    // Do summation for each thread
#pragma unroll 8
    for (int i=tx; i<V/4;i += blockDim.x) {
        if (aligned) {
            float4 val = reinterpret_cast<float4 *>(&data[row_offset+i*4])[0];

            sumVal += val.x * val.x;
            sumVal += val.y * val.y;
            sumVal += val.z * val.z;
            sumVal += val.w * val.w;
        } else {
            int base = row_offset + i*4;
            float v0 = data[base + 0];
            float v1 = data[base + 1];
            float v2 = data[base + 2];
            float v3 = data[base + 3];
            sumVal += v0*v0 + v1*v1 + v2*v2 + v3*v3;
        }
    }

    // Required in case there are tail elements remaining after V/4
    if (!aligned) {
        for (int idx = V & ~3u; idx < V; idx += blockDim.x) {
            if (idx + tx < V) {
                float v = data[row_offset + idx + tx];
                sumVal += v * v;
            }
        }
    }

    // Then for each warp reduce the values
    for (int offset=16; offset>0; offset/=2) {
        sumVal = sumVal + __shfl_xor_sync(0xffffffff, sumVal, offset, 32);
    }

    if (lane == 0) {
        warp_sum[warp_id] = sumVal;
    }

    __syncthreads();

    // Now only use 1st warp to do the reduction
    float global_sum = 0.0f;
    if (warp_id == 0) {
        global_sum = (lane < num_warps) ? warp_sum[lane] : 0.0f;

        unsigned mask = __ballot_sync(0xffffffff, lane < num_warps);

#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            global_sum += __shfl_down_sync(mask, global_sum, offset);

        // Store the value in the first thread only
        if (lane == 0)
            warp_sum[0] = global_sum;
    }

    __syncthreads();
    global_sum = warp_sum[0];

    global_sum /= V;
    global_sum += eps;

    float rsqrt_val = rsqrtf(global_sum);

    for (int i=tx; i<V/4; i += blockDim.x) {
        if (aligned) {
            float4 val = reinterpret_cast<float4 *>(&data[row_offset + i * 4])[0];

            val.x = (val.x * rsqrt_val) * gamma;
            val.y = (val.y * rsqrt_val) * gamma;
            val.z = (val.z * rsqrt_val) * gamma;
            val.w = (val.w * rsqrt_val) * gamma;

            reinterpret_cast<float4 *>(&out[row_offset+i*4])[0] = val;
        } else {
            int base = row_offset + i*4;
            if (base + 0 < row_offset + V) out[base + 0] = data[base + 0] * rsqrt_val * gamma;
            if (base + 1 < row_offset + V) out[base + 1] = data[base + 1] * rsqrt_val * gamma;
            if (base + 2 < row_offset + V) out[base + 2] = data[base + 2] * rsqrt_val * gamma;
            if (base + 3 < row_offset + V) out[base + 3] = data[base + 3] * rsqrt_val * gamma;
        }
    }

    if (!aligned) {
        for (int idx = V & ~3u; idx < V; idx += blockDim.x) {
            if (idx + tx < V) {
                int base = row_offset + idx + tx;
                out[base] = data[base] * rsqrt_val * gamma;
            }
        }
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

        const int THREADS_Y = 1;
        // if (V > 512)
        // {
        //     THREADS_Y = ceil(1024 / THREADS_X);
        // }
        dim3 block(THREADS_X, THREADS_Y, 1);
        dim3 grid(B, (N + THREADS_Y - 1) / THREADS_Y, 1);

        int num_warps = (THREADS_X + 31) / 32;
        size_t shared_mem_size = num_warps * sizeof(float);

        multiWarpRMS<<<grid, block, shared_mem_size>>>(
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

#include <stdio.h>
#include <cuda_runtime.h>

/*
Optimizations
1. Load the data required by block in SMEM in the start
2. Send 1/k ahead of time
3. Pragma unroll
4. Vectorized loads
*/

__global__ void pooling_1d(
    const float *__restrict__ input,
    float *__restrict__ output,
    const int k,
    const int s,
    const int p,
    const size_t in_size,
    const size_t out_size,
    float inv_k)
{
    // The strategy is to load the data from HBM to SMEM
    // once for every block
    extern __shared__ float sh_input[];

    int block_start = blockIdx.x * blockDim.x;
    int gmem_block_read_start = block_start * s - p;
    int num_elements_to_load = s * (blockDim.x - 1) + k;

    int tidx = threadIdx.x;

#pragma unroll
    for (int i = tidx; i < num_elements_to_load / 4; i += blockDim.x)
    {
        int index = gmem_block_read_start + i * 4;
        if ((index >= 0) && ((index % 4) == 0) && ((index + 3) < in_size))
        {
            ((float4 *)sh_input)[i] = reinterpret_cast<const float4 *>(&input[index])[0];
        }
        else
        {
#pragma unroll 4
            // Fallback: load element-by-element.
            for (int j = 0; j < 4; ++j)
            {
                int idx = index + j;
                sh_input[i * 4 + j] = (idx >= 0 && idx < in_size) ? input[idx] : 0.0f;
            }
        }
    }

    int remaining_loads = 4 * (num_elements_to_load / 4);

#pragma unroll
    for (int i = remaining_loads + tidx; i < num_elements_to_load; i += blockDim.x)
    {
        int idx = gmem_block_read_start + i;
        sh_input[i] = (idx >= 0 && idx < in_size) ? input[idx] : 0.0f;
    }

    __syncthreads();

    int out_idx = block_start + tidx;
    if (out_idx >= out_size)
        return;

    float accum = 0.0;

    int shmem_start_idx_for_thread = (s * out_idx - p) - gmem_block_read_start;

#pragma unroll 8
    for (int i = 0; i < k; ++i)
    {
        int current_sh_idx = shmem_start_idx_for_thread + i;
        accum += sh_input[current_sh_idx];
    }

    output[out_idx] = accum * inv_k;
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(
    const float *input, int kernel_size, int stride, int padding, float *output, size_t H)
{
    size_t output_size = floor((H + 2 * padding - kernel_size) / stride) + 1;
    int threads_x = 128;
    if (H == 2097152 || H == 4194304)
    {
        threads_x = 512;
    }

    int blocks_x = (output_size + threads_x - 1) / threads_x;

    dim3 block_size(threads_x, 1, 1);
    dim3 grid(blocks_x, 1, 1);

    int shmem_elems_per_block = stride * (threads_x - 1) + kernel_size;
    size_t shmem_bytes = shmem_elems_per_block * sizeof(float);
    float inv_k = 1.0f / (float)(kernel_size);

    pooling_1d<<<grid, block_size, shmem_bytes>>>(
        input,
        output,
        kernel_size,
        stride,
        padding,
        H,
        output_size,
        inv_k);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }
}

int main()
{
    int H = 20;
    int kernel_size = 7;
    int stride = 4;
    int padding = 3;

    float input[H];
    // Fill random
    for (int i = 0; i < H; i++)
    {
        input[i] = rand() % 10;
    }

    int output_size = floor((H + 2 * padding - kernel_size) / stride) + 1;
    float output[output_size];

    solution(input, kernel_size, stride, padding, output, H);
    for (int i = 0; i < 10; i++)
    {
        printf("%f ", output[i]);
    }

    return 0;
}

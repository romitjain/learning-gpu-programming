#include <stdio.h>
#include <cuda_runtime.h>

/*
Optimizations
1. Load the data required by block in SMEM in the start
2. Send 1/k ahead of time
3. Pragma unroll
*/

__global__ void pooling_1d(
    float *__restrict__ input,
    float *__restrict__ output,
    int k,
    int s,
    int p,
    size_t in_size,
    size_t out_size,
    float inv_k)
{
    // The strategy is to load the data from HBM to SMEM
    // once for every block
    extern __shared__ float sh_input[];

    int block_start = blockIdx.x * blockDim.x;
    int gmem_block_read_start = block_start * s - p;
    int num_elements_to_load = s * (blockDim.x - 1) + k;

    int tidx = threadIdx.x;
    for (int i = tidx; i < num_elements_to_load; i += blockDim.x)
    {
        int gmem_idx = gmem_block_read_start + i;
        if (gmem_idx >= 0 && gmem_idx < in_size)
        {
            sh_input[i] = input[gmem_idx];
        }
        else
        {
            sh_input[i] = 0.0f;
        }
    }

    __syncthreads();

    int out_idx = block_start + tidx;
    if (out_idx >= out_size) return;

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
    const float* input, int kernel_size, int stride, int padding, float* output, size_t H
) {
    size_t output_size = floor((H + 2 * padding - kernel_size)/stride) + 1;
    float *d_input, *d_output;

    cudaMalloc((void **) &d_input, H * sizeof(float));
    cudaMalloc((void **) &d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input, H * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, output_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads_x = 1024;
    int blocks_x = (output_size + threads_x - 1) / threads_x;

    dim3 block_size(threads_x, 1, 1);
    dim3 grid(blocks_x, 1, 1);

    int shmem_elems_per_block = stride * (threads_x - 1) + kernel_size;
    size_t shmem_bytes = shmem_elems_per_block * sizeof(float);
    float inv_k = 1.0f/(float)(kernel_size);

    pooling_1d<<<grid, block_size, shmem_bytes>>>(
        d_input,
        d_output,
        kernel_size,
        stride,
        padding,
        H,
        output_size,
        inv_k);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(output, d_output, output_size*sizeof(float), cudaMemcpyDeviceToHost);
}

int main() {
    float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float output[10];

    solution(input, 3, 1, 0, output, 10);
    for (int i = 0; i < 10; i++) {
        printf("%f ", output[i]);
    }

    return 0;
}
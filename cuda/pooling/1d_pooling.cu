#include <stdio.h>
#include <cuda_runtime.h>

__global__ void pooling_1d(
    float *__restrict__ input,
    float *__restrict__ output,
    int k,
    int s,
    int p,
    size_t in_size,
    size_t out_size)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size) return;

    float accum = 0.0;
    int in_idx = s*out_idx-p;

    for (int i = in_idx; i < in_idx+k; i ++) {
        if (i < in_size && i >= 0) {
            accum += input[i];
        }
    }

    output[out_idx] = accum/((float)(k));
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

    pooling_1d<<<grid, block_size>>>(
        d_input,
        d_output,
        kernel_size,
        stride,
        padding,
        H,
        output_size
    );

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
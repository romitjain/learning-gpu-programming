// WIP
// 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void rgb2gray(
    unsigned char* output,
    unsigned char* input,
    int width,
    int height) {

    int x = threadIdx.x;
    int y = threadIdx.y;

    if ((x < width) & (y < height)) {
        output[x][y] = 0.299 * input[x][y] + 0.587 * input[x][y] + 0.144 * input[x][y];
    }
}

torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(1);

    rgb2gray<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
    }

    return result;
}
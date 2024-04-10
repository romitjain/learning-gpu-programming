"""
Adopted from: https://github.com/cuda-mode/lectures/blob/main/lecture2/rgb_to_grayscale/rgb_to_grayscale.py
Makes it easy to launch kernels
"""

import torch
from pathlib import Path
from torchvision.io import read_image, write_png
from torchvision import transforms
from torch.utils.cpp_extension import load_inline

def compile_extension(kernel):
    cuda_source = Path(kernel).read_text()
    cpp_source = "torch::Tensor rgb_to_grayscale(torch::Tensor image);"

    # Load the CUDA kernel as a PyTorch extension
    ext = load_inline(
        name="rgb_to_grayscale_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_grayscale"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )
    return ext

def main(args):
    ext = compile_extension("./rgb2gray.cu")

    x = read_image("~/Pictures/profile.jpg").permute(1, 2, 0).cuda()
    x = transforms.functional.resize(x, (256, 256))
    print("Mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)

    assert x.dtype == torch.uint8

    y = ext.rgb_to_grayscale(x)

    print("Output image:", y.shape, y.dtype)
    print("Mean", y.float().mean())
    write_png(y.permute(2, 0, 1).cpu(), "output.png")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--kernel')

    args = parser.parse_args()
    main(args)
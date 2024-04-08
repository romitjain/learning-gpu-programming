# learning-gpu-programming

Learnings and experimentation with GPU programming

- cuda
- Triton

The easiest way to set the environment is

1. Have a Linux machine with a GPU
2. Ensure `nvidia-smi` is installed
   1. Note the CUDA version
3. Install `conda`
4. Install the relevant CUDA toolkit version (from 2.1) using `conda`. [Reference](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases)
5. Use `nvcc --version` to verify the installation
6. Run `nvcc -o <output file name> <file to compile>`

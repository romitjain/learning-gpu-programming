"""
Quick usage of swizzle2d to understand how it works
"""

import torch
import triton
import triton.language as tl


@triton.jit
def swizzle_kernel(
    input_ptr,
    output_ptr,
    group_size: tl.constexpr
):
    """
    Implementation inspired from CUDA MODE lecture. This uses as inbuilt swizzle function.
    """
    pidy = tl.program_id(axis=0)
    pidx = tl.program_id(axis=1)

    ysize = tl.num_programs(axis=0)
    xsize = tl.num_programs(axis=1)

    pidy_new, pidx_new = tl.swizzle2d(pidy, pidx, ysize, xsize, group_size)

    # Read data from row major input_ptr
    i = pidy * xsize
    j = pidx
    data = tl.load(input_ptr + i +j, (pidy < ysize) and (j < xsize))

    # Write back to col major output_ptr
    i = pidy_new * xsize
    j = pidx_new
    tl.store(output_ptr + i + j, data)

@triton.jit
def swizzle_kernel1d(
    input_ptr,
    output_ptr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    group_size: tl.constexpr
):
    """
    Implementation of swizlling using 1D launch grid instead of 2D launch grid
    """
    pid = tl.program_id(0)
    pidy = pid//N
    pidx = pid%N

    size = tl.num_programs(0)
    ysize = tl.cdiv(M, BLOCK_SIZE_M)
    xsize = tl.cdiv(N, BLOCK_SIZE_N)

    pidy_new, pidx_new = tl.swizzle2d(pidy, pidx, ysize, xsize, group_size)

    # Read data from row major input_ptr
    i = pidy * xsize
    j = pidx
    data = tl.load(input_ptr + i +j, (pidy < ysize) and (j < xsize))

    # Write back to col major output_ptr
    i = pidy_new * xsize
    j = pidx_new
    tl.store(output_ptr + i + j, data)


@triton.jit
def swizzle_alt_kernel(
    input_ptr,
    stride_am,
    stride_an,
    c_ptr,
    stride_cm,
    stride_cn,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Implementation similar to what is provided here:
    https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-0a3-matrix-multiplication-py

    Note: Not working correctly.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    i_ptrs = input_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    data = tl.load(i_ptrs, c_mask)

    tl.store(c_ptrs, data, c_mask)


def swizzle_triton(
    input: torch.Tensor
) -> torch.Tensor:
    assert input.is_cuda, f"Input matrix is not on GPU, input"
    assert input.is_contiguous(), f"Input is not contaguous"

    output = torch.ones_like(input)

    grid = (output.shape[0], output.shape[1],)
    swizzle_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        group_size=3
    )

    return output


def swizzle_triton_1d(
    input: torch.Tensor
) -> torch.Tensor:
    assert input.is_cuda, f"Input matrix is not on GPU, input"
    assert input.is_contiguous(), f"Input is not contaguous"

    output = torch.ones_like(input)

    grid = (output.shape[0]*output.shape[1],)
    swizzle_kernel1d[grid](
        input_ptr=input,
        output_ptr=output,
        M=input.shape[0],
        N=input.shape[1],
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=1,
        group_size=3
    )

    return output


def swizzle_alt_triton(
    input: torch.Tensor
) -> torch.Tensor:
    assert input.is_cuda, f"Input matrix is not on GPU, input"
    assert input.is_contiguous(), f"Input is not contiguous"

    output = torch.ones_like(input)

    grid = (output.shape[0]*output.shape[1],)
    swizzle_alt_kernel[grid](
        input_ptr=input,
        stride_am=input.stride(0),
        stride_an=input.stride(1),
        c_ptr=output,
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        M=output.shape[0],
        N=output.shape[1],
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=1,
        GROUP_SIZE_M=3
    )

    return output


if __name__ == '__main__':
    m, n = 7, 5
    a = torch.arange(m*n).view(m, n).to('cuda', torch.float32)

    print(f'Input matrix:\n{a}')

    o = swizzle_triton(a)
    print(f'Output matrix swizzle 2D:\n{o}')

    o = swizzle_triton_1d(a)
    print(f'Output matrix swizzle 1D:\n{o}')

    o = swizzle_alt_triton(a)
    print(f'Output matrix triton implementation:\n{o}')

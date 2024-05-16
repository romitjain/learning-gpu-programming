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

if __name__ == '__main__':
    m, n = 7, 5
    a = torch.arange(m*n).view(m, n).to('cuda', torch.float32)

    o = swizzle_triton(a)

    print(f'Input matrix:\n{a}')
    print(f'Output matrix:\n{o}')

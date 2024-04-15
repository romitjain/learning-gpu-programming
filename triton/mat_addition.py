import time
import torch

import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def matadd_kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        rows,
        cols,
        BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    row_offset = row_idx*cols + tl.arange(0, BLOCK_SIZE)
    col_offset = tl.arange(0, BLOCK_SIZE) < cols

    a = tl.load(A_ptr + row_offset, mask=col_offset)
    b = tl.load(B_ptr + row_offset, mask=col_offset)

    out = a + b

    tl.store(C_ptr+row_offset, out, mask=col_offset)

def matadd(A: torch.Tensor, B: torch.Tensor):
    assert A.shape == B.shape, 'A and B are not of the same shape'

    num_rows, num_cols = A.shape[0], A.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(num_cols)
    grid = (num_rows,)

    output = torch.empty(size=A.shape).to(device=device, dtype=A.dtype)

    assert A.is_cuda and B.is_cuda and output.is_cuda, f'One of the matrix is not on GPU {A.is_cuda, B.is_cuda, output.is_cuda}'

    print(f'Block size: {BLOCK_SIZE}, grid: {grid}')
    matadd_kernel[grid](
        A_ptr=A, B_ptr=B, C_ptr=output, rows=num_rows, cols=num_cols, BLOCK_SIZE=BLOCK_SIZE
    )

    return output

if __name__ == '__main__':
    A = torch.randint(0, 10, size=(1122, 2412), device=device, dtype=torch.float32)
    B = torch.randint(0, 10, size=(1122, 2412), device=device, dtype=torch.float32)

    y_torch = A+B
    y_triton = matadd(A, B)

    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    print("Arrays are same")

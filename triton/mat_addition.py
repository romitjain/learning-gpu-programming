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
        N,
        BLOCK_SIZE: tl.constexpr
):
    col_id = tl.program_id(axis=0)
    row_id = tl.program_id(axis=1)

def matadd(A: torch.Tensor, B: torch.Tensor):
    assert A.shape == B.shape, 'A and B are not of the same shape. Currently only supports same sized matrices'

    assert A.shape[0] == A.shape[1], 'A is not a square matrix. Currently only supports square matrix'

    n_elems = A.shape[0]
    BLOCK_SIZE = ((n_elems+1)%1024, (n_elems+1)%1024)
    grid = (triton.cdiv(n_elems/BLOCK_SIZE[0]),)

    output = torch.empty(size=A.shape).to(device)

    assert A.cuda() & B.cuda() & output.cuda(), 'One of the matrix is not on GPU'

    print(f'Block size: {BLOCK_SIZE}, grid: {grid}')
    matadd_kernel[grid](
        A_ptr=A, B_ptr=B, C_ptr=output, N=n_elems, BLOCK_SIZE=BLOCK_SIZE
    )

    return output

if __name__ == '__main__':
    pass

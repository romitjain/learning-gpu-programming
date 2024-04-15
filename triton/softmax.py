import time
import torch

import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def softmax_kernel(
        A_ptr,
        O_ptr,
        M,
        BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(axis=0)
    offsets = row_id*M + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < M

    a = tl.load(A_ptr + offsets, mask=mask, other=-float('inf'))
    row_wise_exp = tl.exp(a)
    row_wise_sum = tl.sum(row_wise_exp, axis=0)
    op = row_wise_exp/row_wise_sum

    tl.store(O_ptr + offsets, op, mask=mask)


def softmax(A: torch.Tensor):
    rows, cols = A.shape
    output = torch.empty(size=A.shape).to(device)

    assert A.is_cuda and output.is_cuda, 'One of the matrix is not on GPU'

    # Block size will be equal to number of columns
    # so that every row is operated in one block
    BLOCK_SIZE = triton.next_power_of_2(cols)
    print(f'Block size: {BLOCK_SIZE}, grid: {(rows,)}')

    softmax_kernel[(rows,)](
        A_ptr=A, O_ptr=output, M=cols, BLOCK_SIZE=BLOCK_SIZE
    )

    return output


if __name__ == '__main__':
    A = torch.randint(0, 10, size=(1823, 781), device=device, dtype=torch.float32)

    y_torch = torch.softmax(A, dim=-1)
    y_triton = softmax(A)

    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    print("Arrays are same")

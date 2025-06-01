import pdb
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.cpp_extension
import triton

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rms_cuda_kernel = torch.utils.cpp_extension.load(
    name='rms_forward',
    sources=["rms.cu", "../utils.c"],
    verbose=True,
    extra_cuda_cflags=["-O2 -diag-suppress 2464"]
)

x = torch.randn((1, 32, 32), device=device)
out_cuda = torch.empty_like(x)
rms_cuda_kernel.rms_forward(x, out_cuda, 1e-6, 1.0, 1)

rms = torch.nn.RMSNorm(
    normalized_shape=x.shape[-1:],
    eps=1e-6,
    device=device
)
out_torch = rms(x)

try:
    assert torch.allclose(out_cuda, out_torch), "Cuda kernel implementation is not correct!"
    print("Cuda kernel test passed!")
except Exception as err:
    diff = abs(out_cuda - out_torch)
    print(
        f"Absolute error: mean, {diff.mean()} max, {diff.max()}, min {diff.min()}"
    )
    import pdb
    pdb.set_trace()



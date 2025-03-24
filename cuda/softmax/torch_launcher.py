import torch
import torch.utils.cpp_extension
import triton

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
softmax_cuda_kernel = torch.utils.cpp_extension.load(
    name='softmax_forward',
    sources=["softmax_interface.cpp", "softmax.cu", "../utils.c"],
    verbose=True,
    extra_cuda_cflags=["-O2 -diag-suppress 2464"]
)

x = torch.randn((2, 16, 32), device=device)
out_cuda = torch.empty_like(x)
softmax_cuda_kernel.softmax_forward(x, out_cuda)
out_torch = torch.nn.functional.softmax(x, dim=-1)

assert torch.allclose(out_cuda, out_torch), "Cuda kernel implementation is not correct!"
print("Cuda kernel test passed!")

@triton.testing.perf_report(
triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2 ** i for i in range(2, 8)],
        line_arg='provider',
        line_vals=[
            'cuda',
            'torch',
        ],
        line_names=[
            "Cuda",
            "Torch",
        ],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="Performance",
        # values for function arguments not in `x_names` and `y_name`
        args={'B': 4, 'V': 32},
    ))
def benchmark(B, N, V, provider):
    x = torch.randn(B, N, V, device=device, dtype=torch.float32)
    output_tensor = torch.empty_like(x)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'cuda':
         ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax_cuda_kernel.softmax_forward(x, output_tensor), quantiles=quantiles
        )
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.softmax(x, dim=-1), quantiles=quantiles
        )

    def gbps(ms): return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(
    show_plots=True,
    print_data=True
)

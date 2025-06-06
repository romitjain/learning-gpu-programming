import pdb
import pandas as pd
from matplotlib import pyplot as plt
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

x = torch.randn((2, 64, 1024), device=device)
out_cuda = torch.empty_like(x)
softmax_cuda_kernel.softmax_forward(x, out_cuda, 2)
out_torch = torch.nn.functional.softmax(x, dim=-1)

try:
    assert torch.allclose(out_cuda, out_torch), "Cuda kernel implementation is not correct!"
    print("Cuda kernel test passed!")
except Exception as err:
    diff = abs(out_cuda - out_torch)
    print(
        f"Absolute error: mean, {diff.mean()} max, {diff.max()}, min {diff.min()}"
    )
    raise err

x_vals = [2**i for i in range(1, 10)]
x_vals += [i for i in range(1024, 2**18+1, 2048)]
x_vals = sorted(x_vals)

print(f"Testing across {x_vals} vocab sizes")

@triton.testing.perf_report(
triton.testing.Benchmark(
        x_names=['V'],
        x_vals=x_vals,
        line_arg='provider',
        line_vals=[
            'cuda (single warp)',
            'cuda (multi warp)',
            'torch',
        ],
        line_names=[
            "Cuda (Single warp)",
            "Cuda (Multi warp)",
            "Torch",
        ],
        styles=[('blue', '-'), ('red', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="Performance (Memory bandwidth consumption) vs Vocab size",
        # values for function arguments not in `x_names` and `y_name`
        args={'B': 1, 'N': 1024},
    ))
def benchmark(B, N, V, provider):
    x = torch.randn(B, N, V, device=device, dtype=torch.float32)
    output_tensor = torch.empty_like(x)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'cuda (single warp)':
         ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax_cuda_kernel.softmax_forward(x, output_tensor, 1), quantiles=quantiles, warmup=50, rep=200
        )
    elif provider == 'cuda (multi warp)':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax_cuda_kernel.softmax_forward(x, output_tensor, 2), quantiles=quantiles, warmup=50, rep=200
        )
    elif provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.softmax(x, dim=-1), quantiles=quantiles, warmup=50, rep=200
        )

    def gbps(ms): return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

df = benchmark.run(
    show_plots=True,
    print_data=True,
    return_df=True
)

plt.figure(figsize=(12, 6))
for column in ["Cuda (Single warp)", "Cuda (Multi warp)", "Torch"]:
    plt.plot(df["V"], df[column], marker='o', label=column)

plt.xlabel("Vocab Size (V)")
plt.ylabel("Memory Bandwidth Usage (GB/s)")
plt.title("Softmax Kernel Performance")
plt.legend()
plt.tight_layout()

# Save the new plot
plot_path_raw = "fixed_batch_rows_val_cols.png"
plt.savefig(plot_path_raw)

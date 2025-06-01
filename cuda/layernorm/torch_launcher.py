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

x = torch.randn((2, 1024, 1024), device=device)
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

x_vals = [2**i for i in range(1, 10)]
x_vals += [i for i in range(32, 2**18+1, 2048)]
x_vals = sorted(x_vals)

print(f"Testing across {x_vals} vocab sizes")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['V'],
        x_vals=x_vals,
        line_arg='provider',
        line_vals=[
            'cuda (single warp)',
            # 'cuda (multi warp)',
            'torch',
        ],
        line_names=[
            "Cuda (Single warp)",
            # "Cuda (Multi warp)",
            "Torch",
        ],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="Performance (Memory bandwidth consumption) vs Dim size",
        args={'B': 1, 'N': 1024},
    ))
def benchmark(B, N, V, provider):
    x = torch.randn(B, N, V, device=device, dtype=torch.float32)
    output_tensor = torch.empty_like(x)
    quantiles = [0.5, 0.2, 0.8]

    eps = 1e-6

    if provider == 'cuda (single warp)':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms_cuda_kernel.rms_forward(x, output_tensor, eps, 1.0, 1), quantiles=quantiles, warmup=20, rep=100
        )
    elif provider == 'torch':
        rms = torch.nn.RMSNorm(
            normalized_shape=x.shape[-1:],
            eps=eps,
            device=device
        )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms(x), warmup=20, rep=100, quantiles=quantiles
        )

    def gbps(ms): return 2 * x.nelement() * \
        x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

df = benchmark.run(
    show_plots=True,
    print_data=True,
    return_df=True
)

plt.figure(figsize=(12, 6))
for column in ["Cuda (Single warp)", "Torch"]:
    plt.plot(df["V"], df[column], marker='o', label=column)

plt.xlabel("Vocab Size (V)")
plt.ylabel("Memory Bandwidth Usage (GB/s)")
plt.title("RMSNorm Kernel Performance")
plt.legend()
plt.tight_layout()

# Save the new plot
plot_path_raw = "fixed_batch_rows_val_cols.png"
plt.savefig(plot_path_raw)

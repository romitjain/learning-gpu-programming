#include <vector>
#include <torch/extension.h>

void launch_rms(torch::Tensor data, torch::Tensor out, float eps, float gamma, int version,);

void rms_forward(torch::Tensor data, torch::Tensor out, float eps, float gamma, int version)
{
    launch_rms(data, out, eps, gamma, version);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_forward", &rms_forward, "Custom CUDA RMS");
}
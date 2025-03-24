#include <vector>
#include <torch/extension.h>

void launch_softmax(torch::Tensor data, torch::Tensor out);

void softmax_forward(torch::Tensor data, torch::Tensor out)
{
    launch_softmax(data, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_forward", &softmax_forward, "Custom CUDA Softmax");
}

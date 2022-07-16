#include "frontend.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_fp32", &forward_fp32,
          "Large DepthWise Conv2d Implicit GEMM forward (fp32)");
    m.def("backward_data_fp32", &backward_data_fp32,
          "Large DepthWise Conv2d Implicit GEMM backward data (fp32)");
    m.def("backward_filter_fp32", &backward_filter_fp32,
          "Large DepthWise Conv2d Implicit GEMM backward filter (fp32)");
    m.def("forward_fp16", &forward_fp16,
          "Large DepthWise Conv2d Implicit GEMM forward (fp16)");
    m.def("backward_data_fp16", &backward_data_fp16,
          "Large DepthWise Conv2d Implicit GEMM backward data (fp16)");
    m.def("backward_filter_fp16", &backward_filter_fp16,
          "Large DepthWise Conv2d Implicit GEMM backward filter (fp16)");
}

#include <pybind11/pybind11.h>

#include "interpolate/trilinear_devox.hpp"


PYBIND11_MODULE(_pvcnn_backend, m) {
  m.def("trilinear_devoxelize_forward", &trilinear_devoxelize_forward,
        "Trilinear Devoxelization forward (CUDA)");
  m.def("trilinear_devoxelize_backward", &trilinear_devoxelize_backward,
        "Trilinear Devoxelization backward (CUDA)");

}

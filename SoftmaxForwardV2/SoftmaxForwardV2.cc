#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <miopen/miopen.h>
#include <stdio.h>

#include <iostream>

#include "common/file_helper.h"
#include "common/generator.h"
#include "common/half.hpp"
#include "common/logging.h"

using half_float::half;
using namespace half_float::literal;

#define CHECK_MIOPEN(msg) \
  CHECK_EQ(reinterpret_cast<miopenStatus_t>(msg), miopenStatusSuccess)

int main() {
  hipSetDevice(0);

  miopenHandle_t handle;
  CHECK_MIOPEN(miopenCreate(&handle));

  // constance
  using dtype = half;
  auto miopen_type = miopenHalf;
  std::vector<int> dims{16, 100000, 1, 1};
  std::vector<int> strides{100000, 1, 1, 1};
  const int input_size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  const int output_size = input_size;

  // inputs
  std::vector<dtype> input = LoadTensorFromFile<half>(
      "/work/MiOpen-OpTest/SoftmaxForwardV2/input.bin", input_size);
  dtype *d_in;
  hipMalloc(&d_in, input_size * sizeof(dtype));
  miopenTensorDescriptor_t desc;
  CHECK_MIOPEN(miopenCreateTensorDescriptor(&desc));
  CHECK_MIOPEN(miopenSetTensorDescriptor(
      desc, miopen_type, dims.size(), dims.data(), strides.data()));

  // outputs
  std::vector<dtype> output(input_size, 0.0_h);
  dtype *d_out;
  hipMalloc(&d_out, output_size * sizeof(dtype));

  // memcpy h2d
  hipMemcpy(
      d_in, input.data(), input_size * sizeof(dtype), hipMemcpyHostToDevice);

  // op
  float alpha = 1.0;
  float beta = 0.0;
  CHECK_MIOPEN(miopenSoftmaxForward_V2(handle,
                                       &alpha,
                                       desc,
                                       d_in,
                                       &beta,
                                       desc,
                                       d_out,
                                       MIOPEN_SOFTMAX_LOG,
                                       MIOPEN_SOFTMAX_MODE_INSTANCE));

  hipDeviceSynchronize();

  // memcpy d2h
  hipMemcpy(
      output.data(), d_out, output_size * sizeof(dtype), hipMemcpyDeviceToHost);

  // free
  miopenDestroyTensorDescriptor(desc);
  hipFree(d_in);
  hipFree(d_out);
  CHECK_MIOPEN(miopenDestroy(handle));

  // result
  SaveTensorToFile("/work/MiOpen-OpTest/SoftmaxForwardV2/output_dcu.bin",
                   output.data(),
                   output_size);
  for (int i = 0; i < 10; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
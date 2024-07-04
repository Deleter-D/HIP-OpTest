#include <hip/hip_runtime.h>

#include <hipcub/hipcub.hpp>
#include <iostream>
#include <vector>

#include "common/generator.h"

using data_t = float;

template <typename T>
__device__ __inline__ void AtomicAddWithWarp(T *sum, T value) {
  using WarpReduce = hipcub::WarpReduce<T>;
  typename WarpReduce::TempStorage temp_storage;
  value = WarpReduce(temp_storage).Sum(value);
  if (hipcub::LaneId() == 0) {
    atomicAdd(sum, value);
  }
}

template <typename T>
__global__ void TestKernel(T *in, T *out) {
  AtomicAddWithWarp(&out[blockIdx.x], in[threadIdx.x]);
}

int main(int argc, char const *argv[]) {
  hipSetDevice(0);

  hipStream_t stream;
  hipStreamCreate(&stream);

  size_t input_size = 1024;

  dim3 block(256);
  dim3 grid((input_size + block.x - 1) / block.x);

  std::vector<data_t> input(input_size, 0.0001);
  std::vector<data_t> output(grid.x, 0);

  data_t *d_in, *d_out;
  hipMalloc((void **)&d_in, sizeof(data_t) * input.size());
  hipMalloc((void **)&d_out, sizeof(data_t) * output.size());

  hipMemcpy(
      d_in, input.data(), sizeof(data_t) * input.size(), hipMemcpyHostToDevice);

  TestKernel<data_t><<<grid, block, 0, stream>>>(d_in, d_out);

  hipStreamSynchronize(stream);

  hipMemcpy(output.data(),
            d_out,
            sizeof(data_t) * output.size(),
            hipMemcpyDeviceToHost);

  for (auto i : output) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  return 0;
}

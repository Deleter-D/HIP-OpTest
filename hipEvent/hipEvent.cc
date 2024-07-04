#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "common/logging.h"

#define CHECK_HIP(msg) CHECK_EQ(reinterpret_cast<hipError_t>(msg), hipSuccess)
#define CHECK_HIP_EXPECTED(msg, expected) \
  CHECK_EQ(reinterpret_cast<hipError_t>(msg), expected)

int main() {
  hipSetDevice(0);

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  hipEvent_t event;
  CHECK_HIP(hipEventCreate(&event));

  CHECK_HIP(hipEventRecord(event, stream));
  CHECK_HIP_EXPECTED(hipEventQuery(event),
                     hipErrorNotReady);  // should be hipErrorNotReady

  CHECK_HIP(hipEventSynchronize(event));
  CHECK_HIP(hipEventQuery(event));  // should be hipSuccess

  CHECK_HIP(hipEventDestroy(event));
  CHECK_HIP(hipStreamDestroy(stream));

  LOG(INFO) << "hipEvent test passed";

  return 0;
}
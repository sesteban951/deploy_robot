// Frozen full-float32 inference-only persistent rollout FFI.
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <string>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
constexpr int kBlocks = 32;
constexpr int kThreads = 512;
constexpr int kShared = 85792, kParameters = 660330;
constexpr int kForwardWorkspacePerStep = 1895;
static CUmodule module;
static CUfunction forward_kernel, batch_kernel;
static int forward_capacity = 0;
static std::string last_error;

static int Check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return 0;
  const char* description = nullptr;
  cuGetErrorString(result, &description);
  last_error = std::string(operation) + ": " + (description ? description : "unknown");
  return int(result);
}

extern "C" const char* kcf_rollout_error_native() { return last_error.c_str(); }

extern "C" int kcf_rollout_initialize_native(const char* ptx, int* info) {
  CUdevice device;
  CUcontext context;
  int status;
  if ((status = Check(cuInit(0), "cuInit"))) return status;
  if ((status = Check(cuDeviceGet(&device, 0), "device"))) return status;
  if ((status = Check(cuCtxGetCurrent(&context), "current context"))) return status;
  if (!context) {
    if ((status = Check(cuDevicePrimaryCtxRetain(&context, device), "primary context"))) return status;
    if ((status = Check(cuCtxSetCurrent(context), "set context"))) return status;
  }
  if ((status = Check(cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr), "load PTX"))) return status;
  if ((status = Check(cuModuleGetFunction(&forward_kernel, module,
                                         "kcf_rollout_forward_kernel"), "forward symbol"))) return status;
  if ((status = Check(cuModuleGetFunction(&batch_kernel, module,
                                         "kcf_rollout_batch_kernel"), "batch symbol"))) return status;
  if ((status = Check(cuDeviceGetAttribute(&info[0], CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
                                          device), "cooperative support"))) return status;
  if ((status = Check(cuDeviceGetAttribute(&info[1], CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                          device), "SM count"))) return status;
  if ((status = Check(cuDeviceGetAttribute(&info[2], CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                          device), "opt-in shared limit"))) return status;
  if (!info[0] || kShared > info[2]) {
    last_error = "Persistent kernel requires cooperative launch and sufficient opt-in shared memory";
    return -1;
  }
  if ((status = Check(cuFuncSetAttribute(forward_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                         kShared), "opt-in dynamic shared memory"))) return status;
  if ((status = Check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&info[3],
                           forward_kernel, kThreads, kShared), "forward occupancy"))) return status;
  if ((status = Check(cuFuncGetAttribute(&info[4], CU_FUNC_ATTRIBUTE_NUM_REGS,
                                        forward_kernel), "forward registers"))) return status;
  if ((status = Check(cuFuncGetAttribute(&info[5], CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                        forward_kernel), "forward local bytes"))) return status;
  forward_capacity = info[1] * info[3];
  if (forward_capacity < kBlocks) {
    last_error = "Persistent-kernel occupancy cannot guarantee the requested resident blocks";
    return -1;
  }
  if ((status = Check(cuFuncGetAttribute(&info[6], CU_FUNC_ATTRIBUTE_NUM_REGS,
                                        batch_kernel), "batch registers"))) return status;
  if ((status = Check(cuFuncGetAttribute(&info[7], CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                        batch_kernel), "batch local bytes"))) return status;
  if ((status = Check(cuFuncGetAttribute(&info[8], CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                        batch_kernel), "batch shared bytes"))) return status;
  if ((status = Check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&info[9],
                           batch_kernel, kThreads, 0), "batch occupancy"))) return status;
  return 0;
}

static int Horizon(ffi::Buffer<ffi::F32> matrices) {
  if (matrices.dimensions().size() != 3 || matrices.dimensions()[0] < 1 ||
      matrices.dimensions()[1] != 103 || matrices.dimensions()[2] != 256 ||
      matrices.element_count() > INT32_MAX) return 0;
  return int(matrices.dimensions()[0]);
}

static ffi::Error Forward(CUstream stream, double timestep, double projection_balance,
    ffi::Buffer<ffi::F32> initial, ffi::Buffer<ffi::F32> matrices,
    ffi::Buffer<ffi::F32> parameters, ffi::ResultBuffer<ffi::F32> states,
    ffi::ResultBuffer<ffi::F32> workspace
    ) {
  int horizon = Horizon(matrices);
  if (!forward_kernel || forward_capacity < kBlocks || !horizon || initial.element_count() != 103 ||
      parameters.element_count() != kParameters || states->element_count() != horizon * 103 ||
      workspace->element_count() != horizon * kForwardWorkspacePerStep ||
      !(std::isfinite(timestep) && timestep > 0 && std::isfinite(projection_balance) && projection_balance > 0))
    return ffi::Error::InvalidArgument("Persistent forward ABI or occupancy mismatch");
  const float* x = initial.typed_data();
  const float* a = matrices.typed_data();
  const float* p = parameters.typed_data();
  float* y = states->typed_data();
  float* cache = workspace->typed_data();
  float dt = float(timestep), balance = float(projection_balance);
  void* args[] = {&x, &a, &p, &y, &cache, &horizon, &dt, &balance
                 };
  if (Check(cuLaunchCooperativeKernel(forward_kernel, kBlocks, 1, 1, kThreads, 1, 1,
                                     kShared, stream, args), "persistent forward launch"))
    return ffi::Error::Internal(last_error);
  return ffi::Error::Success();
}

static ffi::Error Batch(CUstream stream, double timestep, double projection_balance,
    ffi::Buffer<ffi::F32> initials, ffi::Buffer<ffi::F32> matrices,
    ffi::Buffer<ffi::F32> parameters, ffi::ResultBuffer<ffi::F32> states) {
  const auto dims = matrices.dimensions();
  if (dims.size() != 4 || dims[0] < 1 || dims[0] > INT32_MAX || dims[1] < 1
      || matrices.element_count() > INT32_MAX
      || dims[2] != 103 || dims[3] != 256 || initials.element_count() != dims[0] * 103
      || parameters.element_count() != kParameters || states->element_count() != dims[0] * dims[1] * 103
      || !batch_kernel || !(std::isfinite(timestep) && timestep > 0
          && std::isfinite(projection_balance) && projection_balance > 0))
    return ffi::Error::InvalidArgument("Population rollout ABI mismatch");
  const float *x = initials.typed_data(), *a = matrices.typed_data(), *p = parameters.typed_data();
  float* y = states->typed_data();
  int horizon = int(dims[1]);
  float dt = float(timestep), balance = float(projection_balance);
  void* args[] = {&x, &a, &p, &y, &horizon, &dt, &balance};
  if (Check(cuLaunchKernel(batch_kernel, unsigned(dims[0]), 1, 1, kThreads, 1, 1, 0,
                          stream, args, nullptr), "population forward launch"))
    return ffi::Error::Internal(last_error);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kcf_rollout_batch_native, Batch,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>()
        .Attr<double>("dt").Attr<double>("balance")
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(), {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(kcf_rollout_forward_native, Forward,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<CUstream>>()
        .Attr<double>("dt").Attr<double>("balance")
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        ,
    {ffi::Traits::kCmdBufferCompatible});

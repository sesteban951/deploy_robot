// Frozen FP32 G1 inference. Launch cooperatively after actual occupancy checks.
#include <cooperative_groups.h>
#include "projection_rounding.cuh"
#include "projection.cuh"

namespace cg = cooperative_groups;
__device__ __forceinline__ bool IsProjectionGeometry(int index) {
  return index < 6 || (index >= 7 && index <= 10) || index == 34;
}

__device__ __forceinline__ float LoadCachedWeight(const float* pointer) {
  return *pointer;
}

constexpr int kBlocks = 32, kThreads = 512;
constexpr int kWarps = kThreads / 32;
static_assert(kBlocks == 32);
static_assert(kThreads >= 85 && kThreads % 32 == 0 && kWarps <= 32);
constexpr int kHiddenRows = 512 / kBlocks;
constexpr int kFinalRows = (171 + kBlocks - 1) / kBlocks;
constexpr int kHiddenLanes = kThreads / kHiddenRows;
constexpr int kLiftColumns = 256 / kBlocks;
constexpr int kLiftLanes = kThreads / kLiftColumns < 32 ? kThreads / kLiftColumns : 32;
constexpr int kPhysicalRows = (85 + kBlocks - 1) / kBlocks;
static_assert(kHiddenLanes <= 32 && (kHiddenLanes & (kHiddenLanes - 1)) == 0);
static_assert(kWarps >= kPhysicalRows);
__device__ __constant__ int kWeight[4] = {0, 43520, 305664, 567808};
__device__ __constant__ int kBias[4] = {655360, 655872, 656384, 656896};
__device__ __constant__ int kGamma[3] = {657067, 658091, 659115};
__device__ __constant__ int kBeta[3] = {657579, 658603, 659627};
constexpr int kMean = 660139, kScale = 660224, kEpsilon = 660309;
__device__ __constant__ int kCached[4] = {
    0, kHiddenRows * 88, kHiddenRows * 88 + kHiddenRows * 520,
    kHiddenRows * 88 + 2 * kHiddenRows * 520};
constexpr int kCachedFloats = kHiddenRows * 88 + 2 * kHiddenRows * 520
                            + kFinalRows * 520;
constexpr int kSharedFloats = kCachedFloats + 512 + 2 * kWarps;
static_assert(kSharedFloats * 4 == 86848);
static_assert(sizeof(kcf_projection::ProjectionWorkspace<float>) == 53 * sizeof(float));
// Forward projection temporaries have an explicitly checked scalar layout.
static_assert(sizeof(kcf_projection::ProjectionWorkspace<kcf_projection::ReferenceFloat>) ==
              sizeof(kcf_projection::ProjectionWorkspace<float>));
static_assert(alignof(kcf_projection::ReferenceFloat) == alignof(float));

// The same two descending reduction trees are replicated independently in each
// warp. Callers protect each bank's read lifetime; no tail barrier is implied.
__device__ __forceinline__ float SumBlockReplicated(float value, float* temporary) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int step = 16; step; step >>= 1)
    value += __shfl_down_sync(0xffffffff, value, step);
  if (!lane) temporary[warp] = value;
  __syncthreads();
  value = lane < kWarps ? temporary[lane] : 0.f;
  for (int step = 16; step; step >>= 1)
    value += __shfl_down_sync(0xffffffff, value, step);
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ void CacheWeights(const float* parameters, float* cached) {
  for (int layer = 0; layer < 4; ++layer) {
    int rows = layer == 3 ? kFinalRows : kHiddenRows;
    int inputs = layer == 0 ? 85 : 512;
    int outputs = layer == 3 ? 171 : 512;
    int stride = layer == 0 ? 88 : 520;
    for (int linear = threadIdx.x; linear < rows * stride; linear += kThreads) {
      int local_row = linear % rows, input = linear / rows;
      int output = blockIdx.x * rows + local_row;
      cached[kCached[layer] + local_row * stride + input] =
          input < inputs && output < outputs
              ? LoadCachedWeight(parameters + kWeight[layer] + input * outputs + output) : 0.f;
    }
  }
  __syncthreads();
}

__device__ void GemvForward(const float* cached, const float* input,
                           const float* parameters, int layer, float* output) {
  int local_row = threadIdx.x / kHiddenLanes, lane = threadIdx.x % kHiddenLanes;
  int rows = layer == 3 ? kFinalRows : kHiddenRows;
  int inputs = layer == 0 ? 85 : 512;
  int outputs = layer == 3 ? 171 : 512;
  int stride = layer == 0 ? 88 : 520;
  int row = blockIdx.x * rows + local_row;
  float value = 0.f;
  if (local_row < rows && row < outputs) {
    for (int input_index = lane; input_index < inputs; input_index += kHiddenLanes)
      value += cached[kCached[layer] + local_row * stride + input_index] * input[input_index];
  }
  for (int step = kHiddenLanes / 2; step; step >>= 1)
    value += __shfl_down_sync(0xffffffff, value, step, kHiddenLanes);
  if (!lane && local_row < rows && row < outputs)
    output[row] = value + parameters[kBias[layer] + row];
}

__device__ void NormEluForward(const float* preactivation, const float* parameters,
                              int layer, float* activation, float* temporary
                              ) {
  float total = 0.f;
  for (int i = threadIdx.x; i < 512; i += kThreads) total += preactivation[i];
  float mean = SumBlockReplicated(total, temporary) / 512.f;
  float squared = 0.f;
  for (int i = threadIdx.x; i < 512; i += kThreads) {
    float centered = preactivation[i] - mean;
    squared += centered * centered;
  }
  float inverse = rsqrtf(SumBlockReplicated(squared, temporary + kWarps) / 512.f
                         + parameters[kEpsilon + layer]);
  for (int i = threadIdx.x; i < 512; i += kThreads) {
    float value = (preactivation[i] - mean) * inverse * parameters[kGamma[layer] + i]
                  + parameters[kBeta[layer] + i];
    activation[i] = value > 0.f ? value : expm1f(value);
  }
  __syncthreads();
}

extern "C" __global__ void kcf_rollout_forward_kernel(
    const float* initial, const float* matrices, const float* parameters,
    float* states, float* workspace, int horizon, float dt, float balance
    ) {
  extern __shared__ float shared[];
  float* activation = shared + kCachedFloats;
  float* temporary = activation + 512;
  CacheWeights(parameters, shared);
  cg::grid_group grid = cg::this_grid();
  int global_thread = blockIdx.x * kThreads + threadIdx.x;
  for (int step = 0; step < horizon; ++step) {
    const float* previous = step ? states + (step - 1) * 85 : initial;
    float* preactivation = workspace + step * 1536;
    float* lifted = workspace + horizon * 1536 + step * 256;
    float* raw = workspace + horizon * 1792 + step * 85;
    for (int i = threadIdx.x; i < 85; i += kThreads)
      activation[i] = (previous[i] - parameters[kMean + i]) / parameters[kScale + i];
    if (global_thread < 85) lifted[global_thread] =
        (previous[global_thread] - parameters[kMean + global_thread])
        / parameters[kScale + global_thread];
    __syncthreads();
    GemvForward(shared, activation, parameters, 0, preactivation);
    grid.sync();
    for (int layer = 0; layer < 3; ++layer) {
      NormEluForward(preactivation + layer * 512, parameters, layer, activation, temporary);
      GemvForward(shared, activation, parameters, layer + 1,
                   layer == 2 ? lifted + 85 : preactivation + (layer + 1) * 512);
      grid.sync();
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int row = blockIdx.x * kPhysicalRows + warp;
    float value = 0.f;
    if (warp < kPhysicalRows && row < 85) {
      for (int i = lane; i < 256; i += 32)
        value += matrices[(step * 85 + row) * 256 + i] * lifted[i];
    }
    for (int offset = 16; offset; offset >>= 1)
      value += __shfl_down_sync(0xffffffff, value, offset);
    if (!lane && warp < kPhysicalRows && row < 85)
      raw[row] = value * parameters[kScale + row] + parameters[kMean + row];
    grid.sync();
    // These74 writes are disjoint from the eleven scalar geometry outputs.
    // The existing following grid.sync makes both sets visible to next step.
    if (global_thread < 85 && !IsProjectionGeometry(global_thread))
      states[step * 85 + global_thread] = raw[global_thread];
    if (!global_thread) {
      // Preserve the captured FP32 projection instruction qualifiers.
      using F = kcf_projection::ReferenceFloat;
      kcf_projection::ProjectionWorkspace<F> local;
      kcf_projection::forward<F, false, float>(previous, raw, parameters + kScale,
          F(dt), F(balance), states + step * 85, &local);
    }
    grid.sync();
  }
}

#include "batch.cu"

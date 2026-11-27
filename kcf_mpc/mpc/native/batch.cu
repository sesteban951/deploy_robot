// Forward-only population inference. One independent CTA per trajectory.
// The FP32 dot-product trees, LN/ELU and projector match native32. Weights
// are transposed once at startup for contiguous input loads within each warp.
__device__ void BatchGemv(const float* input, const float* parameters,
                          int layer, float* output) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int inputs = layer == 0 ? 104 : 512;
  const int outputs = layer == 3 ? 152 : 512;
  for (int row = warp; row < outputs; row += 16) {
    float value = 0.f;
    for (int i = lane; i < inputs; i += 32)
      value += parameters[kWeight[layer] + row * inputs + i] * input[i];
    for (int offset = 16; offset; offset >>= 1)
      value += __shfl_down_sync(0xffffffff, value, offset);
    if (!lane) output[row] = value + parameters[kBias[layer] + row];
  }
  __syncthreads();
}

extern "C" __global__ void kcf_rollout_batch_kernel(
    const float* initials, const float* all_matrices, const float* parameters,
    float* all_states, int horizon, float dt, float balance) {
  __shared__ float activation[512], preactivation[512], lifted[256], raw[104];
  __shared__ float temporary[32];
  const float* initial = initials + blockIdx.x * 104;
  const float* matrices = all_matrices + blockIdx.x * horizon * 104 * 256;
  float* states = all_states + blockIdx.x * horizon * 104;
  for (int step = 0; step < horizon; ++step) {
    const float* previous = step ? states + (step - 1) * 104 : initial;
    if (threadIdx.x < 104) {
      float value = (previous[threadIdx.x] - parameters[kMean + threadIdx.x])
                    / parameters[kScale + threadIdx.x];
      activation[threadIdx.x] = value;
      lifted[threadIdx.x] = value;
    }
    __syncthreads();
    BatchGemv(activation, parameters, 0, preactivation);
    for (int layer = 0; layer < 3; ++layer) {
      NormEluForward(preactivation, parameters, layer, activation, temporary);
      BatchGemv(activation, parameters, layer + 1,
                layer == 2 ? lifted + 104 : preactivation);
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int row = warp; row < 104; row += 16) {
      float value = 0.f;
      for (int i = lane; i < 256; i += 32)
        value += matrices[(step * 104 + row) * 256 + i] * lifted[i];
      for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
      if (!lane) raw[row] = value * parameters[kScale + row] + parameters[kMean + row];
    }
    __syncthreads();
    if (threadIdx.x < 104 && !IsProjectionGeometry(threadIdx.x))
      states[step * 104 + threadIdx.x] = raw[threadIdx.x];
    if (!threadIdx.x) {
      using F = kcf_projection::ReferenceFloat;
      kcf_projection::ProjectionWorkspace<F> projection;
      kcf_projection::forward<F, false, float>(previous, raw, parameters + kScale,
          F(dt), F(balance), states + step * 104, &projection);
    }
    __syncthreads();
  }
}

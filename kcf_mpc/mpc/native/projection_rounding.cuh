// Frozen actual-Pallas-instruction scalar type; used ONLY for projection.
// No new reciprocal approximation: division/sqrt qualifiers match captured PTX.
#pragma once
namespace kcf_projection {
struct ReferenceFloat {
  float value;
  __device__ __forceinline__ ReferenceFloat() = default;
  __device__ __forceinline__ ReferenceFloat(float x):value(x){}
  __device__ __forceinline__ operator float() const {return value;}
  __device__ __forceinline__ ReferenceFloat& operator+=(ReferenceFloat x) {
    value=__fadd_rn(value,x.value);return *this;
  }
  __device__ __forceinline__ ReferenceFloat& operator-=(ReferenceFloat x) {
    value=__fsub_rn(value,x.value);return *this;
  }
};
static_assert(sizeof(ReferenceFloat)==sizeof(float));
__device__ __forceinline__ ReferenceFloat operator+(ReferenceFloat a,ReferenceFloat b) {
  return __fadd_rn(a.value,b.value);
}
__device__ __forceinline__ ReferenceFloat operator-(ReferenceFloat a,ReferenceFloat b) {
  return __fsub_rn(a.value,b.value);
}
__device__ __forceinline__ ReferenceFloat operator-(ReferenceFloat a) {return -a.value;}
__device__ __forceinline__ ReferenceFloat operator*(ReferenceFloat a,ReferenceFloat b) {
  return __fmul_rn(a.value,b.value);
}
__device__ __forceinline__ ReferenceFloat operator/(ReferenceFloat a,ReferenceFloat b) {
  float result;
  asm("div.full.f32 %0, %1, %2;":"=f"(result):"f"(a.value),"f"(b.value));
  return result;
}
__device__ __forceinline__ ReferenceFloat square_root(ReferenceFloat a) {
  float result;asm("sqrt.approx.f32 %0, %1;":"=f"(result):"f"(a.value));return result;
}
}  // namespace kcf_projection

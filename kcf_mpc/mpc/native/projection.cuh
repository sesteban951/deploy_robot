// Frozen exact scalar forward projector for native GPU inference.
// No fast math, inverse approximation or smoothing.
#pragma once
#if !defined(__CUDACC_RTC__)
#include <cmath>
#endif

#if defined(__CUDACC__)
#define KCF_HD __host__ __device__ __forceinline__
#else
#define KCF_HD inline
#endif

namespace kcf_projection {

template<class T> struct ProjectionWorkspace {
  T normalized_raw[11], raw_rotation[9], previous_rotation[9], a[9];
  T wp[3], wv[3], factors[6], velocity[3];
};
static_assert(sizeof(ProjectionWorkspace<float>) == 53*sizeof(float));

KCF_HD int geometry_index(int i) {
  return i < 6 ? i : (i < 10 ? i+1 : 34);
}
KCF_HD int scale_index(int i) {
  return i < 2 ? i : (i == 2 ? 7 : (i == 3 ? 4 : (i == 4 ? 5 : 34)));
}
template<class T> KCF_HD T maximum(T a,T b) {
  if (a != a) return a;
  if (b != b) return b;
  return a > b ? a : b;
}
KCF_HD float square_root(float a) {return ::sqrtf(a);}
KCF_HD double square_root(double a) {return ::sqrt(a);}
template<int N,class T> KCF_HD void unit(const T* input,const T* fallback,T* output) {
  T squared=T(0);
  for(int i=0;i<N;++i) squared += input[i]*input[i];
  T denominator=square_root(maximum(squared,T(1e-16)));
  for(int i=0;i<N;++i) output[i]=squared>T(1e-16)?input[i]/denominator:fallback[i];
}
template<class T> KCF_HD void rotation(const T* x,T* r) {
  const T heading_fallback[2]={T(0),T(1)},gravity_fallback[3]={T(0),T(0),T(-1)};
  T heading[2],gravity[3],roll_input[2],roll[2];
  unit<2>(x+2,heading_fallback,heading);
  unit<3>(x+7,gravity_fallback,gravity);
  const T lower=T(-1.+1e-7),upper=T(1.-1e-7);
  T sp=maximum(lower,-maximum(-upper,-gravity[0]));
  T cp=square_root(T(1)-sp*sp),sy=heading[0],cy=heading[1];
  roll_input[0]=-gravity[1];roll_input[1]=-gravity[2];
  unit<2>(roll_input,heading_fallback,roll);
  T sr=roll[0],cr=roll[1];
  r[0]=cy*cp;r[1]=cy*sp*sr-sy*cr;r[2]=cy*sp*cr+sy*sr;
  r[3]=sy*cp;r[4]=sy*sp*sr+cy*cr;r[5]=sy*sp*cr-cy*sr;
  r[6]=-sp;r[7]=cp*sr;r[8]=cp*cr;
}
template<class T> KCF_HD void factor(const T* m,T* f) {
  f[0]=m[0];f[3]=m[3]/f[0];f[4]=m[6]/f[0];
  f[1]=m[4]-f[3]*m[3];T residual=m[7]-f[4]*m[3];
  f[5]=residual/f[1];f[2]=m[8]-f[4]*m[6]-f[5]*residual;
}
template<class T> KCF_HD void solve(const T* f,const T* rhs,T* out) {
  T y0=rhs[0],y1=rhs[1]-f[3]*y0,y2=rhs[2]-f[4]*y0-f[5]*y1;
  out[2]=y2/f[2];out[1]=y1/f[1]-f[5]*out[2];
  out[0]=y0/f[0]-f[3]*out[1]-f[4]*out[2];
}
// CopyState=false is reserved for the cooperative caller, which
// initializes all non-geometry channels in disjoint threads before its
// existing post-projection grid barrier. The default keeps the generic ABI.
template<class T,bool CopyState=true,class Storage=T> KCF_HD void forward(
    const Storage* previous,const Storage* raw,const Storage* scale,
    T dt,T balance,Storage* following,ProjectionWorkspace<T>* saved) {
  T p[11],b[3],m[9],rhs[3],position[3];
  auto& s=*saved;
  for(int i=0;i<11;++i) {p[i]=previous[geometry_index(i)];s.normalized_raw[i]=raw[geometry_index(i)];}
  const T hf[2]={T(0),T(1)},gf[3]={T(0),T(0),T(-1)};
  T normalized_heading[2],normalized_gravity[3];
  unit<2>(s.normalized_raw+2,hf,normalized_heading);unit<3>(s.normalized_raw+7,gf,normalized_gravity);
  for(int i=0;i<2;++i) s.normalized_raw[2+i]=normalized_heading[i];
  for(int i=0;i<3;++i) s.normalized_raw[7+i]=normalized_gravity[i];
  rotation(p,s.previous_rotation);
  rotation(s.normalized_raw,s.raw_rotation);
  for(int i=0;i<9;++i) s.a[i]=T(.5)*dt*s.raw_rotation[i];
  for(int i=0;i<3;++i) {
    T world=T(0);for(int j=0;j<3;++j) world+=s.previous_rotation[3*i+j]*p[j==2?10:4+j];
    b[i]=p[i==2?6:i]+T(.5)*dt*world;
    T sp=maximum(T(scale[scale_index(i)]),T(1e-4));
    T sv=maximum(T(scale[scale_index(i+3)]),T(1e-4));
    s.wp[i]=balance/(sp*sp);s.wv[i]=T(1)/(sv*sv);
  }
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    T value=T(0);for(int k=0;k<3;++k) value+=s.a[3*k+i]*s.wp[k]*s.a[3*k+j];
    m[3*i+j]=value+(i==j?s.wv[i]:T(0));
  }
  for(int i=0;i<3;++i) {
    T value=T(0);for(int k=0;k<3;++k) value+=s.a[3*k+i]*s.wp[k]*(s.normalized_raw[k==2?6:k]-b[k]);
    rhs[i]=s.wv[i]*s.normalized_raw[i==2?10:4+i]+value;
  }
  factor(m,s.factors);solve(s.factors,rhs,s.velocity);
  for(int i=0;i<3;++i) {
    T value=T(0);for(int j=0;j<3;++j) value+=s.a[3*i+j]*s.velocity[j];
    position[i]=b[i]+value;
  }
  if constexpr(CopyState) for(int i=0;i<85;++i) following[i]=raw[i];
  for(int i=0;i<11;++i) following[geometry_index(i)]=s.normalized_raw[i];
  for(int i=0;i<3;++i) {following[i==2?7:i]=position[i];following[i==2?34:4+i]=s.velocity[i];}
}

} // namespace kcf_projection
#undef KCF_HD

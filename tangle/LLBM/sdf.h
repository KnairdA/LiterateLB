#pragma once
#include <vector_types.h>
#include <cuda-samples/Common/helper_math.h>

template <typename SDF, typename V>
__device__ __host__
float approximateDistance(SDF sdf, V origin, V dir, float d0, float d1, float eps=1e-2, unsigned N=128) {
  float distance = d0;
  float delta = (d1-d0) / N;
  for (unsigned i=0; i < N; ++i) {
    float d = sdf(origin + distance*dir);
    if (d < eps) {
      return distance;
    }
    distance += d;
    if (distance > d1) {
      return d1;
    }
  }
  return d1;
}

namespace sdf {

template <typename V>
__device__ __host__ float sphere(V p, float r) {
  return length(p) - r;
}

__device__ __host__ float box(float3 p, float3 b) {
  float3 q = fabs(p) - b;
  return length(fmaxf(q,make_float3(0))) + fmin(fmax(q.x,fmax(q.y,q.z)),0);
}

__device__ __host__ float cylinder(float3 p, float r, float h) {
	return fmax(length(make_float2(p.x,p.y)) - r, fabs(p.z) - 0.5*h);
}

__device__ __host__ float add(float a, float b) {
  return fmin(a, b);
}

__device__ __host__ float intersect(float a, float b) {
  return fmax(a, b);
}

__device__ __host__ float sub(float a, float b) {
  return intersect(-a, b);
}

__device__ __host__ float sadd(float a, float b, float k) {
  float h = clamp(0.5f + 0.5f*(b-a)/k, 0.0f, 1.0f);
  return lerp(b, a, h) - k*h*(1.f-h);
}

__device__ __host__ float ssub(float a, float b, float k) {
  float h = clamp(0.5f - 0.5f*(b+a)/k, 0.f, 1.f);
  return lerp(b, -a, h) + k*h*(1.f-h);
}

__device__ __host__ float sintersect(float a, float b, float k) {
  float h = clamp(0.5f - 0.5f*(b-a)/k, 0.f, 1.f);
  return lerp(b, a, h) + k*h*(1.f-h);
}

__device__ __host__ float3 twisted(float3 p, float k) {
  float c = cos(k*p.y);
  float s = sin(k*p.y);
  float3  q = make_float3(0,0,p.y);
  q.x = p.x*c + p.z*-s;
  q.y = p.x*s + p.z* c;
  return q;
}

}

template <typename SDF>
__device__ float3 sdf_normal(SDF sdf, float3 v, float eps=1e-4) {
  return normalize(make_float3(
    sdf(make_float3(v.x + eps, v.y, v.z)) - sdf(make_float3(v.x - eps, v.y, v.z)),
    sdf(make_float3(v.x, v.y + eps, v.z)) - sdf(make_float3(v.x, v.y - eps, v.z)),
    sdf(make_float3(v.x, v.y, v.z + eps)) - sdf(make_float3(v.x, v.y, v.z - eps))
  ));
}

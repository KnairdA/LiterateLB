#pragma once

#include "sampler.h"

#include <LLBM/kernel/collect_moments.h>
#include <LLBM/kernel/collect_curl.h>

#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

template <typename DESCRIPTOR, typename T, typename S, typename SDF>
class CurlNormS : public Sampler {
private:
Lattice<DESCRIPTOR,T,S>& _lattice;
DeviceBuffer<bool>& _mask;
SDF _geometry;

DeviceBuffer<float> _moments_rho;
DeviceBuffer<float> _moments_u;
DeviceBuffer<float> _curl_norm;

float _scale = 1;
float _lower = 0;
float _upper = 1;

public:
CurlNormS(Lattice<DESCRIPTOR,T,S>& lattice, DeviceBuffer<bool>& mask, SDF geometry):
  Sampler("Curl norm", lattice.cuboid()),
  _lattice(lattice),
  _mask(mask),
  _geometry(geometry),
  _moments_rho(lattice.cuboid().volume),
  _moments_u(DESCRIPTOR::d * lattice.cuboid().volume),
  _curl_norm(lattice.cuboid().volume)
{ }

void sample() {
  _lattice.template inspect<CollectMomentsF>(_mask, _moments_rho.device(), _moments_u.device());
  _lattice.template inspect<CollectCurlF>(_mask, _moments_u.device(), _sample_surface, _curl_norm.device());
}

void render(VolumetricRenderConfig& config) {
  raymarch<<<
    dim3(config.canvas_size.x / 32 + 1, config.canvas_size.y / 32 + 1),
    dim3(32, 32)
  >>>(config,
      _geometry,
      [samples=_sample_texture, scale=_scale, lower=_lower, upper=_upper]
      __device__ (float3 p) -> float {
        float sample = scale * tex3D<float>(samples, p.x, p.y, p.z);
        return sample >= lower && sample <= upper ? sample : 0;
      },
      [] __device__ (float x) -> float {
        return x;
      });
}

void scale() {
  auto max = thrust::max_element(thrust::device_pointer_cast(_curl_norm.device()),
                                 thrust::device_pointer_cast(_curl_norm.device() + _lattice.cuboid().volume));
  _scale = 1 / max[0];
}

void interact() {
  ImGui::SliderFloat("Scale", &_scale, 0.01f, 100.f);
  ImGui::SameLine();
  if (ImGui::Button("Auto")) {
    scale();
  }
  ImGui::DragFloatRange2("Bounds", &_lower, &_upper, 0.01f, 0.f, 1.f, "%.2f", "%.2f");
}

};

template <typename DESCRIPTOR, typename T, typename S, typename SDF>
CurlNormS(Lattice<DESCRIPTOR,T,S>&, DeviceBuffer<bool>&, SDF) -> CurlNormS<DESCRIPTOR,T,S,SDF>;

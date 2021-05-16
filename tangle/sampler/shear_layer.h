#pragma once

#include "sampler.h"

#include <LLBM/kernel/collect_moments.h>
#include <LLBM/kernel/collect_shear_layer_normal.h>

template <typename DESCRIPTOR, typename T, typename S, typename SDF>
class ShearLayerVisibilityS : public Sampler {
private:
Lattice<DESCRIPTOR,T,S>& _lattice;
DeviceBuffer<bool>& _mask;
SDF _geometry;

DeviceBuffer<float> _moments_rho;
DeviceBuffer<float> _moments_u;
DeviceBuffer<float> _shear_normals;

float3 _shear_layer;
float _lower = 0;
float _upper = 1;
bool _center = true;

public:
ShearLayerVisibilityS(Lattice<DESCRIPTOR,T,S>& lattice, DeviceBuffer<bool>& mask, SDF geometry, float3 shear_layer):
  Sampler("Shear layer visibility", lattice.cuboid()),
  _lattice(lattice),
  _mask(mask),
  _geometry(geometry),
  _moments_rho(lattice.cuboid().volume),
  _moments_u(DESCRIPTOR::d * lattice.cuboid().volume),
  _shear_normals(DESCRIPTOR::d * lattice.cuboid().volume),
  _shear_layer(shear_layer)
{ }

void sample() {
  _lattice.template inspect<CollectShearLayerNormalsF>(_mask, _moments_rho.device(), _moments_u.device(), _shear_normals.device());
  _lattice.template helper<CollectShearLayerVisibilityF>(_mask, _shear_normals.device(), _shear_layer, _sample_surface);
}

void render(VolumetricRenderConfig& config) {
  raymarch<<<
    dim3(config.canvas_size.x / 32 + 1, config.canvas_size.y / 32 + 1),
    dim3(32, 32)
  >>>(config,
      _geometry,
      [samples=_sample_texture, lower=_lower, upper=_upper, center=_center]
      __device__ (float3 p) -> float {
        float sample = tex3D<float>(samples, p.x, p.y, p.z);
        float centered = center ? 0.5 + 0.5*sample : sample;
        return fabs(sample) >= lower && fabs(sample) <= upper ? fabs(centered) : 0;
      },
      [] __device__ (float x) -> float {
        return x;
      });
}

void interact() {
  ImGui::InputFloat3("Normal", reinterpret_cast<float*>(&_shear_layer));
  ImGui::Checkbox("Center", &_center);
  ImGui::DragFloatRange2("Bounds", &_lower, &_upper, 0.01f, 0.f, 1.f, "%.2f", "%.2f");
}

};

template <typename DESCRIPTOR, typename T, typename S, typename SDF>
ShearLayerVisibilityS(Lattice<DESCRIPTOR,T,S>&, DeviceBuffer<bool>&, SDF) -> ShearLayerVisibilityS<DESCRIPTOR,T,S,SDF>;

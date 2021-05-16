#include <LLBM/base.h>
#include <LLBM/bulk.h>
#include <LLBM/boundary.h>

#include "util/render_window.h"
#include "util/texture.h"
#include "util/colormap.h"

#include "util/volumetric_example.h"
#include "sampler/velocity_norm.h"
#include "sampler/curl_norm.h"
#include "sampler/shear_layer.h"

using T = float;
using DESCRIPTOR = descriptor::D3Q19;

int main() {
cudaSetDevice(0);

const descriptor::Cuboid<DESCRIPTOR> cuboid(320, 96, 96);
Lattice<DESCRIPTOR,T> lattice(cuboid);

CellMaterials<DESCRIPTOR> materials(cuboid, [&cuboid](uint3 p) -> int {
  if (p.x == 0 || p.x == cuboid.nX-1) {
    return 2;
  } else {
    return 1;
  }
});

auto inner_cylinder = [cuboid] __host__ __device__ (float3 p) -> float {
                        float3 q = p - make_float3(0, cuboid.nY/2, cuboid.nZ/2);
                        return sdf::sphere(make_float2(q.y,q.z), cuboid.nY/T{4.5});
                      };
auto geometry = [cuboid,inner_cylinder] __host__ __device__ (float3 p) -> float {
                  float3 q = p - make_float3(0, cuboid.nY/2, cuboid.nZ/2);
                  return sdf::add(-sdf::sphere(make_float2(q.y,q.z), cuboid.nY/T{2.14}), inner_cylinder(p));
                };
materials.sdf(geometry, 0);
SignedDistanceBoundary bouzidi(lattice, materials, geometry, 1, 0);

const float wall = 0.2;

bouzidi.setVelocity([cuboid,wall](float3 p) -> float3 {
  float3 q = p - make_float3(0, cuboid.nY/2, cuboid.nZ/2);
  if (length(make_float2(q.y,q.z)) < cuboid.nY/T{2.5}) {
    return wall * normalize(make_float3(0, -q.z, q.y));
  } else {
    return make_float3(0);
  }
});

auto bulk_mask = materials.mask_of_material(1);
auto bulk_list = materials.list_of_material(1);
auto wall_mask = materials.mask_of_material(2);
auto wall_list = materials.list_of_material(2);

lattice.apply<InitializeO>(bulk_list);
lattice.apply<InitializeO>(wall_list);

cudaDeviceSynchronize();

VolumetricExample renderer(cuboid);
renderer.add<VelocityNormS>(lattice, bulk_mask, inner_cylinder);
renderer.add<ShearLayerVisibilityS>(lattice, bulk_mask, inner_cylinder, make_float3(1,0,0));
renderer.run([&](std::size_t iStep) {
  const float tau = 0.55;
  
  lattice.apply<BgkCollideO>(bulk_list, tau);
  lattice.apply<BounceBackO>(wall_list);
  lattice.apply<BouzidiO>(bouzidi.getCount(), bouzidi.getConfig());
  
  lattice.stream();
});
}

#include <LLBM/base.h>
#include <LLBM/bulk.h>
#include <LLBM/boundary.h>

#include "util/render_window.h"
#include "util/texture.h"
#include "util/colormap.h"

#include "util/volumetric_example.h"
#include "sampler/velocity_norm.h"
#include "sampler/curl_norm.h"
#include "sampler/q_criterion.h"

using T = float;
using DESCRIPTOR = descriptor::D3Q19;

int main() {
cudaSetDevice(0);

const descriptor::Cuboid<DESCRIPTOR> cuboid(500, 80, 80);
Lattice<DESCRIPTOR,T> lattice(cuboid);

CellMaterials<DESCRIPTOR> materials(cuboid, [&cuboid](uint3 p) -> int {
  if (p.y == 0 || p.y == cuboid.nY-1 || p.z == 0 || p.z == cuboid.nZ-1) {
    return 2; // boundary cell
  } else if (p.x == 0) {
    return 3; // inflow cell
  } else if (p.x == cuboid.nX-1) {
    return 4; // outflow cell
  } else {
    return 1; // bulk
  }
});

auto obstacle = [cuboid] __host__ __device__ (float3 p) -> float {
                  float3 q = p - make_float3(cuboid.nX/24.2f, cuboid.nY/2, cuboid.nZ/2);
                  return sdf::ssub(sdf::sphere(make_float2(q.y,q.z), cuboid.nY/T{9}),
                                   sdf::box(q, make_float3(cuboid.nX/128,cuboid.nY/2,cuboid.nZ/2)),
                                   5);
                };
materials.sdf(obstacle, 0);
SignedDistanceBoundary bouzidi(lattice, materials, obstacle, 1, 0);

auto bulk_mask     = materials.mask_of_material(1);
auto boundary_mask = materials.mask_of_material(2);
auto inflow_mask   = materials.mask_of_material(3);
auto outflow_mask  = materials.mask_of_material(4);

cudaDeviceSynchronize();

VolumetricExample renderer(cuboid);
renderer.add<CurlNormS>(lattice, bulk_mask, obstacle);
renderer.add<QCriterionS>(lattice, bulk_mask, obstacle);
renderer.add<VelocityNormS>(lattice, bulk_mask, obstacle);
renderer.run([&](std::size_t iStep) {
  const float tau = 0.501;
  const float smagorinsky = 0.1;
  const float inflow = 0.0075;
  
  lattice.apply(Operator(SmagorinskyBgkCollideO(), bulk_mask, tau, smagorinsky),
                Operator(BounceBackO(), boundary_mask),
                Operator(EquilibriumVelocityWallO(), inflow_mask, std::min(iStep*1e-4, 1.0)*inflow, WallNormal<1,0,0>()),
                Operator(EquilibriumDensityWallO(), outflow_mask, 1, WallNormal<-1,0,0>()));
  lattice.apply<BouzidiO>(bouzidi.getCount(), bouzidi.getConfig());
  
  lattice.stream();
});
}

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
if (cuda::device::count() == 0) {
  std::cerr << "No CUDA devices on this system" << std::endl;
  return -1;
}
auto current = cuda::device::current::get();

const descriptor::Cuboid<DESCRIPTOR> cuboid(500, 100, 100);
Lattice<DESCRIPTOR,T> lattice(cuboid);

CellMaterials<DESCRIPTOR> materials(cuboid, [&cuboid](uint3 p) -> int {
  if (p.z == 0 || p.z == cuboid.nZ-1) {
    return 2; // boundary cell
  } else if (p.y == 0 || p.y == cuboid.nY-1) {
    return 3; // boundary cell
  } else if (p.x == 0) {
    return 4; // inflow cell
  } else if (p.x == cuboid.nX-1) {
    return 5; // outflow cell
  } else {
    return 1; // bulk
  }
});

for (std::size_t iX=0; iX < cuboid.nX; ++iX) {
  materials.set(gid(cuboid, iX, 0,           0), 6);
  materials.set(gid(cuboid, iX, cuboid.nY-1, 0), 6);
  materials.set(gid(cuboid, iX, 0,           cuboid.nZ-1), 6);
  materials.set(gid(cuboid, iX, cuboid.nY-1, cuboid.nZ-1), 6);
}

auto obstacle = [cuboid] __host__ __device__ (float3 p) -> float {
  p -= make_float3(cuboid.nX/5, cuboid.nY/2, cuboid.nZ/2);
  float3 q = sdf::twisted(p, 0.01);
  return sdf::sphere(p, cuboid.nY/3.5) + sin(0.2*q.x)*sin(0.2*q.y)*sin(0.2*q.z);
};

materials.sdf(obstacle, 0);
SignedDistanceBoundary bouzidi(lattice, materials, obstacle, 1, 0);

auto bulk_mask    = materials.mask_of_material(1);
auto wall_mask_z  = materials.mask_of_material(2);
auto wall_mask_y  = materials.mask_of_material(3);
auto inflow_mask  = materials.mask_of_material(4);
auto outflow_mask = materials.mask_of_material(5);
auto edge_mask    = materials.mask_of_material(6);

cuda::synchronize(current);

VolumetricExample renderer(cuboid);
renderer.add<QCriterionS>(lattice, bulk_mask, obstacle);
renderer.add<CurlNormS>(lattice, bulk_mask, obstacle);
renderer.add<VelocityNormS>(lattice, bulk_mask, obstacle);
renderer.run([&](std::size_t iStep) {
  const float tau = 0.501;
  const float smagorinsky = 0.1;
  const float inflow = 0.04;
  
  lattice.apply(Operator(SmagorinskyBgkCollideO(), bulk_mask, tau, smagorinsky),
                Operator(BounceBackFreeSlipO(), wall_mask_z, WallNormal<0,0,1>()),
                Operator(BounceBackFreeSlipO(), wall_mask_y, WallNormal<0,1,0>()),
                Operator(EquilibriumVelocityWallO(), inflow_mask, std::min(iStep*1e-4, 1.0)*inflow, WallNormal<1,0,0>()),
                Operator(EquilibriumDensityWallO(), outflow_mask, 1, WallNormal<-1,0,0>()),
                Operator(BounceBackO(), edge_mask));
  lattice.apply<BouzidiO>(bouzidi.getCount(), bouzidi.getConfig());
  
  lattice.stream();
});
}

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

const descriptor::Cuboid<DESCRIPTOR> cuboid(100, 100, 100);
Lattice<DESCRIPTOR,T> lattice(cuboid);

CellMaterials<DESCRIPTOR> materials(cuboid, [&cuboid](uint3 p) -> int {
  if (p.x == 0 || p.x == cuboid.nX-1 || p.y == 0 || p.y == cuboid.nY-1 || p.z == cuboid.nZ-1) {
    return 2; // boundary cell
  } else if (p.z == 0) {
    return 3; // lid cell
  } else {
    return 1; // bulk
  }
});

auto bulk_mask = materials.mask_of_material(1);
auto wall_mask = materials.mask_of_material(2);
auto lid_mask  = materials.mask_of_material(3);

lattice.apply(Operator(InitializeO(), bulk_mask),
              Operator(InitializeO(), wall_mask),
              Operator(InitializeO(), lid_mask));

cudaDeviceSynchronize();

auto none = [] __device__ (float3) -> float { return 1; };
VolumetricExample renderer(cuboid);
renderer.add<CurlNormS>(lattice, bulk_mask, none);
renderer.add<ShearLayerVisibilityS>(lattice, bulk_mask, none, make_float3(0,1,0));
renderer.add<VelocityNormS>(lattice, bulk_mask, none);
renderer.run([&](std::size_t iStep) {
  const float tau = 0.56;
  const float lid = 0.10;
  
  lattice.apply(Operator(BgkCollideO(), bulk_mask, tau),
                Operator(BounceBackO(), wall_mask),
                Operator(BounceBackMovingWallO(), lid_mask, std::min(iStep*1e-3, 1.0)*lid, 0.f, 0.f));
  
  lattice.stream();
});
}

#include <LLBM/base.h>
#include <LLBM/bulk.h>
#include <LLBM/boundary.h>

#include "util/render_window.h"
#include "util/texture.h"
#include "util/colormap.h"

#include <LLBM/kernel/collect_moments.h>
#include <LLBM/kernel/collect_velocity_norm.h>

using T = float;
using DESCRIPTOR = descriptor::D2Q9;

int main() {
if (cuda::device::count() == 0) {
  std::cerr << "No CUDA devices on this system" << std::endl;
  return -1;
}
auto current = cuda::device::current::get();

const descriptor::Cuboid<DESCRIPTOR> cuboid(1200, 500);
Lattice<DESCRIPTOR,T> lattice(cuboid);

const float tau = 0.52;
const float u_inflow = 0.02;
const float u_rotate = 0.08;

CellMaterials<DESCRIPTOR> materials(cuboid, [&cuboid](uint2 p) -> int {
  if (p.x == 0) {
    return 3; // inflow
  } else if (p.x == cuboid.nX-1) {
    return 4; // outflow
  } else if (p.y == 0 || p.y == cuboid.nY-1) {
    return 2; // wall
  } else {
    return 1; // bulk
  }
});

materials.set(gid(cuboid, 0,0), 2);
materials.set(gid(cuboid, 0,cuboid.nY-1), 2);
materials.set(gid(cuboid, cuboid.nX-1,0), 5);
materials.set(gid(cuboid, cuboid.nX-1,cuboid.nY-1), 5);

auto cylinder = [cuboid] __host__ __device__ (float2 p) -> float {
                  float2 q = p - make_float2(cuboid.nX/6, 3*cuboid.nY/4);
                  float2 r = p - make_float2(cuboid.nX/6, 1*cuboid.nY/4);
                  return sdf::add(sdf::sphere(q, cuboid.nY/18),
                                  sdf::sphere(r, cuboid.nY/18));
                };

materials.sdf(cylinder, 0);
SignedDistanceBoundary bouzidi(lattice, materials, cylinder, 1, 0);

bouzidi.setVelocity([cuboid,u_rotate](float2 p) -> float2 {
  float2 q = p - make_float2(cuboid.nX/6, 3*cuboid.nY/4);
  if (length(q) < 1.1*cuboid.nY/18) {
    return u_rotate * normalize(make_float2(-q.y, q.x));
  } else {
    return make_float2(0);
  }
});

auto bulk_mask = materials.mask_of_material(1);
auto wall_mask = materials.mask_of_material(2);
auto inflow_mask  = materials.mask_of_material(3);
auto outflow_mask = materials.mask_of_material(4);
auto edge_mask = materials.mask_of_material(5);

cuda::synchronize(current);

RenderWindow window("Magnus");
cudaSurfaceObject_t colormap;
ColorPalette palette(colormap);
DeviceBuffer<T> moments_rho(cuboid.volume);
DeviceBuffer<T> moments_u(2*cuboid.volume);
T* u = moments_u.device();

std::size_t iStep = 0;
while (window.isOpen()) {
  lattice.apply(Operator(BgkCollideO(), bulk_mask, tau),
                Operator(BounceBackFreeSlipO(), wall_mask, WallNormal<0,1>()),
                Operator(EquilibriumVelocityWallO(), inflow_mask, std::min(iStep*1e-5, 1.)*u_inflow, WallNormal<1,0>()),
                Operator(EquilibriumDensityWallO(), outflow_mask, 1., WallNormal<-1,0>()),
                Operator(BounceBackO(), edge_mask));
  lattice.apply<BouzidiO>(bouzidi.getCount(), bouzidi.getConfig());
  lattice.stream();
  if (iStep % 200 == 0) {
    cuda::synchronize(current);
    lattice.inspect<CollectMomentsF>(bulk_mask, moments_rho.device(), moments_u.device());
    renderSliceViewToTexture<<<
      dim3(cuboid.nX / 32 + 1, cuboid.nY / 32 + 1),
      dim3(32,32)
    >>>(cuboid.nX, cuboid.nY,
        [cuboid] __device__ (int iX, int iY) -> std::size_t {
          return descriptor::gid(cuboid,iX,cuboid.nY-1-iY);
        },
        [u,u_rotate] __device__ (std::size_t gid) -> float {
          return length(make_float2(u[2*gid+0], u[2*gid+1])) / u_rotate;
        },
        [colormap] __device__ (float x) -> float3 {
          return colorFromTexture(colormap, clamp(x, 0.f, 1.f));
        },
        window.getRenderSurface());
    window.draw([&]() {
      ImGui::Begin("Render");
      palette.interact();
      ImGui::End();
    }, [](sf::Event&) { });
  }
  ++iStep;
}
}

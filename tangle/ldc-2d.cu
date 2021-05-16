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
cudaSetDevice(0);

const descriptor::Cuboid<DESCRIPTOR> cuboid(500, 500);
Lattice<DESCRIPTOR,T> lattice(cuboid);

CellMaterials<DESCRIPTOR> materials(cuboid, [&cuboid](uint2 p) -> int {
  if (p.x == 0 || p.y == 0 || p.x == cuboid.nX-1) {
    return 2; // boundary cell
  } else if (p.y == cuboid.nY-1) {
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

const float tau = 0.51;
const float u_lid = 0.05;

RenderWindow window("LDC");
cudaSurfaceObject_t colormap;
ColorPalette palette(colormap);
auto slice = [cuboid] __device__ (int iX, int iY) -> std::size_t {
               return descriptor::gid(cuboid,iX,cuboid.nY-1-iY);
             };
DeviceBuffer<T> moments_rho(cuboid.volume);
DeviceBuffer<T> moments_u(2*cuboid.volume);
T* u = moments_u.device();
std::size_t iStep = 0;

while (window.isOpen()) {
  lattice.apply(Operator(BgkCollideO(), bulk_mask, tau),
                Operator(BounceBackO(), wall_mask),
                Operator(BounceBackMovingWallO(), lid_mask, std::min(iStep*1e-3, 1.0)*u_lid, 0.f));
  lattice.stream();
  if (iStep % 100 == 0) {
    cudaDeviceSynchronize();
    lattice.inspect<CollectMomentsF>(bulk_mask, moments_rho.device(), moments_u.device());
    renderSliceViewToTexture<<<
      dim3(cuboid.nX / 32 + 1, cuboid.nY / 32 + 1),
      dim3(32,32)
    >>>(cuboid.nX, cuboid.nY,
        slice,
        [u,u_lid] __device__ (std::size_t gid) -> float {
          return length(make_float2(u[2*gid+0], u[2*gid+1])) / u_lid;
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

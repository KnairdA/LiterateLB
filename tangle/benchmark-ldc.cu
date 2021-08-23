#include <LLBM/base.h>

#include <LLBM/kernel/collide.h>
#include <LLBM/kernel/bounce_back.h>
#include <LLBM/kernel/bounce_back_moving_wall.h>

#include "util/timer.h"

#include <iostream>

using DESCRIPTOR = descriptor::D3Q19;

template <typename T>
void simulate(descriptor::Cuboid<DESCRIPTOR> cuboid, std::size_t nStep) {
  cudaSetDevice(0);

  Lattice<DESCRIPTOR,T> lattice(cuboid);
  
  CellMaterials<DESCRIPTOR> materials(cuboid, [&cuboid](uint3 p) -> int {
    if (p.x == 0 || p.x == cuboid.nX-1 || p.y == 0 || p.y == cuboid.nY-1 || p.z == 0) {
      return 2; // boundary cell
    } else if (p.z == cuboid.nZ-1) {
      return 3; // lid cell
    } else {
      return 1; // bulk
    }
  });
  
  auto bulk_mask = materials.mask_of_material(1);
  auto box_mask  = materials.mask_of_material(2);
  auto lid_mask  = materials.mask_of_material(3);
  
  cudaDeviceSynchronize();

  for (std::size_t iStep=0; iStep < 100; ++iStep) {
    lattice.apply(Operator(BgkCollideO(), bulk_mask, 0.56),
                  Operator(BounceBackO(), box_mask),
                  Operator(BounceBackMovingWallO(), lid_mask, 0.05f, 0.f, 0.f));
    lattice.stream();
  }

  cudaDeviceSynchronize();

  auto start = timer::now();

  for (std::size_t iStep=0; iStep < nStep; ++iStep) {
    lattice.apply(Operator(BgkCollideO(), bulk_mask, 0.56),
                  Operator(BounceBackO(), box_mask),
                  Operator(BounceBackMovingWallO(), lid_mask, 0.05f, 0.f, 0.f));
    lattice.stream();
  }

  cudaDeviceSynchronize();

  auto mlups = timer::mlups(cuboid.volume, nStep, start);

  std::cout << sizeof(T) << ", " << cuboid.nX << ", " << nStep << ", " << mlups << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Invalid parameter count" << std::endl;
    return -1;
  }

  const std::size_t n     = atoi(argv[1]);
  const std::size_t steps = atoi(argv[2]);

  unsigned precision = 4;
  if (argc == 4) {
    precision = atoi(argv[3]);
  }

  switch (precision) {
  case 4:
    simulate<float>({ n, n, n}, steps);
    break;
  case 8:
    simulate<double>({ n, n, n}, steps);
    break;
  default:
    std::cerr << "Invalid precision" << std::endl;
    return -1;
  }

  return 0;
}

#pragma once
#include <LLBM/call_tag.h>

struct CollectCurlF {

using call_tag = tag::call_by_spatial_cell_mask;

template <typename T, typename S>
__device__ static void apply(
    descriptor::D3Q19
  , S f_curr[19]
  , descriptor::CuboidD<3> cuboid
  , std::size_t gid
  , std::size_t iX
  , std::size_t iY
  , std::size_t iZ
  , S* moments_u
  , cudaSurfaceObject_t surface
  , S* curl_norm = nullptr
) {
  auto u_x = [moments_u,cuboid,gid] __device__ (int x, int y, int z) -> T {
    return moments_u[3*(gid + descriptor::offset(cuboid,x,y,z)) + 0];
  };
  auto u_y = [moments_u,cuboid,gid] __device__ (int x, int y, int z) -> T {
    return moments_u[3*(gid + descriptor::offset(cuboid,x,y,z)) + 1];
  };
  auto u_z = [moments_u,cuboid,gid] __device__ (int x, int y, int z) -> T {
    return moments_u[3*(gid + descriptor::offset(cuboid,x,y,z)) + 2];
  };

  T curl_0 = T{0.500000000000000}*u_y(0, 0, -1) - T{0.500000000000000}*u_y(0, 0, 1) - T{0.500000000000000}*u_z(0, -1, 0) + T{0.500000000000000}*u_z(0, 1, 0);
  T curl_1 = -T{0.500000000000000}*u_x(0, 0, -1) + T{0.500000000000000}*u_x(0, 0, 1) + T{0.500000000000000}*u_z(-1, 0, 0) - T{0.500000000000000}*u_z(1, 0, 0);
  T curl_2 = T{0.500000000000000}*u_x(0, -1, 0) - T{0.500000000000000}*u_x(0, 1, 0) - T{0.500000000000000}*u_y(-1, 0, 0) + T{0.500000000000000}*u_y(1, 0, 0);
  float3 curl = make_float3(curl_0, curl_1, curl_2);
  float norm = length(curl);

  surf3Dwrite(norm, surface, iX*sizeof(float), iY, iZ);

  if (curl_norm != nullptr) {
    curl_norm[gid] = norm; 
  }
}

};

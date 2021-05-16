#pragma once
#include <LLBM/call_tag.h>

struct CollectVelocityNormF {

using call_tag = tag::post_process_by_spatial_cell_mask;

template <typename T, typename S>
__device__ static void apply(
    descriptor::D2Q9
  , std::size_t gid
  , std::size_t iX
  , std::size_t iY
  , std::size_t iZ
  , T* u
  , cudaSurfaceObject_t surface
) {
  float norm = length(make_float2(u[2*gid+0], u[2*gid+1]));
  surf2Dwrite(norm, surface, iX*sizeof(float), iY);
}

template <typename T, typename S>
__device__ static void apply(
    descriptor::D3Q19
  , std::size_t gid
  , std::size_t iX
  , std::size_t iY
  , std::size_t iZ
  , T* u
  , cudaSurfaceObject_t surface
  , T* u_norm = nullptr
) {
  float norm = length(make_float3(u[3*gid+0], u[3*gid+1], u[3*gid+2]));
  surf3Dwrite(norm, surface, iX*sizeof(float), iY, iZ);
  if (u_norm != nullptr) {
    u_norm[gid] = norm;
  }
}

};

template <typename SLICE, typename SAMPLE, typename PALETTE>
__global__ void renderSliceViewToTexture(std::size_t width, std::size_t height, SLICE slice, SAMPLE sample, PALETTE palette, cudaSurfaceObject_t texture) {
  const int screenX = threadIdx.x + blockIdx.x * blockDim.x;
  const int screenY = threadIdx.y + blockIdx.y * blockDim.y;

  if (screenX > width-1 || screenY > height-1) {
    return;
  }
  
  const std::size_t gid = slice(screenX,screenY);
  float3 color = palette(sample(gid));

  uchar4 pixel {
    static_cast<unsigned char>(color.x * 255),
    static_cast<unsigned char>(color.y * 255),
    static_cast<unsigned char>(color.z * 255),
    255
  };
  surf2Dwrite(pixel, texture, screenX*sizeof(uchar4), screenY);
}

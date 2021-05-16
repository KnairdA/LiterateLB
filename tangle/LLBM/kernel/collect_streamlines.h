#pragma once

__device__ void writeDot(cudaSurfaceObject_t texture, int x, int y, uchar4 pixel) {
  surf2Dwrite(pixel, texture, (x-1)*sizeof(uchar4), (y  ));
  surf2Dwrite(pixel, texture, (x-1)*sizeof(uchar4), (y+1));
  surf2Dwrite(pixel, texture, (x  )*sizeof(uchar4), (y  ));
  surf2Dwrite(pixel, texture, (x  )*sizeof(uchar4), (y+1));
} 

__device__ float2 readVelocity(cudaSurfaceObject_t u_x, cudaSurfaceObject_t u_y, float2 p) {
  return make_float2(tex2D<float>(u_x, p.x, p.y), tex2D<float>(u_y, p.x, p.y));
}

template <typename DESCRIPTOR, typename T>
__global__ void renderStreamlinesToTexture(
    descriptor::Cuboid<DESCRIPTOR> cuboid
  , cudaSurfaceObject_t u_x
  , cudaSurfaceObject_t u_y
  , T* origins
  , unsigned nSteps
  , float charU
  , cudaSurfaceObject_t texture
)  {
  const int iOrigin = threadIdx.x + blockIdx.x * blockDim.x;

  float h = 1;
  float2 p = make_float2(origins[2*iOrigin+0], origins[2*iOrigin+1]);

  for (unsigned iStep = 0; iStep < nSteps; ++iStep) {
    float2 u = make_float2(tex2D<float>(u_x, p.x, p.y), tex2D<float>(u_y, p.x, p.y));

    float2 k1 = readVelocity(u_x, u_y, p) / charU;
    float2 k2 = readVelocity(u_x, u_y, p + 0.5f*h * k1) / charU;
    float2 k3 = readVelocity(u_x, u_y, p + 0.5f*h * k2) / charU;
    float2 k4 = readVelocity(u_x, u_y, p +      h * k3) / charU;

    p += h * (1.f/6.f*k1 + 1.f/3.f*k2 + 1.f/3.f*k3 + 1.f/6.f*k4);

    const int screenX = std::nearbyint(p.x);
    const int screenY = cuboid.nY-1 - std::nearbyint(p.y);

    if (screenX < 1 || screenY < 1 || screenX > cuboid.nX-2 || screenY > cuboid.nY-2) {
      break;
    }

    writeDot(texture, screenX, screenY, {255,255,255,255});
  }
}

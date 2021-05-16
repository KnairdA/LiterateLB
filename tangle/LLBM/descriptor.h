#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <cuda-samples/Common/helper_math.h>

#ifdef __CUDA_ARCH__
  #define DATA device_data
#else
  #define DATA host_data
#endif

using pop_index_t = std::uint8_t;

namespace descriptor {

struct D2Q9 {
  static constexpr unsigned d = 2;
  static constexpr unsigned q = 9;
};

struct D3Q19 {
  static constexpr unsigned d = 3;
  static constexpr unsigned q = 19;
};

namespace device_data {
  template <typename DESCRIPTOR>
  __constant__ pop_index_t opposite[DESCRIPTOR::q] { };

  template <typename DESCRIPTOR>
  __constant__ int c[DESCRIPTOR::q][DESCRIPTOR::d] { };

  template <typename DESCRIPTOR>
  __constant__ float c_length[DESCRIPTOR::q] { };

  template <typename DESCRIPTOR>
  __constant__ float weight[DESCRIPTOR::q] { };

  template <>
  __constant__ pop_index_t opposite<D2Q9>[9] = {
    8, 7, 6, 5, 4, 3, 2, 1, 0
  };
  
  template <>
  __constant__ int c<D2Q9>[9][2] = {
    {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,0}, {0,1}, {1,-1}, {1,0}, {1,1}
  };
  
  template <>
  __constant__ float c_length<D2Q9>[9] = {
    1.41421356237310, 1.00000000000000, 1.41421356237310, 1.00000000000000, 0, 1.00000000000000, 1.41421356237310, 1.00000000000000, 1.41421356237310
  };
  
  template <>
  __constant__ float weight<D2Q9>[9] = {
    0.0277777777777778, 0.111111111111111, 0.0277777777777778, 0.111111111111111, 0.444444444444444, 0.111111111111111, 0.0277777777777778, 0.111111111111111, 0.0277777777777778
  };

  template <>
  __constant__ pop_index_t opposite<D3Q19>[19] = {
    18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  };
  
  template <>
  __constant__ int c<D3Q19>[19][3] = {
    {0,1,1}, {-1,0,1}, {0,0,1}, {1,0,1}, {0,-1,1}, {-1,1,0}, {0,1,0}, {1,1,0}, {-1,0,0}, {0,0,0}, {1,0,0}, {-1,-1,0}, {0,-1,0}, {1,-1,0}, {0,1,-1}, {-1,0,-1}, {0,0,-1}, {1,0,-1}, {0,-1,-1}
  };
  
  template <>
  __constant__ float c_length<D3Q19>[19] = {
    1.41421356237310, 1.41421356237310, 1.00000000000000, 1.41421356237310, 1.41421356237310, 1.41421356237310, 1.00000000000000, 1.41421356237310, 1.00000000000000, 0, 1.00000000000000, 1.41421356237310, 1.00000000000000, 1.41421356237310, 1.41421356237310, 1.41421356237310, 1.00000000000000, 1.41421356237310, 1.41421356237310
  };
  
  template <>
  __constant__ float weight<D3Q19>[19] = {
    0.0277777777777778, 0.0277777777777778, 0.0555555555555556, 0.0277777777777778, 0.0277777777777778, 0.0277777777777778, 0.0555555555555556, 0.0277777777777778, 0.0555555555555556, 0.333333333333333, 0.0555555555555556, 0.0277777777777778, 0.0555555555555556, 0.0277777777777778, 0.0277777777777778, 0.0277777777777778, 0.0555555555555556, 0.0277777777777778, 0.0277777777777778
  };
}

namespace host_data {
  template <typename DESCRIPTOR>
  constexpr pop_index_t opposite[DESCRIPTOR::q] { };

  template <typename DESCRIPTOR>
  constexpr int c[DESCRIPTOR::q][DESCRIPTOR::d] { };

  template <typename DESCRIPTOR>
  constexpr float c_length[DESCRIPTOR::q] { };

  template <typename DESCRIPTOR>
  constexpr float weight[DESCRIPTOR::q] { };

  template <>
  constexpr pop_index_t opposite<D2Q9>[9] = {
    8, 7, 6, 5, 4, 3, 2, 1, 0
  };
  
  template <>
  constexpr int c<D2Q9>[9][2] = {
    {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,0}, {0,1}, {1,-1}, {1,0}, {1,1}
  };
  
  template <>
  constexpr float c_length<D2Q9>[9] = {
    1.41421356237310, 1.00000000000000, 1.41421356237310, 1.00000000000000, 0, 1.00000000000000, 1.41421356237310, 1.00000000000000, 1.41421356237310
  };
  
  template <>
  constexpr float weight<D2Q9>[9] = {
    0.0277777777777778, 0.111111111111111, 0.0277777777777778, 0.111111111111111, 0.444444444444444, 0.111111111111111, 0.0277777777777778, 0.111111111111111, 0.0277777777777778
  };

  template <>
  constexpr pop_index_t opposite<D3Q19>[19] = {
    18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  };
  
  template <>
  constexpr int c<D3Q19>[19][3] = {
    {0,1,1}, {-1,0,1}, {0,0,1}, {1,0,1}, {0,-1,1}, {-1,1,0}, {0,1,0}, {1,1,0}, {-1,0,0}, {0,0,0}, {1,0,0}, {-1,-1,0}, {0,-1,0}, {1,-1,0}, {0,1,-1}, {-1,0,-1}, {0,0,-1}, {1,0,-1}, {0,-1,-1}
  };
  
  template <>
  constexpr float c_length<D3Q19>[19] = {
    1.41421356237310, 1.41421356237310, 1.00000000000000, 1.41421356237310, 1.41421356237310, 1.41421356237310, 1.00000000000000, 1.41421356237310, 1.00000000000000, 0, 1.00000000000000, 1.41421356237310, 1.00000000000000, 1.41421356237310, 1.41421356237310, 1.41421356237310, 1.00000000000000, 1.41421356237310, 1.41421356237310
  };
  
  template <>
  constexpr float weight<D3Q19>[19] = {
    0.0277777777777778, 0.0277777777777778, 0.0555555555555556, 0.0277777777777778, 0.0277777777777778, 0.0277777777777778, 0.0555555555555556, 0.0277777777777778, 0.0555555555555556, 0.333333333333333, 0.0555555555555556, 0.0277777777777778, 0.0555555555555556, 0.0277777777777778, 0.0277777777777778, 0.0277777777777778, 0.0555555555555556, 0.0277777777777778, 0.0277777777777778
  };
}

template <typename DESCRIPTOR>
__host__ __device__
pop_index_t opposite(pop_index_t iPop) {
  return DESCRIPTOR::q - 1 - iPop;
}

template <typename DESCRIPTOR>
__host__ __device__
int velocity(pop_index_t iPop, unsigned iDim) {
  return DATA::template c<DESCRIPTOR>[iPop][iDim];
}

template <typename DESCRIPTOR>
__host__ __device__
std::enable_if_t<DESCRIPTOR::d == 2, float2> velocity(pop_index_t iPop) {
  return make_float2(DATA::template c<DESCRIPTOR>[iPop][0],
                     DATA::template c<DESCRIPTOR>[iPop][1]);
}

template <typename DESCRIPTOR>
__host__ __device__
std::enable_if_t<DESCRIPTOR::d == 3, float3> velocity(pop_index_t iPop) {
  return make_float3(DATA::template c<DESCRIPTOR>[iPop][0],
                     DATA::template c<DESCRIPTOR>[iPop][1],
                     DATA::template c<DESCRIPTOR>[iPop][2]);
}

template <typename DESCRIPTOR>
__host__ __device__
float velocity_length(pop_index_t iPop) {
  return DATA::template c_length<DESCRIPTOR>[iPop];
}

template <typename DESCRIPTOR>
__host__ __device__
float weight(pop_index_t iPop) {
  return DATA::template weight<DESCRIPTOR>[iPop];
}

template <unsigned D>
struct CuboidD;

template <>
struct CuboidD<2> {
  const std::size_t nX;
  const std::size_t nY;
  const std::size_t nZ;
  const std::size_t volume;

  CuboidD(std::size_t x, std::size_t y):
    nX(x), nY(y), nZ(1),
    volume(x*y) { };
};

template <>
struct CuboidD<3> {
  const std::size_t nX;
  const std::size_t nY;
  const std::size_t nZ;
  const std::size_t volume;
  const std::size_t plane;

  CuboidD(std::size_t x, std::size_t y, std::size_t z):
    nX(x), nY(y), nZ(z),
    volume(x*y*z),
    plane(x*y) { };
};

template <typename DESCRIPTOR>
using Cuboid = CuboidD<DESCRIPTOR::d>;

__host__ __device__
std::size_t gid(const CuboidD<2>& c, int iX, int iY, int iZ=0) {
  return iY*c.nX + iX;
}

__host__ __device__
std::size_t gid(const CuboidD<3>& c, int iX, int iY, int iZ) {
  return iZ*c.plane + iY*c.nX + iX;
}

__host__ __device__
int offset(const CuboidD<2>& c, int iX, int iY) {
  return iY*c.nX + iX;
}

template <typename DESCRIPTOR>
__host__ __device__
int offset(const CuboidD<2>& c, pop_index_t iPop) {
  static_assert(DESCRIPTOR::d == 2, "Dimensions must match");
  return offset(c,
    descriptor::velocity<DESCRIPTOR>(iPop, 0),
    descriptor::velocity<DESCRIPTOR>(iPop, 1)
  );
}

__host__ __device__
int offset(const CuboidD<3>& c, int iX, int iY, int iZ) {
  return iZ*c.plane + iY*c.nX + iX;
}

template <typename DESCRIPTOR>
__host__ __device__
int offset(const CuboidD<3>& c, pop_index_t iPop) {
  static_assert(DESCRIPTOR::d == 3, "Dimensions must match");
  return offset(c,
    descriptor::velocity<DESCRIPTOR>(iPop, 0),
    descriptor::velocity<DESCRIPTOR>(iPop, 1),
    descriptor::velocity<DESCRIPTOR>(iPop, 2)
  );
}

template <typename DESCRIPTOR>
__host__ __device__
std::size_t neighbor(const CuboidD<2>& c, std::size_t iCell, pop_index_t iPop) {
  return iCell + offset<DESCRIPTOR>(c, iPop);
}

template <typename DESCRIPTOR>
__host__ __device__
std::size_t neighbor(const CuboidD<3>& c, std::size_t iCell, pop_index_t iPop) {
  return iCell + offset<DESCRIPTOR>(c, iPop);
}

__host__ __device__
uint2 gidInverse(const CuboidD<2>& c, std::size_t gid) {
  int iY = gid / c.nX;
  int iX = gid % c.nX;
  return make_uint2(iX, iY);
}

__host__ __device__
uint3 gidInverse(const CuboidD<3>& c, std::size_t gid) {
  int iZ = gid / c.plane;
  int iY = (gid % c.plane) / c.nX;
  int iX = (gid % c.plane) % c.nX;
  return make_uint3(iX,iY,iZ);
}

__host__ __device__
float2 gidInverseSmooth(const CuboidD<2>& c, std::size_t gid) {
  int iY = gid / c.nX;
  int iX = gid % c.nX;
  return make_float2(iX, iY);
}

__host__ __device__
float3 gidInverseSmooth(const CuboidD<3>& c, std::size_t gid) {
  int iZ = gid / c.plane;
  int iY = (gid % c.plane) / c.nX;
  int iX = (gid % c.plane) % c.nX;
  return make_float3(iX,iY,iZ);
}

bool isInside(const CuboidD<2>& c, std::size_t gid) {
  return gid < c.volume;
}

bool isInside(const CuboidD<3>& c, std::size_t gid) {
  return gid < c.volume;
}

}

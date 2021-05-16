#pragma once

#include <LLBM/operator.h>

namespace kernel {

template <typename OPERATOR, typename DESCRIPTOR, typename T, typename S, typename... ARGS>
__global__ void call_operator(
    LatticeView<DESCRIPTOR,S> lattice
  , std::size_t* cells
  , std::size_t  cell_count
  , ARGS... args
) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(index < cell_count)) {
      return;
  }
  const std::size_t gid = cells[index];

  S f_curr[DESCRIPTOR::q];
  S f_next[DESCRIPTOR::q];
  S* preshifted_f[DESCRIPTOR::q];
  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    preshifted_f[iPop] = lattice.pop(iPop, gid);
    f_curr[iPop] = *preshifted_f[iPop];
  }
  OPERATOR::template apply<T,S>(DESCRIPTOR(), f_curr, f_next, gid, std::forward<ARGS>(args)...);
  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    *preshifted_f[iPop] = f_next[iPop];
  }
}

template <typename OPERATOR, typename DESCRIPTOR, typename T, typename S, typename... ARGS>
__global__ void call_operator(
    LatticeView<DESCRIPTOR,S> lattice
  , bool* mask
  , ARGS... args
) {
  const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(gid < lattice.cuboid.volume) || !mask[gid]) {
      return;
  }

  S f_curr[DESCRIPTOR::q];
  S f_next[DESCRIPTOR::q];
  S* preshifted_f[DESCRIPTOR::q];
  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    preshifted_f[iPop] = lattice.pop(iPop, gid);
    f_curr[iPop] = *preshifted_f[iPop];
  }
  OPERATOR::template apply<T,S>(DESCRIPTOR(), f_curr, f_next, gid, std::forward<ARGS>(args)...);
  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    *preshifted_f[iPop] = f_next[iPop];
  }
}

template <typename FUNCTOR, typename DESCRIPTOR, typename T, typename S, typename... ARGS>
__global__ void call_functor(
    LatticeView<DESCRIPTOR,S> lattice
  , bool* mask
  , ARGS... args
) {
  const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(gid < lattice.cuboid.volume) || !mask[gid]) {
      return;
  }

  S f_curr[DESCRIPTOR::q];
  S* preshifted_f[DESCRIPTOR::q];
  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    preshifted_f[iPop] = lattice.pop(iPop, gid);
    f_curr[iPop] = *preshifted_f[iPop];
  }
  FUNCTOR::template apply<T,S>(DESCRIPTOR(), f_curr, gid, std::forward<ARGS>(args)...);
}

template <typename DESCRIPTOR, typename T, typename S, typename... OPERATOR>
__global__ void call_operators(
    LatticeView<DESCRIPTOR,S> lattice
  , OPERATOR... ops
) {
  const std::size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(gid < lattice.cuboid.volume)) {
      return;
  }

  S f_curr[DESCRIPTOR::q];
  S f_next[DESCRIPTOR::q];
  S* preshifted_f[DESCRIPTOR::q];
  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    preshifted_f[iPop] = lattice.pop(iPop, gid);
    f_curr[iPop] = *preshifted_f[iPop];
  }
  (ops.template apply<DESCRIPTOR,T,S>(DESCRIPTOR(), f_curr, f_next, gid) || ... || false);
  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    *preshifted_f[iPop] = f_next[iPop];
  }
}

template <typename OPERATOR, typename DESCRIPTOR, typename T, typename S, typename... ARGS>
__global__ void call_operator_using_list(
    LatticeView<DESCRIPTOR,S> lattice
  , std::size_t count
  , ARGS... args
) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(index < count)) {
      return;
  }
  OPERATOR::template apply<T,S>(lattice, index, count, std::forward<ARGS>(args)...);
}

template <typename OPERATOR, typename DESCRIPTOR, typename T, typename S, typename... ARGS>
__global__ void call_operator_using_list(
    DESCRIPTOR descriptor
  , std::size_t count
  , ARGS... args
) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(index < count)) {
      return;
  }
  OPERATOR::template apply<T,S>(descriptor, index, count, std::forward<ARGS>(args)...);
}

template <typename FUNCTOR, typename DESCRIPTOR, typename T, typename S, typename... ARGS>
__global__ void call_spatial_functor(
    LatticeView<DESCRIPTOR,S> lattice
  , bool* mask
  , ARGS... args
) {
  const std::size_t iX = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t iY = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t iZ = blockIdx.z * blockDim.z + threadIdx.z;
  if (!(iX < lattice.cuboid.nX && iY < lattice.cuboid.nY && iZ < lattice.cuboid.nZ)) {
      return;
  }
  const std::size_t gid = descriptor::gid(lattice.cuboid,iX,iY,iZ);
  if (!mask[gid]) {
      return;
  }

  S f_curr[DESCRIPTOR::q];
  S* preshifted_f[DESCRIPTOR::q];
  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    preshifted_f[iPop] = lattice.pop(iPop, gid);
    f_curr[iPop] = *preshifted_f[iPop];
  }
  FUNCTOR::template apply<T,S>(DESCRIPTOR(), f_curr, lattice.cuboid, gid, iX, iY, iZ, std::forward<ARGS>(args)...);
}

template <typename OPERATOR, typename DESCRIPTOR, typename T, typename S, typename... ARGS>
__global__ void call_spatial_operator(
    descriptor::Cuboid<DESCRIPTOR> cuboid
  , bool* mask
  , ARGS... args
) {
  const std::size_t iX = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t iY = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t iZ = blockIdx.z * blockDim.z + threadIdx.z;
  if (!(iX < cuboid.nX && iY < cuboid.nY && iZ < cuboid.nZ)) {
      return;
  }
  const std::size_t gid = descriptor::gid(cuboid,iX,iY,iZ);
  if (!mask[gid]) {
      return;
  }
  OPERATOR::template apply<T,S>(DESCRIPTOR(), gid, iX, iY, iZ, std::forward<ARGS>(args)...);
}

}

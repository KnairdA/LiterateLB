#pragma once

#include "memory.h"
#include "descriptor.h"
#include "kernel/propagate.h"

#include <cuda.h>

template <typename DESCRIPTOR, typename S>
struct LatticeView {
  const descriptor::Cuboid<DESCRIPTOR> cuboid;
  S** population;

  __device__ __forceinline__
  S* pop(pop_index_t iPop, std::size_t gid) const;
};

template <typename DESCRIPTOR, typename S>
class CyclicPopulationBuffer {
protected:
  const descriptor::Cuboid<DESCRIPTOR> _cuboid;

  const std::size_t _page_size;
  const std::size_t _volume;

  CUmemGenericAllocationHandle _handle[DESCRIPTOR::q];
  CUmemAllocationProp _prop{};
  CUmemAccessDesc _access{};
  CUdeviceptr _ptr;

  SharedVector<S*> _base;
  SharedVector<S*> _population;

  S* device() {
    return reinterpret_cast<S*>(_ptr);
  }

public:
  CyclicPopulationBuffer(descriptor::Cuboid<DESCRIPTOR> cuboid);

  LatticeView<DESCRIPTOR,S> view() {
    return LatticeView<DESCRIPTOR,S>{ _cuboid, _population.device() };
  }

  void stream();

};

std::size_t getDevicePageSize(int device_id=-1) {
  if (device_id == -1) {
    cudaGetDevice(&device_id);
  }
  std::size_t granularity = 0;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  return granularity;
}

template <typename DESCRIPTOR, typename S>
CyclicPopulationBuffer<DESCRIPTOR,S>::CyclicPopulationBuffer(
  descriptor::Cuboid<DESCRIPTOR> cuboid):
  _cuboid(cuboid),
  _page_size{getDevicePageSize()},
  _volume{((cuboid.volume * sizeof(S) - 1) / _page_size + 1) * _page_size},
  _base(DESCRIPTOR::q),
  _population(DESCRIPTOR::q)
{

  int device_id = -1;
  cudaGetDevice(&device_id);

  _prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  _prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  _prop.location.id = device_id;
  cuMemAddressReserve(&_ptr, 2 * _volume * DESCRIPTOR::q, 0, 0, 0);

  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    // per-population handle until cuMemMap accepts non-zero offset
    cuMemCreate(&_handle[iPop], _volume, &_prop, 0); 
    cuMemMap(_ptr + iPop * 2 * _volume,           _volume, 0, _handle[iPop], 0);
    cuMemMap(_ptr + iPop * 2 * _volume + _volume, _volume, 0, _handle[iPop], 0);
  }

  _access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  _access.location.id = 0;
  _access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(_ptr, 2 * _volume * DESCRIPTOR::q, &_access, 1);
  cuMemsetD8(_ptr, 0, 2 * _volume * DESCRIPTOR::q);

  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    _base[iPop] = device() + iPop * 2 * (_volume / sizeof(S));
    _population[iPop] = _base[iPop] + iPop * ((_volume / sizeof(S)) / DESCRIPTOR::q);
  }

  _base.syncDeviceFromHost();
  _population.syncDeviceFromHost();
}

template <typename DESCRIPTOR, typename S>
__device__ __forceinline__
S* LatticeView<DESCRIPTOR,S>::pop(pop_index_t iPop, std::size_t gid) const {
  return population[iPop] + gid;
}

template <typename DESCRIPTOR, typename S>
void CyclicPopulationBuffer<DESCRIPTOR,S>::stream() {
  propagate<DESCRIPTOR,S><<<1,1>>>(view(), _base.device(), _volume / sizeof(S));
}

#pragma once
#include <LLBM/memory.h>
#include <LLBM/materials.h>
#include <LLBM/kernel/bouzidi.h>
#include <iostream>

template <typename DESCRIPTOR, typename T, typename S, typename SDF>
class SignedDistanceBoundary {
private:
const descriptor::Cuboid<DESCRIPTOR> _cuboid;
const std::size_t _count;

SharedVector<std::size_t> _boundary;
SharedVector<std::size_t> _fluid;
SharedVector<std::size_t> _solid;
SharedVector<S> _distance;
SharedVector<S> _correction;
SharedVector<S> _factor;
SharedVector<pop_index_t> _missing;

void set(std::size_t index, std::size_t iCell, pop_index_t iPop, S dist) {
  pop_index_t jPop = descriptor::opposite<DESCRIPTOR>(iPop);
  const std::size_t jPopCell = descriptor::neighbor<DESCRIPTOR>(_cuboid, iCell, jPop);
  const std::size_t iPopCell = descriptor::neighbor<DESCRIPTOR>(_cuboid, iCell, iPop);

  _boundary[index] = iCell;
  _solid[index] = jPopCell;
  _distance[index] = dist;
  _correction[index] = 0;
  _missing[index] = iPop;

  T q = dist / descriptor::velocity_length<DESCRIPTOR>(iPop);
  if (q > 0.5) {
    _fluid[index] = iCell;
    _factor[index] = 1 / (2*q);
  } else {
    _fluid[index] = iPopCell;
    _factor[index] = 2*q;
  }
}

void syncDeviceFromHost() {
  _boundary.syncDeviceFromHost();
  _fluid.syncDeviceFromHost();
  _solid.syncDeviceFromHost();
  _distance.syncDeviceFromHost();
  _correction.syncDeviceFromHost();
  _factor.syncDeviceFromHost();
  _missing.syncDeviceFromHost();
}

public:
SignedDistanceBoundary(Lattice<DESCRIPTOR,T,S>&, CellMaterials<DESCRIPTOR>& materials, SDF geometry, int bulk, int solid):
  _cuboid(materials.cuboid()),
  _count(materials.get_link_count(bulk, solid)),
  _boundary(_count),
  _fluid(_count),
  _solid(_count),
  _distance(_count),
  _correction(_count),
  _factor(_count),
  _missing(_count)
{
  std::size_t index = 0;
  materials.for_links(bulk, solid, [&](std::size_t iCell, pop_index_t iPop) {
    auto p         = gidInverseSmooth(_cuboid, iCell);
    auto direction = normalize(descriptor::velocity<DESCRIPTOR>(iPop));
    float length   = descriptor::velocity_length<DESCRIPTOR>(iPop);
    float distance = approximateDistance(geometry, p, direction, 0, length);
    if (distance == 0.f || distance > length) {
      std::cout << "Bogus distance d=" << distance << " at cell " << iCell
                << " in direction " << std::to_string(iPop) << std::endl;
    }
    set(index++, iCell, descriptor::opposite<DESCRIPTOR>(iPop), distance);
  });
  syncDeviceFromHost();
}

template <typename VELOCITY>
void setVelocity(VELOCITY field) {
  for (std::size_t index=0; index < _count; ++index) {
    pop_index_t jPop = descriptor::opposite<DESCRIPTOR>(_missing[index]);
    auto direction = normalize(descriptor::velocity<DESCRIPTOR>(jPop));
    float length = descriptor::velocity_length<DESCRIPTOR>(jPop);
    auto p = descriptor::gidInverseSmooth(_cuboid, _boundary[index]);
    auto u_w = field(p + _distance[index] * direction);
    _correction[index] = 2*3*descriptor::weight<DESCRIPTOR>(jPop)
                       * dot(u_w, descriptor::velocity<DESCRIPTOR>(jPop));
    if (_distance[index] / length > 0.5) {
      _correction[index] *= _factor[index];
    }
  }
  _correction.syncDeviceFromHost();
}

std::size_t getCount() const {
  return _count;
}

BouzidiConfig<S> getConfig() {
  return BouzidiConfig<S>{
    _boundary.device(),
    _solid.device(),
    _fluid.device(),
    _factor.device(),
    _correction.device(),
    _missing.device()
  };
}

};

template <typename DESCRIPTOR, typename T, typename S, typename SDF>
SignedDistanceBoundary(Lattice<DESCRIPTOR,T,S>&, CellMaterials<DESCRIPTOR>&, SDF, int, int) -> SignedDistanceBoundary<DESCRIPTOR,T,S,SDF>;

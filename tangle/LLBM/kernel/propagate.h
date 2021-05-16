#pragma once

template <typename DESCRIPTOR, typename S>
class LatticeView;

template <typename DESCRIPTOR, typename S>
__global__ void propagate(LatticeView<DESCRIPTOR,S> lattice, S** base, std::size_t size) {

  for (unsigned iPop=0; iPop < DESCRIPTOR::q; ++iPop) {
    std::ptrdiff_t shift = -descriptor::offset<DESCRIPTOR>(lattice.cuboid, iPop);

    lattice.population[iPop] += shift;

    if (lattice.population[iPop] < base[iPop]) {
      lattice.population[iPop] += size;
    } else if (lattice.population[iPop] + size > base[iPop] + 2*size) {
      lattice.population[iPop] -= size;
    }
  }
}

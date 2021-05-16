#pragma once
#include <LLBM/call_tag.h>
#include <LLBM/lattice.h>

template <typename S>
struct BouzidiConfig {
  std::size_t* boundary; // boundary cell to be interpolated
  std::size_t* solid;    // adjacent solid cell
  std::size_t* fluid;    // adjacent fluid cell
  S* distance;           // precomputed distance factor q
  S* correction;         // correction for moving walls
  pop_index_t* missing;  // population to be reconstructed
};

struct BouzidiO {

using call_tag = tag::call_by_list_index;

template <typename T, typename S, typename DESCRIPTOR>
__device__ static void apply(
    LatticeView<DESCRIPTOR,S> lattice
  , std::size_t index
  , std::size_t count
  , BouzidiConfig<S> config
) {
  pop_index_t& iPop = config.missing[index];
  pop_index_t  jPop = descriptor::opposite<DESCRIPTOR>(iPop);
  pop_index_t  kPop = config.boundary[index] == config.fluid[index] ? iPop : jPop;

  S f_bound_j = *lattice.pop(jPop, config.boundary[index]);
  S f_fluid_j = *lattice.pop(kPop, config.fluid[index]);
  S* f_next_i =  lattice.pop(iPop, config.solid[index]);

  *f_next_i = config.distance[index] * f_bound_j
            + (1. - config.distance[index]) * f_fluid_j
            + config.correction[index];
}

};

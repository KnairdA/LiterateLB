#pragma once

#include "memory.h"
#include "call_tag.h"
#include "operator.h"

#include "propagate.h"
#include "kernel/executor.h"

template <typename DESCRIPTOR, typename T, typename S=T>
class Lattice {
private:
const descriptor::Cuboid<DESCRIPTOR> _cuboid;

CyclicPopulationBuffer<DESCRIPTOR,S> _population;

public:
Lattice(descriptor::Cuboid<DESCRIPTOR> cuboid):
  _cuboid(cuboid),
  _population(cuboid) { }

descriptor::Cuboid<DESCRIPTOR> cuboid() const {
  return _cuboid;
}

void stream() {
  _population.stream();
}

template <typename... OPERATOR>
void apply(OPERATOR... ops) {
  const auto block_size = 32;
  const auto block_count = (_cuboid.volume + block_size - 1) / block_size;
  kernel::call_operators<DESCRIPTOR,T,S,OPERATOR...><<<block_count,block_size>>>(
    _population.view(), ops...
  );
}

template <typename OPERATOR, typename... ARGS>
void apply(ARGS&&... args) {
  call_operator<OPERATOR>(typename OPERATOR::call_tag{}, std::forward<ARGS&&>(args)...);
}

template <typename OPERATOR, typename... ARGS>
void call_operator(tag::call_by_cell_id, DeviceBuffer<std::size_t>& cells, ARGS... args) {
  const auto block_size = 32;
  const auto block_count = (cells.size() + block_size - 1) / block_size;
  kernel::call_operator<OPERATOR,DESCRIPTOR,T,S,ARGS...><<<block_count,block_size>>>(
    _population.view(), cells.device(), cells.size(), std::forward<ARGS>(args)...
  );
}

template <typename OPERATOR, typename... ARGS>
void call_operator(tag::call_by_cell_id, DeviceBuffer<bool>& mask, ARGS... args) {
  const auto block_size = 32;
  const auto block_count = (_cuboid.volume + block_size - 1) / block_size;
  kernel::call_operator<OPERATOR,DESCRIPTOR,T,S,ARGS...><<<block_count,block_size>>>(
    _population.view(), mask.device(), std::forward<ARGS>(args)...
  );
}

template <typename OPERATOR, typename... ARGS>
void call_operator(tag::call_by_list_index, std::size_t count, ARGS... args) {
  const auto block_size = 32;
  const auto block_count = (count + block_size - 1) / block_size;
  kernel::call_operator_using_list<OPERATOR,DESCRIPTOR,T,S,ARGS...><<<block_count,block_size>>>(
    _population.view(), count, std::forward<ARGS>(args)...
  );
}

template <typename FUNCTOR, typename... ARGS>
void inspect(ARGS&&... args) {
  call_functor<FUNCTOR>(typename FUNCTOR::call_tag{}, std::forward<ARGS&&>(args)...);
}

template <typename FUNCTOR, typename... ARGS>
void call_functor(tag::call_by_cell_id, DeviceBuffer<std::size_t>& cells, ARGS... args) {
  const auto block_size = 32;
  const auto block_count = (cells.size() + block_size - 1) / block_size;
  kernel::call_functor<FUNCTOR,DESCRIPTOR,T,S,ARGS...><<<block_count,block_size>>>(
    _population.view(), cells.device(), cells.size(), std::forward<ARGS>(args)...
  );
}

template <typename FUNCTOR, typename... ARGS>
void call_functor(tag::call_by_cell_id, DeviceBuffer<bool>& mask, ARGS... args) {
  const auto block_size = 32;
  const auto block_count = (_cuboid.volume + block_size - 1) / block_size;
  kernel::call_functor<FUNCTOR,DESCRIPTOR,T,S,ARGS...><<<block_count,block_size>>>(
    _population.view(), mask.device(), std::forward<ARGS>(args)...
  );
}

template <typename FUNCTOR, typename... ARGS>
void call_functor(tag::call_by_spatial_cell_mask, DeviceBuffer<bool>& mask, ARGS... args) {
  const dim3 block(32,8,4);
  const dim3 grid((_cuboid.nX + block.x - 1) / block.x,
                  (_cuboid.nY + block.y - 1) / block.y,
                  (_cuboid.nZ + block.z - 1) / block.z);
  kernel::call_spatial_functor<FUNCTOR,DESCRIPTOR,T,S,ARGS...><<<grid,block>>>(
    _population.view(), mask.device(), std::forward<ARGS>(args)...
  );
}

template <typename OPERATOR, typename... ARGS>
void helper(ARGS&&... args) {
  tagged_helper<OPERATOR>(typename OPERATOR::call_tag{}, std::forward<ARGS&&>(args)...);
}

template <typename OPERATOR, typename... ARGS>
void tagged_helper(tag::post_process_by_list_index, std::size_t count, ARGS... args) {
  const auto block_size = 32;
  const auto block_count = (count + block_size - 1) / block_size;
  kernel::call_operator_using_list<OPERATOR,DESCRIPTOR,T,S,ARGS...><<<block_count,block_size>>>(
    DESCRIPTOR(), count, std::forward<ARGS>(args)...
  );
}

template <typename OPERATOR, typename... ARGS>
void tagged_helper(tag::post_process_by_spatial_cell_mask, DeviceBuffer<bool>& mask, ARGS... args) {
  const dim3 block(32,8,4);
  const dim3 grid((_cuboid.nX + block.x - 1) / block.x,
                  (_cuboid.nY + block.y - 1) / block.y,
                  (_cuboid.nZ + block.z - 1) / block.z);
  kernel::call_spatial_operator<OPERATOR,DESCRIPTOR,T,S,ARGS...><<<grid,block>>>(
    _cuboid, mask.device(), std::forward<ARGS>(args)...
  );
}

};

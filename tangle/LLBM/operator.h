#pragma once

#include <tuple>

template <typename OPERATOR, typename... ARGS>
struct Operator {
  bool* const mask;
  const std::tuple<ARGS...> config;

  Operator(OPERATOR, DeviceBuffer<bool>& m, ARGS... args):
    mask(m.device()),
    config(args...) { }

  template <typename DESCRIPTOR, typename T, typename S>
  __device__ bool apply(DESCRIPTOR d, S f_curr[DESCRIPTOR::q], S f_next[DESCRIPTOR::q], std::size_t gid) const {
    if (mask[gid]) {
      std::apply([](auto... args) { OPERATOR::template apply<T,S>(args...); },
                 std::tuple_cat(std::make_tuple(d, f_curr, f_next, gid), config));
      return true;
    } else {
      return false;
    }
  }
};

template <typename OPERATOR, typename... ARGS>
Operator(OPERATOR, DeviceBuffer<bool>&, ARGS... args) -> Operator<OPERATOR,std::remove_reference_t<ARGS>...>;

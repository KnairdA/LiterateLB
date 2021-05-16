#pragma once
#include <LLBM/call_tag.h>

struct InitializeO {

using call_tag = tag::call_by_cell_id;

template <typename T, typename S>
__device__ static void apply(descriptor::D2Q9, S f_curr[9], S f_next[9], std::size_t gid) {
  f_next[0] = T{0.0277777777777778};
  f_next[1] = T{0.111111111111111};
  f_next[2] = T{0.0277777777777778};
  f_next[3] = T{0.111111111111111};
  f_next[4] = T{0.444444444444444};
  f_next[5] = T{0.111111111111111};
  f_next[6] = T{0.0277777777777778};
  f_next[7] = T{0.111111111111111};
  f_next[8] = T{0.0277777777777778};
}

template <typename T, typename S>
__device__ static void apply(descriptor::D3Q19, S f_curr[19], S f_next[19], std::size_t gid) {
  f_next[0] = T{0.0277777777777778};
  f_next[1] = T{0.0277777777777778};
  f_next[2] = T{0.0555555555555556};
  f_next[3] = T{0.0277777777777778};
  f_next[4] = T{0.0277777777777778};
  f_next[5] = T{0.0277777777777778};
  f_next[6] = T{0.0555555555555556};
  f_next[7] = T{0.0277777777777778};
  f_next[8] = T{0.0555555555555556};
  f_next[9] = T{0.333333333333333};
  f_next[10] = T{0.0555555555555556};
  f_next[11] = T{0.0277777777777778};
  f_next[12] = T{0.0555555555555556};
  f_next[13] = T{0.0277777777777778};
  f_next[14] = T{0.0277777777777778};
  f_next[15] = T{0.0277777777777778};
  f_next[16] = T{0.0555555555555556};
  f_next[17] = T{0.0277777777777778};
  f_next[18] = T{0.0277777777777778};
}

};

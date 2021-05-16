#pragma once
#include <LLBM/call_tag.h>

struct BounceBackMovingWallO {

using call_tag = tag::call_by_cell_id;

template <typename T, typename S>
__device__ static void apply(descriptor::D2Q9, S f_curr[9], S f_next[9], std::size_t gid, T u_0, T u_1) {
  f_next[0] = -T{0.166666666666667}*u_0 - T{0.166666666666667}*u_1 + f_curr[8];
  f_next[1] = -T{0.666666666666667}*u_0 + f_curr[7];
  f_next[2] = -T{0.166666666666667}*u_0 + T{0.166666666666667}*u_1 + f_curr[6];
  f_next[3] = -T{0.666666666666667}*u_1 + f_curr[5];
  f_next[4] = f_curr[4];
  f_next[5] = T{0.666666666666667}*u_1 + f_curr[3];
  f_next[6] = T{0.166666666666667}*u_0 - T{0.166666666666667}*u_1 + f_curr[2];
  f_next[7] = T{0.666666666666667}*u_0 + f_curr[1];
  f_next[8] = T{0.166666666666667}*u_0 + T{0.166666666666667}*u_1 + f_curr[0];
}

template <typename T, typename S>
__device__ static void apply(descriptor::D3Q19, S f_curr[19], S f_next[19], std::size_t gid, T u_0, T u_1, T u_2) {
  f_next[0] = T{0.166666666666667}*u_1 + T{0.166666666666667}*u_2 + f_curr[18];
  f_next[1] = -T{0.166666666666667}*u_0 + T{0.166666666666667}*u_2 + f_curr[17];
  f_next[2] = T{0.333333333333333}*u_2 + f_curr[16];
  f_next[3] = T{0.166666666666667}*u_0 + T{0.166666666666667}*u_2 + f_curr[15];
  f_next[4] = -T{0.166666666666667}*u_1 + T{0.166666666666667}*u_2 + f_curr[14];
  f_next[5] = -T{0.166666666666667}*u_0 + T{0.166666666666667}*u_1 + f_curr[13];
  f_next[6] = T{0.333333333333333}*u_1 + f_curr[12];
  f_next[7] = T{0.166666666666667}*u_0 + T{0.166666666666667}*u_1 + f_curr[11];
  f_next[8] = -T{0.333333333333333}*u_0 + f_curr[10];
  f_next[9] = f_curr[9];
  f_next[10] = T{0.333333333333333}*u_0 + f_curr[8];
  f_next[11] = -T{0.166666666666667}*u_0 - T{0.166666666666667}*u_1 + f_curr[7];
  f_next[12] = -T{0.333333333333333}*u_1 + f_curr[6];
  f_next[13] = T{0.166666666666667}*u_0 - T{0.166666666666667}*u_1 + f_curr[5];
  f_next[14] = T{0.166666666666667}*u_1 - T{0.166666666666667}*u_2 + f_curr[4];
  f_next[15] = -T{0.166666666666667}*u_0 - T{0.166666666666667}*u_2 + f_curr[3];
  f_next[16] = -T{0.333333333333333}*u_2 + f_curr[2];
  f_next[17] = T{0.166666666666667}*u_0 - T{0.166666666666667}*u_2 + f_curr[1];
  f_next[18] = -T{0.166666666666667}*u_1 - T{0.166666666666667}*u_2 + f_curr[0];
}

};

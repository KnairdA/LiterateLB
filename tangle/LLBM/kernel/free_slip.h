#pragma once
#include <LLBM/call_tag.h>
#include <LLBM/wall.h>
#include <LLBM/descriptor.h>

struct BounceBackFreeSlipO {

using call_tag = tag::call_by_cell_id;

template <typename T, typename S>
__device__ static void apply(descriptor::D2Q9, S f_curr[9], S f_next[9], std::size_t gid, WallNormal<1,0>) {
  f_next[0] = f_curr[6];
  f_next[1] = f_curr[7];
  f_next[2] = f_curr[8];
  f_next[3] = f_curr[3];
  f_next[4] = f_curr[4];
  f_next[5] = f_curr[5];
  f_next[6] = f_curr[0];
  f_next[7] = f_curr[1];
  f_next[8] = f_curr[2];
}

template <typename T, typename S>
__device__ static void apply(descriptor::D2Q9, S f_curr[9], S f_next[9], std::size_t gid, WallNormal<0,1>) {
  f_next[0] = f_curr[2];
  f_next[1] = f_curr[1];
  f_next[2] = f_curr[0];
  f_next[3] = f_curr[5];
  f_next[4] = f_curr[4];
  f_next[5] = f_curr[3];
  f_next[6] = f_curr[8];
  f_next[7] = f_curr[7];
  f_next[8] = f_curr[6];
}

template <typename T, typename S>
__device__ static void apply(descriptor::D3Q19, S f_curr[19], S f_next[19], std::size_t gid, WallNormal<0,1,0>) {
  f_next[0] = f_curr[4];
  f_next[1] = f_curr[1];
  f_next[2] = f_curr[2];
  f_next[3] = f_curr[3];
  f_next[4] = f_curr[0];
  f_next[5] = f_curr[11];
  f_next[6] = f_curr[12];
  f_next[7] = f_curr[13];
  f_next[8] = f_curr[8];
  f_next[9] = f_curr[9];
  f_next[10] = f_curr[10];
  f_next[11] = f_curr[5];
  f_next[12] = f_curr[6];
  f_next[13] = f_curr[7];
  f_next[14] = f_curr[18];
  f_next[15] = f_curr[15];
  f_next[16] = f_curr[16];
  f_next[17] = f_curr[17];
  f_next[18] = f_curr[14];
}

template <typename T, typename S>
__device__ static void apply(descriptor::D3Q19, S f_curr[19], S f_next[19], std::size_t gid, WallNormal<0,0,1>) {
  f_next[0] = f_curr[14];
  f_next[1] = f_curr[15];
  f_next[2] = f_curr[16];
  f_next[3] = f_curr[17];
  f_next[4] = f_curr[18];
  f_next[5] = f_curr[5];
  f_next[6] = f_curr[6];
  f_next[7] = f_curr[7];
  f_next[8] = f_curr[8];
  f_next[9] = f_curr[9];
  f_next[10] = f_curr[10];
  f_next[11] = f_curr[11];
  f_next[12] = f_curr[12];
  f_next[13] = f_curr[13];
  f_next[14] = f_curr[0];
  f_next[15] = f_curr[1];
  f_next[16] = f_curr[2];
  f_next[17] = f_curr[3];
  f_next[18] = f_curr[4];
}

};

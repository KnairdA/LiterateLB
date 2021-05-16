#pragma once
#include <LLBM/call_tag.h>

struct CollectMomentsF {

using call_tag = tag::call_by_cell_id;

template <typename T, typename S>
__device__ static void apply(descriptor::D2Q9, S f_curr[9], std::size_t gid, T* cell_rho, T* cell_u) {
  T m0 = f_curr[1] + f_curr[2];
  T m1 = f_curr[3] + f_curr[6];
  T m2 = m0 + m1 + f_curr[0] + f_curr[4] + f_curr[5] + f_curr[7] + f_curr[8];
  T m3 = f_curr[0] - f_curr[8];
  T m4 = T{1} / (m2);
  T rho = m2;
  T u_0 = -m4*(m0 + m3 - f_curr[6] - f_curr[7]);
  T u_1 = -m4*(m1 + m3 - f_curr[2] - f_curr[5]);

  cell_rho[gid] = rho;
  cell_u[2*gid+0] = u_0;
  cell_u[2*gid+1] = u_1;
}

template <typename T, typename S>
__device__ static void apply(descriptor::D3Q19, S f_curr[19], std::size_t gid, T* cell_rho, T* cell_u) {
  T m0 = f_curr[10] + f_curr[13] + f_curr[17];
  T m1 = f_curr[14] + f_curr[5] + f_curr[6];
  T m2 = f_curr[1] + f_curr[2] + f_curr[4];
  T m3 = m0 + m1 + m2 + f_curr[0] + f_curr[11] + f_curr[12] + f_curr[15] + f_curr[16] + f_curr[18] + f_curr[3] + f_curr[7] + f_curr[8] + f_curr[9];
  T m4 = -f_curr[11] + f_curr[7];
  T m5 = -f_curr[15] + f_curr[3];
  T m6 = T{1} / (m3);
  T m7 = f_curr[0] - f_curr[18];
  T rho = m3;
  T u_0 = m6*(m0 + m4 + m5 - f_curr[1] - f_curr[5] - f_curr[8]);
  T u_1 = m6*(m1 + m4 + m7 - f_curr[12] - f_curr[13] - f_curr[4]);
  T u_2 = m6*(m2 + m5 + m7 - f_curr[14] - f_curr[16] - f_curr[17]);

  cell_rho[gid] = rho;
  cell_u[3*gid+0] = u_0;
  cell_u[3*gid+1] = u_1;
  cell_u[3*gid+2] = u_2;
}

};

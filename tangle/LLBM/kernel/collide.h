#pragma once
#include <LLBM/call_tag.h>

struct BgkCollideO {

using call_tag = tag::call_by_cell_id;

template <typename T, typename S>
__device__ static void apply(descriptor::D2Q9, S f_curr[9], S f_next[9], std::size_t gid, T tau) {
  T m0 = f_curr[1] + f_curr[2];
  T m1 = f_curr[3] + f_curr[6];
  T m2 = m0 + m1 + f_curr[0] + f_curr[4] + f_curr[5] + f_curr[7] + f_curr[8];
  T m3 = f_curr[0] - f_curr[8];
  T m4 = T{1} / (m2);
  T rho = m2;
  T u_0 = -m4*(m0 + m3 - f_curr[6] - f_curr[7]);
  T u_1 = -m4*(m1 + m3 - f_curr[2] - f_curr[5]);
  T x0 = T{1} / (tau);
  T x1 = T{0.0138888888888889}*x0;
  T x2 = T{6.00000000000000}*u_1;
  T x3 = T{6.00000000000000}*u_0;
  T x4 = u_0 + u_1;
  T x5 = T{9.00000000000000}*(x4*x4);
  T x6 = u_1*u_1;
  T x7 = T{3.00000000000000}*x6;
  T x8 = u_0*u_0;
  T x9 = T{3.00000000000000}*x8;
  T x10 = x9 + T{-2.00000000000000};
  T x11 = x10 + x7;
  T x12 = T{0.0555555555555556}*x0;
  T x13 = -x3;
  T x14 = T{2.00000000000000} - x7;
  T x15 = x14 + T{6.00000000000000}*x8;
  T x16 = -x9;
  T x17 = x16 + x2;
  T x18 = u_0 - u_1;
  T x19 = x14 + T{9.00000000000000}*(x18*x18);
  T x20 = T{6.00000000000000}*x6;
  f_next[0] = -x1*(rho*(x11 + x2 + x3 - x5) + T{72.0000000000000}*f_curr[0]) + f_curr[0];
  f_next[1] = x12*(rho*(x13 + x15) - T{18.0000000000000}*f_curr[1]) + f_curr[1];
  f_next[2] = x1*(rho*(x13 + x17 + x19) - T{72.0000000000000}*f_curr[2]) + f_curr[2];
  f_next[3] = -x12*(rho*(x10 + x2 - x20) + T{18.0000000000000}*f_curr[3]) + f_curr[3];
  f_next[4] = -T{0.111111111111111}*x0*(T{2.00000000000000}*rho*x11 + T{9.00000000000000}*f_curr[4]) + f_curr[4];
  f_next[5] = x12*(rho*(x17 + x20 + T{2.00000000000000}) - T{18.0000000000000}*f_curr[5]) + f_curr[5];
  f_next[6] = x1*(rho*(x16 + x19 - x2 + x3) - T{72.0000000000000}*f_curr[6]) + f_curr[6];
  f_next[7] = x12*(rho*(x15 + x3) - T{18.0000000000000}*f_curr[7]) + f_curr[7];
  f_next[8] = x1*(rho*(x14 + x17 + x3 + x5) - T{72.0000000000000}*f_curr[8]) + f_curr[8];
  
}

template <typename T, typename S>
__device__ static void apply(descriptor::D3Q19, S f_curr[19], S f_next[19], std::size_t gid, T tau) {
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
  T x0 = T{1} / (tau);
  T x1 = T{0.0138888888888889}*x0;
  T x2 = u_1 + u_2;
  T x3 = T{9.00000000000000}*(x2*x2);
  T x4 = T{6.00000000000000}*u_2;
  T x5 = u_1*u_1;
  T x6 = T{3.00000000000000}*x5;
  T x7 = -x6;
  T x8 = u_0*u_0;
  T x9 = T{3.00000000000000}*x8;
  T x10 = T{2.00000000000000} - x9;
  T x11 = x10 + x7;
  T x12 = x11 + x4;
  T x13 = u_2*u_2;
  T x14 = T{3.00000000000000}*x13;
  T x15 = -x14;
  T x16 = T{6.00000000000000}*u_1;
  T x17 = x15 + x16;
  T x18 = T{6.00000000000000}*u_0;
  T x19 = -u_0;
  T x20 = x19 + u_2;
  T x21 = x14 + x6 + T{-2.00000000000000};
  T x22 = x21 + x9;
  T x23 = x22 - x4;
  T x24 = T{0.0277777777777778}*x0;
  T x25 = T{6.00000000000000}*x13;
  T x26 = u_0 + u_2;
  T x27 = T{9.00000000000000}*(x26*x26);
  T x28 = x15 + x18;
  T x29 = -u_1;
  T x30 = x29 + u_2;
  T x31 = x19 + u_1;
  T x32 = -x16;
  T x33 = x18 + x22;
  T x34 = T{6.00000000000000}*x5;
  T x35 = u_0 + u_1;
  T x36 = T{9.00000000000000}*(x35*x35);
  T x37 = T{6.00000000000000}*x8;
  T x38 = x9 + T{-2.00000000000000};
  T x39 = x29 + u_0;
  T x40 = -x18 + x22;
  T x41 = -u_2;
  T x42 = x41 + u_1;
  T x43 = x22 + x4;
  T x44 = x41 + u_0;
  f_next[0] = x1*(rho*(x12 + x17 + x3) - T{72.0000000000000}*f_curr[0]) + f_curr[0];
  f_next[1] = -x1*(rho*(x18 + x23 - T{9.00000000000000}*x20*x20) + T{72.0000000000000}*f_curr[1]) + f_curr[1];
  f_next[2] = x24*(rho*(x12 + x25) - T{36.0000000000000}*f_curr[2]) + f_curr[2];
  f_next[3] = x1*(rho*(x12 + x27 + x28) - T{72.0000000000000}*f_curr[3]) + f_curr[3];
  f_next[4] = -x1*(rho*(x16 + x23 - T{9.00000000000000}*x30*x30) + T{72.0000000000000}*f_curr[4]) + f_curr[4];
  f_next[5] = -x1*(rho*(x32 + x33 - T{9.00000000000000}*x31*x31) + T{72.0000000000000}*f_curr[5]) + f_curr[5];
  f_next[6] = x24*(rho*(x10 + x17 + x34) - T{36.0000000000000}*f_curr[6]) + f_curr[6];
  f_next[7] = x1*(rho*(x11 + x17 + x18 + x36) - T{72.0000000000000}*f_curr[7]) + f_curr[7];
  f_next[8] = -x24*(rho*(x18 + x21 - x37) + T{36.0000000000000}*f_curr[8]) + f_curr[8];
  f_next[9] = -T{0.166666666666667}*x0*(rho*x22 + T{6.00000000000000}*f_curr[9]) + f_curr[9];
  f_next[10] = x24*(rho*(x28 + x37 + x7 + T{2.00000000000000}) - T{36.0000000000000}*f_curr[10]) + f_curr[10];
  f_next[11] = -x1*(rho*(x16 + x33 - x36) + T{72.0000000000000}*f_curr[11]) + f_curr[11];
  f_next[12] = -x24*(rho*(x14 + x16 - x34 + x38) + T{36.0000000000000}*f_curr[12]) + f_curr[12];
  f_next[13] = -x1*(rho*(x16 + x40 - T{9.00000000000000}*x39*x39) + T{72.0000000000000}*f_curr[13]) + f_curr[13];
  f_next[14] = -x1*(rho*(x32 + x43 - T{9.00000000000000}*x42*x42) + T{72.0000000000000}*f_curr[14]) + f_curr[14];
  f_next[15] = -x1*(rho*(-x27 + x33 + x4) + T{72.0000000000000}*f_curr[15]) + f_curr[15];
  f_next[16] = -x24*(rho*(-x25 + x38 + x4 + x6) + T{36.0000000000000}*f_curr[16]) + f_curr[16];
  f_next[17] = -x1*(rho*(x4 + x40 - T{9.00000000000000}*x44*x44) + T{72.0000000000000}*f_curr[17]) + f_curr[17];
  f_next[18] = -x1*(rho*(x16 - x3 + x43) + T{72.0000000000000}*f_curr[18]) + f_curr[18];
}

};

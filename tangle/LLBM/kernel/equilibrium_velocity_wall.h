#pragma once
#include <LLBM/call_tag.h>
#include <LLBM/wall.h>
#include <LLBM/descriptor.h>

struct EquilibriumVelocityWallO {

using call_tag = tag::call_by_cell_id;

template <typename T, typename S>
__device__ static void apply(descriptor::D2Q9, S f_curr[9], S f_next[9], std::size_t gid, T u_w, WallNormal<1,0>) {
  T rho = -(T{2.00000000000000}*f_curr[0] + T{2.00000000000000}*f_curr[1] + T{2.00000000000000}*f_curr[2] + f_curr[3] + f_curr[4] + f_curr[5])/(u_w + T{-1.00000000000000});
  T u_0 = u_w;
  T u_1 = 0.;
  T e0 = T{0.0277777777777778}*rho;
  T e1 = T{3.00000000000000}*u_1;
  T e2 = T{3.00000000000000}*u_0;
  T e3 = u_0 + u_1;
  T e4 = T{4.50000000000000}*(e3*e3);
  T e5 = u_1*u_1;
  T e6 = T{1.50000000000000}*e5;
  T e7 = u_0*u_0;
  T e8 = T{1.50000000000000}*e7;
  T e9 = e8 + T{-1.00000000000000};
  T e10 = e6 + e9;
  T e11 = T{0.111111111111111}*rho;
  T e12 = -e2;
  T e13 = T{1.00000000000000} - e6;
  T e14 = e13 + T{3.00000000000000}*e7;
  T e15 = -e8;
  T e16 = e1 + e15;
  T e17 = u_0 - u_1;
  T e18 = e13 + T{4.50000000000000}*(e17*e17);
  T e19 = T{3.00000000000000}*e5;
  f_next[0] = -e0*(e1 + e10 + e2 - e4);
  f_next[1] = e11*(e12 + e14);
  f_next[2] = e0*(e12 + e16 + e18);
  f_next[3] = -e11*(e1 - e19 + e9);
  f_next[4] = -T{0.444444444444444}*e10*rho;
  f_next[5] = e11*(e16 + e19 + T{1.00000000000000});
  f_next[6] = e0*(-e1 + e15 + e18 + e2);
  f_next[7] = e11*(e14 + e2);
  f_next[8] = e0*(e13 + e16 + e2 + e4);
  
}

template <typename T, typename S>
__device__ static void apply(descriptor::D2Q9, S f_curr[9], S f_next[9], std::size_t gid, T u_w, WallNormal<-1,0>) {
  T rho = (f_curr[3] + f_curr[4] + f_curr[5] + T{2.00000000000000}*f_curr[6] + T{2.00000000000000}*f_curr[7] + T{2.00000000000000}*f_curr[8])/(u_w + T{1.00000000000000});
  T u_0 = u_w;
  T u_1 = 0;
  T e0 = T{0.0277777777777778}*rho;
  T e1 = T{3.00000000000000}*u_1;
  T e2 = T{3.00000000000000}*u_0;
  T e3 = u_0 + u_1;
  T e4 = T{4.50000000000000}*(e3*e3);
  T e5 = u_1*u_1;
  T e6 = T{1.50000000000000}*e5;
  T e7 = u_0*u_0;
  T e8 = T{1.50000000000000}*e7;
  T e9 = e8 + T{-1.00000000000000};
  T e10 = e6 + e9;
  T e11 = T{0.111111111111111}*rho;
  T e12 = -e2;
  T e13 = T{1.00000000000000} - e6;
  T e14 = e13 + T{3.00000000000000}*e7;
  T e15 = -e8;
  T e16 = e1 + e15;
  T e17 = u_0 - u_1;
  T e18 = e13 + T{4.50000000000000}*(e17*e17);
  T e19 = T{3.00000000000000}*e5;
  f_next[0] = -e0*(e1 + e10 + e2 - e4);
  f_next[1] = e11*(e12 + e14);
  f_next[2] = e0*(e12 + e16 + e18);
  f_next[3] = -e11*(e1 - e19 + e9);
  f_next[4] = -T{0.444444444444444}*e10*rho;
  f_next[5] = e11*(e16 + e19 + T{1.00000000000000});
  f_next[6] = e0*(-e1 + e15 + e18 + e2);
  f_next[7] = e11*(e14 + e2);
  f_next[8] = e0*(e13 + e16 + e2 + e4);
  
}

template <typename T, typename S>
__device__ static void apply(descriptor::D3Q19, S f_curr[19], S f_next[19], std::size_t gid, T u_w, WallNormal<1,0,0>) {
  T rho = -(f_curr[0] + T{2.00000000000000}*f_curr[11] + f_curr[12] + f_curr[14] + T{2.00000000000000}*f_curr[15] + f_curr[16] + f_curr[18] + T{2.00000000000000}*f_curr[1] + f_curr[2] + f_curr[4] + T{2.00000000000000}*f_curr[5] + f_curr[6] + T{2.00000000000000}*f_curr[8] + f_curr[9])/(u_w + T{-1.00000000000000});
  T u_0 = u_w;
  T u_1 = 0;
  T u_2 = 0;
  T e0 = T{0.0277777777777778}*rho;
  T e1 = u_1 + u_2;
  T e2 = T{4.50000000000000}*(e1*e1);
  T e3 = T{3.00000000000000}*u_2;
  T e4 = u_1*u_1;
  T e5 = T{1.50000000000000}*e4;
  T e6 = -e5;
  T e7 = u_0*u_0;
  T e8 = T{1.50000000000000}*e7;
  T e9 = T{1.00000000000000} - e8;
  T e10 = e6 + e9;
  T e11 = e10 + e3;
  T e12 = T{3.00000000000000}*u_1;
  T e13 = u_2*u_2;
  T e14 = T{1.50000000000000}*e13;
  T e15 = -e14;
  T e16 = e12 + e15;
  T e17 = T{3.00000000000000}*u_0;
  T e18 = -u_2;
  T e19 = e18 + u_0;
  T e20 = -T{4.50000000000000}*e19*e19;
  T e21 = e14 + e5 + T{-1.00000000000000};
  T e22 = e21 + e8;
  T e23 = e22 - e3;
  T e24 = T{0.0555555555555556}*rho;
  T e25 = T{3.00000000000000}*e13;
  T e26 = u_0 + u_2;
  T e27 = T{4.50000000000000}*(e26*e26);
  T e28 = e15 + e17;
  T e29 = e18 + u_1;
  T e30 = -T{4.50000000000000}*e29*e29;
  T e31 = -e12;
  T e32 = u_0 - u_1;
  T e33 = -T{4.50000000000000}*e32*e32;
  T e34 = e17 + e22;
  T e35 = T{3.00000000000000}*e4;
  T e36 = u_0 + u_1;
  T e37 = T{4.50000000000000}*(e36*e36);
  T e38 = T{3.00000000000000}*e7;
  T e39 = e8 + T{-1.00000000000000};
  T e40 = -e17 + e22;
  T e41 = e22 + e3;
  f_next[0] = e0*(e11 + e16 + e2);
  f_next[1] = -e0*(e17 + e20 + e23);
  f_next[2] = e24*(e11 + e25);
  f_next[3] = e0*(e11 + e27 + e28);
  f_next[4] = -e0*(e12 + e23 + e30);
  f_next[5] = -e0*(e31 + e33 + e34);
  f_next[6] = e24*(e16 + e35 + e9);
  f_next[7] = e0*(e10 + e16 + e17 + e37);
  f_next[8] = -e24*(e17 + e21 - e38);
  f_next[9] = -T{0.333333333333333}*e22*rho;
  f_next[10] = e24*(e28 + e38 + e6 + T{1.00000000000000});
  f_next[11] = -e0*(e12 + e34 - e37);
  f_next[12] = -e24*(e12 + e14 - e35 + e39);
  f_next[13] = -e0*(e12 + e33 + e40);
  f_next[14] = -e0*(e30 + e31 + e41);
  f_next[15] = -e0*(-e27 + e3 + e34);
  f_next[16] = -e24*(-e25 + e3 + e39 + e5);
  f_next[17] = -e0*(e20 + e3 + e40);
  f_next[18] = -e0*(e12 - e2 + e41);
}

template <typename T, typename S>
__device__ static void apply(descriptor::D3Q19, S f_curr[19], S f_next[19], std::size_t gid, T u_w, WallNormal<-1,0,0>) {
  T rho = (f_curr[0] + T{2.00000000000000}*f_curr[10] + f_curr[12] + T{2.00000000000000}*f_curr[13] + f_curr[14] + f_curr[16] + T{2.00000000000000}*f_curr[17] + f_curr[18] + f_curr[2] + T{2.00000000000000}*f_curr[3] + f_curr[4] + f_curr[6] + T{2.00000000000000}*f_curr[7] + f_curr[9])/(u_w + T{1.00000000000000});
  T u_0 = u_w;
  T u_1 = 0;
  T u_2 = 0;
  T e0 = T{0.0277777777777778}*rho;
  T e1 = u_1 + u_2;
  T e2 = T{4.50000000000000}*(e1*e1);
  T e3 = T{3.00000000000000}*u_2;
  T e4 = u_1*u_1;
  T e5 = T{1.50000000000000}*e4;
  T e6 = -e5;
  T e7 = u_0*u_0;
  T e8 = T{1.50000000000000}*e7;
  T e9 = T{1.00000000000000} - e8;
  T e10 = e6 + e9;
  T e11 = e10 + e3;
  T e12 = T{3.00000000000000}*u_1;
  T e13 = u_2*u_2;
  T e14 = T{1.50000000000000}*e13;
  T e15 = -e14;
  T e16 = e12 + e15;
  T e17 = T{3.00000000000000}*u_0;
  T e18 = -u_2;
  T e19 = e18 + u_0;
  T e20 = -T{4.50000000000000}*e19*e19;
  T e21 = e14 + e5 + T{-1.00000000000000};
  T e22 = e21 + e8;
  T e23 = e22 - e3;
  T e24 = T{0.0555555555555556}*rho;
  T e25 = T{3.00000000000000}*e13;
  T e26 = u_0 + u_2;
  T e27 = T{4.50000000000000}*(e26*e26);
  T e28 = e15 + e17;
  T e29 = e18 + u_1;
  T e30 = -T{4.50000000000000}*e29*e29;
  T e31 = -e12;
  T e32 = u_0 - u_1;
  T e33 = -T{4.50000000000000}*e32*e32;
  T e34 = e17 + e22;
  T e35 = T{3.00000000000000}*e4;
  T e36 = u_0 + u_1;
  T e37 = T{4.50000000000000}*(e36*e36);
  T e38 = T{3.00000000000000}*e7;
  T e39 = e8 + T{-1.00000000000000};
  T e40 = -e17 + e22;
  T e41 = e22 + e3;
  f_next[0] = e0*(e11 + e16 + e2);
  f_next[1] = -e0*(e17 + e20 + e23);
  f_next[2] = e24*(e11 + e25);
  f_next[3] = e0*(e11 + e27 + e28);
  f_next[4] = -e0*(e12 + e23 + e30);
  f_next[5] = -e0*(e31 + e33 + e34);
  f_next[6] = e24*(e16 + e35 + e9);
  f_next[7] = e0*(e10 + e16 + e17 + e37);
  f_next[8] = -e24*(e17 + e21 - e38);
  f_next[9] = -T{0.333333333333333}*e22*rho;
  f_next[10] = e24*(e28 + e38 + e6 + T{1.00000000000000});
  f_next[11] = -e0*(e12 + e34 - e37);
  f_next[12] = -e24*(e12 + e14 - e35 + e39);
  f_next[13] = -e0*(e12 + e33 + e40);
  f_next[14] = -e0*(e30 + e31 + e41);
  f_next[15] = -e0*(-e27 + e3 + e34);
  f_next[16] = -e24*(-e25 + e3 + e39 + e5);
  f_next[17] = -e0*(e20 + e3 + e40);
  f_next[18] = -e0*(e12 - e2 + e41);
}

};

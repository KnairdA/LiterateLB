#pragma once
#include <LLBM/call_tag.h>

struct CollectQCriterionF {

using call_tag = tag::call_by_spatial_cell_mask;

template <typename T, typename S>
__device__ static void apply(
    descriptor::D3Q19
  , S f_curr[19]
  , descriptor::CuboidD<3> cuboid
  , std::size_t gid
  , std::size_t iX
  , std::size_t iY
  , std::size_t iZ
  , T* cell_rho 
  , T* cell_u
  , T* cell_curl_norm
  , cudaSurfaceObject_t surface
  , T* cell_q = nullptr
) {
  const T rho = cell_rho[gid];
  const T u_0 = cell_u[3*gid + 0];
  const T u_1 = cell_u[3*gid + 1];
  const T u_2 = cell_u[3*gid + 2];

  T x0 = T{72.0000000000000}*f_curr[5];
  T x1 = T{72.0000000000000}*f_curr[13];
  T x2 = T{6.00000000000000}*u_1;
  T x3 = -x2;
  T x4 = -u_0;
  T x5 = x4 + u_1;
  T x6 = u_1*u_1;
  T x7 = T{3.00000000000000}*x6;
  T x8 = u_0*u_0;
  T x9 = T{3.00000000000000}*x8;
  T x10 = x9 + T{-2.00000000000000};
  T x11 = x10 + x7;
  T x12 = u_2*u_2;
  T x13 = T{3.00000000000000}*x12;
  T x14 = T{6.00000000000000}*u_0;
  T x15 = x13 + x14;
  T x16 = rho*(x11 + x15 + x3 - T{9.00000000000000}*x5*x5);
  T x17 = -x14;
  T x18 = -u_1;
  T x19 = x18 + u_0;
  T x20 = x13 + x2;
  T x21 = x11 + x20;
  T x22 = rho*(x17 + x21 - T{9.00000000000000}*x19*x19);
  T x23 = u_0 + u_1;
  T x24 = T{9.00000000000000}*(x23*x23);
  T x25 = -x7;
  T x26 = T{2.00000000000000} - x9;
  T x27 = x25 + x26;
  T x28 = -x13;
  T x29 = x2 + x28;
  T x30 = -rho*(x14 + x21 - x24) + rho*(x14 + x24 + x27 + x29) - T{72.0000000000000}*f_curr[11] - T{72.0000000000000}*f_curr[7];
  T x31 = T{72.0000000000000}*f_curr[1];
  T x32 = T{72.0000000000000}*f_curr[17];
  T x33 = x4 + u_2;
  T x34 = T{6.00000000000000}*u_2;
  T x35 = x11 - x34;
  T x36 = rho*(x15 + x35 - T{9.00000000000000}*x33*x33);
  T x37 = -u_2;
  T x38 = x37 + u_0;
  T x39 = x11 + x34;
  T x40 = x13 + x39;
  T x41 = rho*(x17 + x40 - T{9.00000000000000}*x38*x38);
  T x42 = u_0 + u_2;
  T x43 = T{9.00000000000000}*(x42*x42);
  T x44 = x27 + x34;
  T x45 = x14 + x28;
  T x46 = -rho*(x15 + x39 - x43) + rho*(x43 + x44 + x45) - T{72.0000000000000}*f_curr[15] - T{72.0000000000000}*f_curr[3];
  T x47 = T{72.0000000000000}*f_curr[4];
  T x48 = T{72.0000000000000}*f_curr[14];
  T x49 = x18 + u_2;
  T x50 = rho*(x20 + x35 - T{9.00000000000000}*x49*x49);
  T x51 = x37 + u_1;
  T x52 = rho*(x3 + x40 - T{9.00000000000000}*x51*x51);
  T x53 = u_1 + u_2;
  T x54 = T{9.00000000000000}*(x53*x53);
  T x55 = -rho*(x20 + x39 - x54) + rho*(x29 + x44 + x54) - T{72.0000000000000}*f_curr[0] - T{72.0000000000000}*f_curr[18];
  T x56 = T{2.00000000000000}*rho;
  T x57 = T{6.00000000000000}*x8;
  T x58 = -x31 - x32 - x36 - x41 + x46;
  T x59 = -x0 - x1 - x16 - x22 + x30;
  T x60 = T{6.00000000000000}*x6;
  T x61 = -x47 - x48 - x50 - x52 + x55;
  T x62 = T{6.00000000000000}*x12;
  T strain = T{0.0277777777777778}*sqrt((x0 + x1 + x16 + x22 + x30)*(x0 + x1 + x16 + x22 + x30) + (x31 + x32 + x36 + x41 + x46)*(x31 + x32 + x36 + x41 + x46) + (x47 + x48 + x50 + x52 + x55)*(x47 + x48 + x50 + x52 + x55) + T{0.500000000000000}*((-x56*(x39 - x62) + x56*(x44 + x62) + x58 + x61 - 72*f_curr[16] - 72*f_curr[2])*(-x56*(x39 - x62) + x56*(x44 + x62) + x58 + x61 - 72*f_curr[16] - 72*f_curr[2])) + T{0.500000000000000}*((-x56*(x10 + x20 - x60) + x56*(x26 + x29 + x60) + x59 + x61 - 72*f_curr[12] - 72*f_curr[6])*(-x56*(x10 + x20 - x60) + x56*(x26 + x29 + x60) + x59 + x61 - 72*f_curr[12] - 72*f_curr[6])) + T{0.500000000000000}*((-x56*(x15 - x57 + x7 - 2) + x56*(x25 + x45 + x57 + 2) + x58 + x59 - 72*f_curr[10] - 72*f_curr[8])*(-x56*(x15 - x57 + x7 - 2) + x56*(x25 + x45 + x57 + 2) + x58 + x59 - 72*f_curr[10] - 72*f_curr[8])));

  float vorticity = cell_curl_norm[gid];
  float q = vorticity*vorticity - strain*strain;
  q = q > 0 ? q : 0;

  surf3Dwrite(q, surface, iX*sizeof(float), iY, iZ);

  if (cell_q != nullptr) {
    cell_q[gid] = q;
  }
}

};

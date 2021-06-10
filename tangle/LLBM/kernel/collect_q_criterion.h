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
  T x31 = x0 + x1 + x16 + x22 + x30;
  T x32 = T{72.0000000000000}*f_curr[1];
  T x33 = T{72.0000000000000}*f_curr[17];
  T x34 = x4 + u_2;
  T x35 = T{6.00000000000000}*u_2;
  T x36 = x11 - x35;
  T x37 = rho*(x15 + x36 - T{9.00000000000000}*x34*x34);
  T x38 = -u_2;
  T x39 = x38 + u_0;
  T x40 = x11 + x35;
  T x41 = x13 + x40;
  T x42 = rho*(x17 + x41 - T{9.00000000000000}*x39*x39);
  T x43 = u_0 + u_2;
  T x44 = T{9.00000000000000}*(x43*x43);
  T x45 = x27 + x35;
  T x46 = x14 + x28;
  T x47 = -rho*(x15 + x40 - x44) + rho*(x44 + x45 + x46) - T{72.0000000000000}*f_curr[15] - T{72.0000000000000}*f_curr[3];
  T x48 = x32 + x33 + x37 + x42 + x47;
  T x49 = T{72.0000000000000}*f_curr[4];
  T x50 = T{72.0000000000000}*f_curr[14];
  T x51 = x18 + u_2;
  T x52 = rho*(x20 + x36 - T{9.00000000000000}*x51*x51);
  T x53 = x38 + u_1;
  T x54 = rho*(x3 + x41 - T{9.00000000000000}*x53*x53);
  T x55 = u_1 + u_2;
  T x56 = T{9.00000000000000}*(x55*x55);
  T x57 = -rho*(x20 + x40 - x56) + rho*(x29 + x45 + x56) - T{72.0000000000000}*f_curr[0] - T{72.0000000000000}*f_curr[18];
  T x58 = x49 + x50 + x52 + x54 + x57;
  T x59 = T{2.00000000000000}*rho;
  T x60 = T{6.00000000000000}*x8;
  T x61 = -x32 - x33 - x37 - x42 + x47;
  T x62 = -x0 - x1 - x16 - x22 + x30;
  T x63 = -x59*(x15 - x60 + x7 + T{-2.00000000000000}) + x59*(x25 + x46 + x60 + T{2.00000000000000}) + x61 + x62 - T{72.0000000000000}*f_curr[10] - T{72.0000000000000}*f_curr[8];
  T x64 = T{6.00000000000000}*x6;
  T x65 = -x49 - x50 - x52 - x54 + x57;
  T x66 = -x59*(x10 + x20 - x64) + x59*(x26 + x29 + x64) + x62 + x65 - T{72.0000000000000}*f_curr[12] - T{72.0000000000000}*f_curr[6];
  T x67 = T{6.00000000000000}*x12;
  T x68 = -x59*(x40 - x67) + x59*(x45 + x67) + x61 + x65 - T{72.0000000000000}*f_curr[16] - T{72.0000000000000}*f_curr[2];
  T strain = T{0.0277777777777778}*sqrt(x31*x31 + x48*x48 + x58*x58 + T{0.500000000000000}*(x63*x63) + T{0.500000000000000}*(x66*x66) + T{0.500000000000000}*(x68*x68));

  float vorticity = cell_curl_norm[gid];
  float q = vorticity*vorticity - strain*strain;
  q = q > 0 ? q : 0;

  surf3Dwrite(q, surface, iX*sizeof(float), iY, iZ);

  if (cell_q != nullptr) {
    cell_q[gid] = q;
  }
}

};

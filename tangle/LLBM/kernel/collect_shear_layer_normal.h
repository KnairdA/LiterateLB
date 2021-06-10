#pragma once
#include <LLBM/call_tag.h>

struct CollectShearLayerNormalsF {

using call_tag = tag::call_by_cell_id;

template <typename T, typename S>
__device__ static void apply(
    descriptor::D3Q19
  , S f_curr[19]
  , std::size_t gid
  , T* cell_rho 
  , T* cell_u
  , T* cell_shear_normal
) {
  T x0 = f_curr[10] + f_curr[13] + f_curr[17];
  T x1 = f_curr[14] + f_curr[5] + f_curr[6];
  T x2 = f_curr[1] + f_curr[2] + f_curr[4];
  T x3 = x0 + x1 + x2 + f_curr[0] + f_curr[11] + f_curr[12] + f_curr[15] + f_curr[16] + f_curr[18] + f_curr[3] + f_curr[7] + f_curr[8] + f_curr[9];
  T x4 = -f_curr[11] + f_curr[7];
  T x5 = -f_curr[15] + f_curr[3];
  T x6 = T{1} / (x3);
  T x7 = f_curr[0] - f_curr[18];
  T x13 = T{72.0000000000000}*f_curr[5];
  T x14 = T{72.0000000000000}*f_curr[13];
  T x37 = T{72.0000000000000}*f_curr[1];
  T x38 = T{72.0000000000000}*f_curr[17];
  T x58 = T{72.0000000000000}*f_curr[4];
  T x59 = T{72.0000000000000}*f_curr[14];
  T rho = x3;
  T x53 = T{2.00000000000000}*rho;
  T u_0 = x6*(x0 + x4 + x5 - f_curr[1] - f_curr[5] - f_curr[8]);
  T x8 = u_0*u_0;
  T x15 = -u_0;
  T x17 = -T{3.00000000000000}*x8;
  T x20 = T{6.00000000000000}*u_0;
  T x21 = -x20;
  T u_1 = x6*(x1 + x4 + x7 - f_curr[12] - f_curr[13] - f_curr[4]);
  T x9 = u_1*u_1;
  T x16 = x15 + u_1;
  T x18 = T{6.00000000000000}*u_1;
  T x19 = x17 + x18;
  T x22 = -T{3.00000000000000}*x9;
  T x27 = -u_1;
  T x28 = x27 + u_0;
  T x29 = x17 - x18;
  T x32 = u_0 + u_1;
  T x33 = T{9.00000000000000}*(x32*x32);
  T u_2 = x6*(x2 + x5 + x7 - f_curr[14] - f_curr[16] - f_curr[17]);
  T x10 = u_2*u_2;
  T x11 = x10 + x8 + x9;
  T x12 = pow(x11, T{-0.500000000000000});
  T x23 = T{2.00000000000000} - T{3.00000000000000}*x10;
  T x24 = x22 + x23;
  T x25 = x21 + x24;
  T x26 = rho*(x19 + x25 + T{9.00000000000000}*(x16*x16));
  T x30 = x20 + x24;
  T x31 = rho*(x29 + x30 + T{9.00000000000000}*(x28*x28));
  T x34 = rho*(x19 + x30 + x33) + rho*(x25 + x29 + x33) - T{72.0000000000000}*f_curr[11] - T{72.0000000000000}*f_curr[7];
  T x35 = x13 + x14 - x26 - x31 + x34;
  T x36 = x12*x35;
  T x39 = x15 + u_2;
  T x40 = T{6.00000000000000}*u_2;
  T x41 = x17 + x40;
  T x42 = rho*(x25 + x41 + T{9.00000000000000}*(x39*x39));
  T x43 = -u_2;
  T x44 = x43 + u_0;
  T x45 = -x40;
  T x46 = x17 + x45;
  T x47 = rho*(x30 + x46 + T{9.00000000000000}*(x44*x44));
  T x48 = u_0 + u_2;
  T x49 = T{9.00000000000000}*(x48*x48);
  T x50 = rho*(x25 + x46 + x49) + rho*(x30 + x41 + x49) - T{72.0000000000000}*f_curr[15] - T{72.0000000000000}*f_curr[3];
  T x51 = x37 + x38 - x42 - x47 + x50;
  T x52 = x12*x51;
  T x54 = x24 + T{6.00000000000000}*x8;
  T x55 = -x13 - x14 + x26 + x31 + x34;
  T x56 = -x37 - x38 + x42 + x47 + x50;
  T x57 = x53*(x20 + x54) + x53*(x21 + x54) + x55 + x56 - T{72.0000000000000}*f_curr[10] - T{72.0000000000000}*f_curr[8];
  T x60 = x27 + u_2;
  T x61 = x24 + x29;
  T x62 = rho*(x40 + x61 + T{9.00000000000000}*(x60*x60));
  T x63 = x43 + u_1;
  T x64 = x19 + x24;
  T x65 = rho*(x45 + x64 + T{9.00000000000000}*(x63*x63));
  T x66 = u_1 + u_2;
  T x67 = T{9.00000000000000}*(x66*x66);
  T x68 = rho*(x40 + x64 + x67) + rho*(x45 + x61 + x67) - T{72.0000000000000}*f_curr[0] - T{72.0000000000000}*f_curr[18];
  T x69 = x58 + x59 - x62 - x65 + x68;
  T x70 = x23 + T{6.00000000000000}*x9;
  T x71 = -x58 - x59 + x62 + x65 + x68;
  T x72 = x53*(x19 + x70) + x53*(x29 + x70) + x55 + x71 - T{72.0000000000000}*f_curr[12] - T{72.0000000000000}*f_curr[6];
  T x73 = T{6.00000000000000}*x10 + x22 + T{2.00000000000000};
  T x74 = x53*(x41 + x73) + x53*(x46 + x73) + x56 + x71 - T{72.0000000000000}*f_curr[16] - T{72.0000000000000}*f_curr[2];
  T x75 = ((x35*u_0 + x69*u_2 + x72*u_1)*u_1 + (x35*u_1 + x51*u_2 + x57*u_0)*u_0 + (x51*u_0 + x69*u_1 + x74*u_2)*u_2)/x11;
  T x76 = x12*x69;
  T n_0 = -T{0.0277777777777778}*x12*x57*u_0 - T{0.0277777777777778}*x36*u_1 - T{0.0277777777777778}*x52*u_2 + T{0.0277777777777778}*x75*u_0;
  T n_1 = -T{0.0277777777777778}*x12*x72*u_1 - T{0.0277777777777778}*x36*u_0 + T{0.0277777777777778}*x75*u_1 - T{0.0277777777777778}*x76*u_2;
  T n_2 = -T{0.0277777777777778}*x12*x74*u_2 - T{0.0277777777777778}*x52*u_0 + T{0.0277777777777778}*x75*u_2 - T{0.0277777777777778}*x76*u_1;

  cell_rho[gid] = rho;

  cell_u[3*gid+0] = u_0;
  cell_u[3*gid+1] = u_1;
  cell_u[3*gid+2] = u_2;

  float3 n = normalize(make_float3(n_0, n_1, n_2));
  cell_shear_normal[3*gid+0] = n.x;
  cell_shear_normal[3*gid+1] = n.y;
  cell_shear_normal[3*gid+2] = n.z;
}

};

struct CollectShearLayerVisibilityF {

using call_tag = tag::post_process_by_spatial_cell_mask;

template <typename T, typename S>
__device__ static void apply(
    descriptor::D3Q19
  , std::size_t gid
  , std::size_t iX
  , std::size_t iY
  , std::size_t iZ
  , T* shear_normal
  , float3 view_direction
  , cudaSurfaceObject_t surface
) {
  float3 n = make_float3(shear_normal[3*gid+0], shear_normal[3*gid+1], shear_normal[3*gid+2]);
  float visibility = dot(n, view_direction);
  surf3Dwrite(visibility, surface, iX*sizeof(float), iY, iZ);
}

};

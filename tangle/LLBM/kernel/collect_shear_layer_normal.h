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
  T x14 = T{72.0000000000000}*f_curr[5];
  T x15 = T{72.0000000000000}*f_curr[13];
  T x39 = T{72.0000000000000}*f_curr[1];
  T x40 = T{72.0000000000000}*f_curr[17];
  T x61 = T{72.0000000000000}*f_curr[4];
  T x62 = T{72.0000000000000}*f_curr[14];
  T rho = x3;
  T x56 = T{2.00000000000000}*rho;
  T u_0 = x6*(x0 + x4 + x5 - f_curr[1] - f_curr[5] - f_curr[8]);
  T x9 = u_0*u_0;
  T x16 = -u_0;
  T x18 = -T{3.00000000000000}*x9;
  T x21 = T{6.00000000000000}*u_0;
  T x22 = -x21;
  T x55 = T{0.0277777777777778}*u_0;
  T u_1 = x6*(x1 + x4 + x7 - f_curr[12] - f_curr[13] - f_curr[4]);
  T x8 = T{0.0277777777777778}*u_1;
  T x10 = u_1*u_1;
  T x17 = x16 + u_1;
  T x19 = T{6.00000000000000}*u_1;
  T x20 = x18 + x19;
  T x23 = -T{3.00000000000000}*x10;
  T x28 = -u_1;
  T x29 = x28 + u_0;
  T x30 = x18 - x19;
  T x33 = u_0 + u_1;
  T x34 = T{9.00000000000000}*(x33*x33);
  T u_2 = x6*(x2 + x5 + x7 - f_curr[14] - f_curr[16] - f_curr[17]);
  T x11 = u_2*u_2;
  T x12 = x10 + x11 + x9;
  T x13 = pow(x12, T{-0.500000000000000});
  T x24 = T{2.00000000000000} - T{3.00000000000000}*x11;
  T x25 = x23 + x24;
  T x26 = x22 + x25;
  T x27 = rho*(x20 + x26 + T{9.00000000000000}*(x17*x17));
  T x31 = x21 + x25;
  T x32 = rho*(x30 + x31 + T{9.00000000000000}*(x29*x29));
  T x35 = rho*(x20 + x31 + x34) + rho*(x26 + x30 + x34) - T{72.0000000000000}*f_curr[11] - T{72.0000000000000}*f_curr[7];
  T x36 = x14 + x15 - x27 - x32 + x35;
  T x37 = x13*x36;
  T x38 = T{0.0277777777777778}*u_2;
  T x41 = x16 + u_2;
  T x42 = T{6.00000000000000}*u_2;
  T x43 = x18 + x42;
  T x44 = rho*(x26 + x43 + T{9.00000000000000}*(x41*x41));
  T x45 = -u_2;
  T x46 = x45 + u_0;
  T x47 = -x42;
  T x48 = x18 + x47;
  T x49 = rho*(x31 + x48 + T{9.00000000000000}*(x46*x46));
  T x50 = u_0 + u_2;
  T x51 = T{9.00000000000000}*(x50*x50);
  T x52 = rho*(x26 + x48 + x51) + rho*(x31 + x43 + x51) - T{72.0000000000000}*f_curr[15] - T{72.0000000000000}*f_curr[3];
  T x53 = x39 + x40 - x44 - x49 + x52;
  T x54 = x13*x53;
  T x57 = x25 + T{6.00000000000000}*x9;
  T x58 = -x14 - x15 + x27 + x32 + x35;
  T x59 = -x39 - x40 + x44 + x49 + x52;
  T x60 = x56*(x21 + x57) + x56*(x22 + x57) + x58 + x59 - T{72.0000000000000}*f_curr[10] - T{72.0000000000000}*f_curr[8];
  T x63 = x28 + u_2;
  T x64 = x25 + x30;
  T x65 = rho*(x42 + x64 + T{9.00000000000000}*(x63*x63));
  T x66 = x45 + u_1;
  T x67 = x20 + x25;
  T x68 = rho*(x47 + x67 + T{9.00000000000000}*(x66*x66));
  T x69 = u_1 + u_2;
  T x70 = T{9.00000000000000}*(x69*x69);
  T x71 = rho*(x42 + x67 + x70) + rho*(x47 + x64 + x70) - T{72.0000000000000}*f_curr[0] - T{72.0000000000000}*f_curr[18];
  T x72 = x61 + x62 - x65 - x68 + x71;
  T x73 = T{6.00000000000000}*x10 + x24;
  T x74 = -x61 - x62 + x65 + x68 + x71;
  T x75 = x56*(x20 + x73) + x56*(x30 + x73) + x58 + x74 - T{72.0000000000000}*f_curr[12] - T{72.0000000000000}*f_curr[6];
  T x76 = T{6.00000000000000}*x11 + x23 + T{2.00000000000000};
  T x77 = x56*(x43 + x76) + x56*(x48 + x76) + x59 + x74 - T{72.0000000000000}*f_curr[16] - T{72.0000000000000}*f_curr[2];
  T x78 = ((x36*u_0 + x72*u_2 + x75*u_1)*u_1 + (x36*u_1 + x53*u_2 + x60*u_0)*u_0 + (x53*u_0 + x72*u_1 + x77*u_2)*u_2)/x12;
  T x79 = x13*x72;
  T n_0 = -x13*x55*x60 - x37*x8 - x38*x54 + x55*x78;
  T n_1 = -x13*x75*x8 - x37*x55 - x38*x79 + x78*x8;
  T n_2 = -x13*x38*x77 + x38*x78 - x54*x55 - x79*x8;

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

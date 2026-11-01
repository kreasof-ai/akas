#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const float* __restrict__ activation_scales, const uchar* __restrict__ activations, const int* __restrict__ counts, float* __restrict__ out, const int* __restrict__ routes, const bfloat16_t* __restrict__ scales, const bfloat16_t* __restrict__ scales2, const int* __restrict__ tile_experts, const int* __restrict__ tile_offsets, const uchar* __restrict__ weights, const uchar* __restrict__ weights2);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(const float* __restrict__ activation_scales, const uchar* __restrict__ activations, const int* __restrict__ counts, float* __restrict__ out, const int* __restrict__ routes, const bfloat16_t* __restrict__ scales, const bfloat16_t* __restrict__ scales2, const int* __restrict__ tile_experts, const int* __restrict__ tile_offsets, const uchar* __restrict__ weights, const uchar* __restrict__ weights2) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* selected = ((void*)((char*)buf_dyn_shmem + 0));
  void* paired_values = ((void*)((char*)buf_dyn_shmem + 128));
  void* rhs = ((void*)((char*)buf_dyn_shmem + 128));
  void* lhs = ((void*)((char*)buf_dyn_shmem + 32896));
  void* scale = ((void*)((char*)buf_dyn_shmem + 41088));
  float total[16];
  float block[16];
  float values[8];
  int expert = tile_experts[(((int)blockIdx.y) + 6336)];
  int tokens = tile_offsets[(((int)blockIdx.y) + 6336)];
  if (0 <= expert) {
    int condval;
    if ((expert < 64)) {
      condval = counts[((int64_t)expert)];
    } else {
      condval = 0;
    }
    if ((tokens * 32) < condval) {
      if (((int)threadIdx.x) < 32) {
        int condval_2;
        if ((expert < 64)) {
          condval_2 = counts[((int64_t)expert)];
        } else {
          condval_2 = 0;
        }
        int condval_1;
        if ((((tokens * 32) + ((int)threadIdx.x)) < condval_2)) {
          int condval_3;
          if (((0 <= tokens) && (expert < 64))) {
            condval_3 = routes[((((int64_t)expert) * (int64_t)2048) + min(((((int64_t)tokens) * (int64_t)32) + ((int64_t)((int)threadIdx.x))), (int64_t)2047))];
          } else {
            condval_3 = 0;
          }
          condval_1 = condval_3;
        } else {
          condval_1 = -1;
        }
        ((int*)selected)[((int)threadIdx.x)] = condval_1;
      }
    }
    int condval_4;
    if ((expert < 64)) {
      condval_4 = counts[((int64_t)expert)];
    } else {
      condval_4 = 0;
    }
    if ((tokens * 32) < condval_4) {
      __syncthreads();
    }
    int condval_5;
    if ((expert < 64)) {
      condval_5 = counts[((int64_t)expert)];
    } else {
      condval_5 = 0;
    }
    if ((tokens * 32) < condval_5) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float broadcast_var = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(total + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
      }
    }
    int condval_6;
    if ((expert < 64)) {
      condval_6 = counts[((int64_t)expert)];
    } else {
      condval_6 = 0;
    }
    if ((tokens * 32) < condval_6) {
      for (int tile = 0; tile < 16; ++tile) {
        __syncthreads();
        if (((int)threadIdx.x) < 32) {
          float condval_7;
          if ((0 <= ((int*)selected)[((int)threadIdx.x)])) {
            float condval_8;
            if ((((int*)selected)[((int)threadIdx.x)] < 8192)) {
              condval_8 = activation_scales[(((max(((int64_t)((int*)selected)[((int64_t)((int)threadIdx.x))]), (int64_t)0) >> (int64_t)2) * (int64_t)16) + ((int64_t)tile))];
            } else {
              condval_8 = 0x0p+0f/*0.000000e+00*/;
            }
            condval_7 = condval_8;
          } else {
            condval_7 = 0x1p+0f/*1.000000e+00*/;
          }
          ((float*)scale)[((int)threadIdx.x)] = condval_7;
        }
        __syncthreads();
        if (0 <= ((int*)selected)[(((int)threadIdx.x) >> 3)]) {
          for (int lane = 0; lane < 2; ++lane) {
            fp8_e4_t activations_local_cast_1[8];
            half_t lhs_local_cast[8];
            fp8_e4_t broadcast_var_1 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
            fp8_e4_8_t condval_9;
            if (((0 <= ((int*)selected)[(((int)threadIdx.x) >> 3)]) && (((int*)selected)[(((int)threadIdx.x) >> 3)] < 8192))) {
              condval_9 = *(fp8_e4_8_t*)(((fp8_e4_t*)activations) + (((((((int64_t)((int*)selected)[(((int64_t)((int)threadIdx.x)) >> (int64_t)3)]) >> (int64_t)2) * (int64_t)2048) + (((int64_t)tile) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)16)) + (((int64_t)lane) * (int64_t)8)));
            } else {
              condval_9 = make_fp8_e4_8_t(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
            }
            *(fp8_e4_8_t*)(activations_local_cast_1 + 0) = condval_9;
            uint4 __1;
            fp8_e4_8_t v_ = *(fp8_e4_8_t*)(activations_local_cast_1 + 0);
            ((half2*)(&__1))[0] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_))[0], __NV_E4M3);
            ((half2*)(&__1))[1] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_))[1], __NV_E4M3);
            ((half2*)(&__1))[2] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_))[2], __NV_E4M3);
            ((half2*)(&__1))[3] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_))[3], __NV_E4M3);
            *(uint4*)(lhs_local_cast + 0) = __1;
            *(uint4*)(((half_t*)lhs) + (((((((((int)threadIdx.x) & 7) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + lane) & 1) * 8))) = *(uint4*)(lhs_local_cast + 0);
          }
        } else {
          for (int lane_1 = 0; lane_1 < 2; ++lane_1) {
            half_t broadcast_var_2 = half_t(0x0p+0f/*0.000000e+00*/);
            *(uint4*)(((half_t*)lhs) + (((((((((int)threadIdx.x) & 7) >> 2) * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + lane_1) & 1) * 8))) = make_uint4(__pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2));
          }
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; ++i_1) {
          fp8_e4_t weights_local_cast_3[8];
          half_t rhs_local_cast_2[8];
          fp8_e4_t broadcast_var_3 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
          fp8_e4_8_t condval_10;
          if ((expert < 64)) {
            condval_10 = *(fp8_e4_8_t*)(((fp8_e4_t*)weights) + ((((((((int64_t)expert) * (int64_t)3145728) + (((int64_t)((int)blockIdx.x)) * (int64_t)131072)) + (((int64_t)i_1) * (int64_t)32768)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)tile) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_10 = make_fp8_e4_8_t(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
          }
          *(fp8_e4_8_t*)(weights_local_cast_3 + 0) = condval_10;
          uint4 __2;
          fp8_e4_8_t v__1 = *(fp8_e4_8_t*)(weights_local_cast_3 + 0);
          ((half2*)(&__2))[0] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__1))[0], __NV_E4M3);
          ((half2*)(&__2))[1] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__1))[1], __NV_E4M3);
          ((half2*)(&__2))[2] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__1))[2], __NV_E4M3);
          ((half2*)(&__2))[3] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__1))[3], __NV_E4M3);
          *(uint4*)(rhs_local_cast_2 + 0) = __2;
          *(uint4*)(((half_t*)rhs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(rhs_local_cast_2 + 0);
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 4; ++i_2) {
          fp8_e4_t weights2_local_cast_5[8];
          half_t rhs_local_cast_4[8];
          fp8_e4_t broadcast_var_4 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
          fp8_e4_8_t condval_11;
          if ((expert < 64)) {
            condval_11 = *(fp8_e4_8_t*)(((fp8_e4_t*)weights2) + ((((((((int64_t)expert) * (int64_t)3145728) + (((int64_t)((int)blockIdx.x)) * (int64_t)131072)) + (((int64_t)i_2) * (int64_t)32768)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)tile) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_11 = make_fp8_e4_8_t(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
          }
          *(fp8_e4_8_t*)(weights2_local_cast_5 + 0) = condval_11;
          uint4 __3;
          fp8_e4_8_t v__2 = *(fp8_e4_8_t*)(weights2_local_cast_5 + 0);
          ((half2*)(&__3))[0] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__2))[0], __NV_E4M3);
          ((half2*)(&__3))[1] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__2))[1], __NV_E4M3);
          ((half2*)(&__3))[2] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__2))[2], __NV_E4M3);
          ((half2*)(&__3))[3] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__2))[3], __NV_E4M3);
          *(uint4*)(rhs_local_cast_4 + 0) = __3;
          *(uint4*)(((half_t*)rhs) + (((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)) = *(uint4*)(rhs_local_cast_4 + 0);
        }
        {
          half_t A_local[8];
          half_t B_local[16];
          #pragma unroll
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(block + (i_3 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
          }
          __syncthreads();
          for (int ki = 0; ki < 8; ++ki) {
            tl::ptx_ldmatrix_x4((&(((half_t*)lhs)[(((((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
            #pragma unroll
            for (int i_4 = 0; i_4 < 2; ++i_4) {
              tl::ptx_ldmatrix_x4((&(((half_t*)rhs)[(((((((((ki >> 2) * 8192) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
            }
            for (int j = 0; j < 2; ++j) {
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(block + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
              tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(block + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
            }
          }
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 16; ++i_5) {
          float condval_12;
          if ((((int)threadIdx.x) < 128)) {
            bfloat16_t condval_13;
            if ((expert < 64)) {
              condval_13 = scales[(((((int64_t)expert) * (int64_t)192) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)16)) + ((int64_t)tile))];
            } else {
              condval_13 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            }
            condval_12 = ((float)condval_13);
          } else {
            bfloat16_t condval_14;
            if ((expert < 64)) {
              condval_14 = scales2[(((((int64_t)expert) * (int64_t)192) + ((((int64_t)((int)blockIdx.x)) >> (int64_t)1) * (int64_t)16)) + ((int64_t)tile))];
            } else {
              condval_14 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            }
            condval_12 = ((float)condval_14);
          }
          float coefficient = condval_12;
          float product = __fmul_rn(block[i_5], ((float*)scale)[(((((((int)threadIdx.x) & 63) >> 5) * 16) + (((i_5 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
          total[i_5] = __fmaf_rn(product, coefficient, total[i_5]);
        }
      }
    }
  }
  __syncthreads();
  if (0 <= expert) {
    int condval_15;
    if ((expert < 64)) {
      condval_15 = counts[((int64_t)expert)];
    } else {
      condval_15 = 0;
    }
    if ((tokens * 32) < condval_15) {
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        *(float2*)(((float*)paired_values) + ((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_6 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + ((i_6 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(total + (i_6 * 2));
      }
    }
    int condval_16;
    if ((expert < 64)) {
      condval_16 = counts[((int64_t)expert)];
    } else {
      condval_16 = 0;
    }
    if ((tokens * 32) < condval_16) {
      __syncthreads();
    }
    int condval_17;
    if ((expert < 64)) {
      condval_17 = counts[((int64_t)expert)];
    } else {
      condval_17 = 0;
    }
    if ((tokens * 32) < condval_17) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        float broadcast_var_6 = 0x1p+0f/*1.000000e+00*/;
        float broadcast_var_7 = -0x1p+0f/*-1.000000e+00*/;
        float4 __4;
          float4 __5;
            float4 v__3 = *(float4*)(((float*)paired_values) + (((i_7 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)));
            float4 __6;
              float4 v__4 = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
              float4 __7;
              float4 __8;
                float4 v__5 = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
                __8.x = (v__3.x*v__5.x);
                __8.y = (v__3.y*v__5.y);
                __8.z = (v__3.z*v__5.z);
                __8.w = (v__3.w*v__5.w);
              __7.x = expf(__8.x);
              __7.y = expf(__8.y);
              __7.z = expf(__8.z);
              __7.w = expf(__8.w);
              __6.x = (v__4.x+__7.x);
              __6.y = (v__4.y+__7.y);
              __6.z = (v__4.z+__7.z);
              __6.w = (v__4.w+__7.w);
            __5.x = (v__3.x/__6.x);
            __5.y = (v__3.y/__6.y);
            __5.z = (v__3.z/__6.z);
            __5.w = (v__3.w/__6.w);
          float4 v__6 = *(float4*)(((float*)paired_values) + ((((i_7 * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + 64));
          __4.x = (__5.x*v__6.x);
          __4.y = (__5.y*v__6.y);
          __4.z = (__5.z*v__6.z);
          __4.w = (__5.w*v__6.w);
        *(float4*)(values + (i_7 * 4)) = __4;
      }
    }
    int condval_18;
    if ((expert < 64)) {
      condval_18 = counts[((int64_t)expert)];
    } else {
      condval_18 = 0;
    }
    if ((tokens * 32) < condval_18) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        if (0 <= ((int*)selected)[((i_8 * 16) + (((int)threadIdx.x) >> 4))]) {
          if (0 <= ((int*)selected)[((i_8 * 16) + (((int)threadIdx.x) >> 4))]) {
            if (((int*)selected)[((i_8 * 16) + (((int)threadIdx.x) >> 4))] < 8192) {
              *(float4*)(out + (((((int64_t)((int*)selected)[((((int64_t)i_8) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4))]) * (int64_t)1536) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)4))) = *(float4*)(values + (i_8 * 4));
            }
          }
        }
      }
    }
  }
}


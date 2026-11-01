#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const float* __restrict__ activation_scales, const uchar* __restrict__ activations, float* __restrict__ out, const bfloat16_t* __restrict__ scales, const uchar* __restrict__ weights);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const float* __restrict__ activation_scales, const uchar* __restrict__ activations, float* __restrict__ out, const bfloat16_t* __restrict__ scales, const uchar* __restrict__ weights) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* lhs = ((void*)((char*)buf_dyn_shmem + 0));
  void* rhs = ((void*)((char*)buf_dyn_shmem + 4096));
  void* activation_scales_1 = ((void*)((char*)buf_dyn_shmem + 20480));
  float total[8];
  float block[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(total + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    fp8_e4_t activations_local_cast_1[8];
    half_t lhs_local_cast[8];
    *(fp8_e4_8_t*)(activations_local_cast_1 + 0) = *(fp8_e4_8_t*)(((fp8_e4_t*)activations) + ((((((int)blockIdx.y) * 32768) + (i_1 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + ((((int)threadIdx.x) & 15) * 8)));
    uint4 __1;
    fp8_e4_8_t v_ = *(fp8_e4_8_t*)(activations_local_cast_1 + 0);
    ((half2*)(&__1))[0] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_))[0], __NV_E4M3);
    ((half2*)(&__1))[1] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_))[1], __NV_E4M3);
    ((half2*)(&__1))[2] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_))[2], __NV_E4M3);
    ((half2*)(&__1))[3] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v_))[3], __NV_E4M3);
    *(uint4*)(lhs_local_cast + 0) = __1;
    *(uint4*)(((half_t*)lhs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(lhs_local_cast + 0);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    fp8_e4_t weights_local_cast_3[8];
    half_t rhs_local_cast_2[8];
    *(fp8_e4_8_t*)(weights_local_cast_3 + 0) = *(fp8_e4_8_t*)(((fp8_e4_t*)weights) + ((((((int)blockIdx.x) * 131072) + (i_2 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + ((((int)threadIdx.x) & 15) * 8)));
    uint4 __2;
    fp8_e4_8_t v__1 = *(fp8_e4_8_t*)(weights_local_cast_3 + 0);
    ((half2*)(&__2))[0] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__1))[0], __NV_E4M3);
    ((half2*)(&__2))[1] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__1))[1], __NV_E4M3);
    ((half2*)(&__2))[2] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__1))[2], __NV_E4M3);
    ((half2*)(&__2))[3] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__1))[3], __NV_E4M3);
    *(uint4*)(rhs_local_cast_2 + 0) = __2;
    *(uint4*)(((half_t*)rhs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(rhs_local_cast_2 + 0);
  }
  tl::cp_async_commit();
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs<4>((&(((float*)activation_scales_1)[((int)threadIdx.x)])), (&(activation_scales[((((int)blockIdx.y) * 256) + (((int)threadIdx.x) * 16))])));
  }
  tl::cp_async_commit();
  for (int local_tile = 0; local_tile < 15; ++local_tile) {
    __syncthreads();
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[8];
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(block + (i_3 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      }
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)lhs)[((((((ki >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((half_t*)rhs)[(((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(block + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(block + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      fp8_e4_t activations_local_cast_5[8];
      half_t lhs_local_cast_4[8];
      *(fp8_e4_8_t*)(activations_local_cast_5 + 0) = *(fp8_e4_8_t*)(((fp8_e4_t*)activations) + ((((((((int)blockIdx.y) * 32768) + (i_4 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (local_tile * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 128));
      uint4 __3;
      fp8_e4_8_t v__2 = *(fp8_e4_8_t*)(activations_local_cast_5 + 0);
      ((half2*)(&__3))[0] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__2))[0], __NV_E4M3);
      ((half2*)(&__3))[1] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__2))[1], __NV_E4M3);
      ((half2*)(&__3))[2] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__2))[2], __NV_E4M3);
      ((half2*)(&__3))[3] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__2))[3], __NV_E4M3);
      *(uint4*)(lhs_local_cast_4 + 0) = __3;
      *(uint4*)(((half_t*)lhs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(lhs_local_cast_4 + 0);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      fp8_e4_t weights_local_cast_7[8];
      half_t rhs_local_cast_6[8];
      *(fp8_e4_8_t*)(weights_local_cast_7 + 0) = *(fp8_e4_8_t*)(((fp8_e4_t*)weights) + ((((((((int)blockIdx.x) * 131072) + (i_5 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (local_tile * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 128));
      uint4 __4;
      fp8_e4_8_t v__3 = *(fp8_e4_8_t*)(weights_local_cast_7 + 0);
      ((half2*)(&__4))[0] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__3))[0], __NV_E4M3);
      ((half2*)(&__4))[1] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__3))[1], __NV_E4M3);
      ((half2*)(&__4))[2] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__3))[2], __NV_E4M3);
      ((half2*)(&__4))[3] = __tl_cvt_fp8x2_to_half2((reinterpret_cast<__nv_fp8x2_storage_t*>(&v__3))[3], __NV_E4M3);
      *(uint4*)(rhs_local_cast_6 + 0) = __4;
      *(uint4*)(((half_t*)rhs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(rhs_local_cast_6 + 0);
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      float activation_scales_local_cast_8[2];
      *(float2*)(activation_scales_local_cast_8 + 0) = make_float2(((float*)activation_scales_1)[(((i_6 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)activation_scales_1)[(((i_6 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2))]);
      float2 __5;
        float2 __6;
          float2 __7;
            float2 v__4 = *(float2*)(block + (i_6 * 2));
            float2 v__5 = *(float2*)(activation_scales_local_cast_8 + 0);
            __7.x = (v__4.x*v__5.x);
            __7.y = (v__4.y*v__5.y);
          float2 v__6 = make_float2(((float)scales[(((((int)blockIdx.x) >> 1) * 16) + local_tile)]), ((float)scales[(((((int)blockIdx.x) >> 1) * 16) + local_tile)]));
          __6.x = (__7.x*v__6.x);
          __6.y = (__7.y*v__6.y);
        float2 v__7 = *(float2*)(total + (i_6 * 2));
        __5.x = (__6.x+v__7.x);
        __5.y = (__6.y+v__7.y);
      *(float2*)(total + (i_6 * 2)) = __5;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      tl::cp_async_gs<4>((&(((float*)activation_scales_1)[((int)threadIdx.x)])), (&(activation_scales[((((((int)blockIdx.y) * 256) + (((int)threadIdx.x) * 16)) + local_tile) + 1)])));
    }
    tl::cp_async_commit();
  }
  __syncthreads();
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_1[8];
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(block + (i_7 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((half_t*)lhs)[((((((ki_1 >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[0])));
      tl::ptx_ldmatrix_x4((&(((half_t*)rhs)[(((((((ki_1 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
      tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(block + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
      tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(block + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    float activation_scales_local_cast_9[2];
    *(float2*)(activation_scales_local_cast_9 + 0) = make_float2(((float*)activation_scales_1)[(((i_8 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)activation_scales_1)[(((i_8 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2))]);
    float2 __8;
      float2 __9;
        float2 __10;
          float2 v__8 = *(float2*)(block + (i_8 * 2));
          float2 v__9 = *(float2*)(activation_scales_local_cast_9 + 0);
          __10.x = (v__8.x*v__9.x);
          __10.y = (v__8.y*v__9.y);
        float2 v__10 = make_float2(((float)scales[(((((int)blockIdx.x) >> 1) * 16) + 15)]), ((float)scales[(((((int)blockIdx.x) >> 1) * 16) + 15)]));
        __9.x = (__10.x*v__10.x);
        __9.y = (__10.y*v__10.y);
      float2 v__11 = *(float2*)(total + (i_8 * 2));
      __8.x = (__9.x+v__11.x);
      __8.y = (__9.y+v__11.y);
    *(float2*)(total + (i_8 * 2)) = __8;
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    *(float2*)(out + (((((((((int)blockIdx.y) * 32768) + ((i_9 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_9 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(total + (i_9 * 2));
  }
}


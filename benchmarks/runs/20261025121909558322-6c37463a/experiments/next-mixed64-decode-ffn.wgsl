// Function: kernel_kernel
//----------------------------------------
// Function: kernel_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> out : array<f32>;
@group(0) @binding(1) var<storage, read> w : array<u32>;
@group(0) @binding(2) var<storage, read> w2 : array<u32>;
@group(0) @binding(3) var<storage, read> x : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> v_4 : array<f32, 128>;
var<workgroup> v_5 : array<f32, 128>;
@compute @workgroup_size(128, 1, 1)
fn kernel_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
  @builtin(subgroup_size) tensorSubgroupSize : u32
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var v : array<f32, 1>;
  var v_1 : array<f32, 1>;
  var v_2 : array<f32, 1>;
  var v_3 : array<f32, 1>;
  v[0i] = 0.000000e+00f;
  v_1[0i] = 0.000000e+00f;
  v_2[0i] = 0.000000e+00f;
  v_3[0i] = 0.000000e+00f;
  for (var tile : i32 = 0i; tile < 8i; tile++) {
    let bits : u32 = ((w[((((v__1 * 1152i) + ((i32(threadIdx.x)>>5u) * 288i)) + (tile * 36i)) + ((((i32(threadIdx.x) & 31i)>>2u) * 18i)>>2u))]>>u32((((i32(threadIdx.x) & 7i)>>2u) * 16i))) & 65535u);
    let scale : f32 = tensor_unpack_f16(bits);
    let packed : u32 = tensor_byte_align_u32(w[(((((v__1 * 1152i) + ((i32(threadIdx.x)>>5u) * 288i)) + (tile * 36i)) + (((((i32(threadIdx.x) & 31i)>>2u) * 9i) + 1i)>>1u)) + (i32(threadIdx.x) & 3i))], w[((((((v__1 * 1152i) + ((i32(threadIdx.x)>>5u) * 288i)) + (tile * 36i)) + ((((i32(threadIdx.x) & 31i)>>2u) * 9i)>>1u)) + (i32(threadIdx.x) & 3i)) + 1i)], u32((((((i32(threadIdx.x) & 31i)>>2u) * 18i) + 2i) & 3i)));
    let left : vec4<f32> = vec4<f32>(x[(((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i))], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 1i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 2i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 3i)]);
    let right : vec4<f32> = vec4<f32>((f32(((packed>>0u) & 15u)) - 8.000000e+00f), (f32(((packed>>8u) & 15u)) - 8.000000e+00f), (f32(((packed>>16u) & 15u)) - 8.000000e+00f), (f32(((packed>>24u) & 15u)) - 8.000000e+00f));
    v[0i] = fma(scale, dot(left, right), v[0i]);
    let left_1 : vec4<f32> = vec4<f32>(x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 16i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 17i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 18i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 19i)]);
    let right_1 : vec4<f32> = vec4<f32>((f32(((packed>>4u) & 15u)) - 8.000000e+00f), (f32(((packed>>12u) & 15u)) - 8.000000e+00f), (f32(((packed>>20u) & 15u)) - 8.000000e+00f), (f32(((packed>>28u) & 15u)) - 8.000000e+00f));
    v[0i] = fma(scale, dot(left_1, right_1), v[0i]);
    let bits_1 : u32 = ((w2[((((v__1 * 1152i) + ((i32(threadIdx.x)>>5u) * 288i)) + (tile * 36i)) + ((((i32(threadIdx.x) & 31i)>>2u) * 18i)>>2u))]>>u32((((i32(threadIdx.x) & 7i)>>2u) * 16i))) & 65535u);
    let scale_1 : f32 = tensor_unpack_f16(bits_1);
    let packed_1 : u32 = tensor_byte_align_u32(w2[(((((v__1 * 1152i) + ((i32(threadIdx.x)>>5u) * 288i)) + (tile * 36i)) + (((((i32(threadIdx.x) & 31i)>>2u) * 9i) + 1i)>>1u)) + (i32(threadIdx.x) & 3i))], w2[((((((v__1 * 1152i) + ((i32(threadIdx.x)>>5u) * 288i)) + (tile * 36i)) + ((((i32(threadIdx.x) & 31i)>>2u) * 9i)>>1u)) + (i32(threadIdx.x) & 3i)) + 1i)], u32((((((i32(threadIdx.x) & 31i)>>2u) * 18i) + 2i) & 3i)));
    let left_2 : vec4<f32> = vec4<f32>(x[(((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i))], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 1i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 2i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 3i)]);
    let right_2 : vec4<f32> = vec4<f32>((f32(((packed_1>>0u) & 15u)) - 8.000000e+00f), (f32(((packed_1>>8u) & 15u)) - 8.000000e+00f), (f32(((packed_1>>16u) & 15u)) - 8.000000e+00f), (f32(((packed_1>>24u) & 15u)) - 8.000000e+00f));
    v_1[0i] = fma(scale_1, dot(left_2, right_2), v_1[0i]);
    let left_3 : vec4<f32> = vec4<f32>(x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 16i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 17i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 18i)], x[((((tile * 256i) + (((i32(threadIdx.x) & 31i)>>2u) * 32i)) + ((i32(threadIdx.x) & 3i) * 4i)) + 19i)]);
    let right_3 : vec4<f32> = vec4<f32>((f32(((packed_1>>4u) & 15u)) - 8.000000e+00f), (f32(((packed_1>>12u) & 15u)) - 8.000000e+00f), (f32(((packed_1>>20u) & 15u)) - 8.000000e+00f), (f32(((packed_1>>28u) & 15u)) - 8.000000e+00f));
    v_1[0i] = fma(scale_1, dot(left_3, right_3), v_1[0i]);
  }
  let value : f32 = v[0i];
  v_2[0i] = (v_2[0i] + value);
  let value_1 : f32 = v_1[0i];
  v_3[0i] = (v_3[0i] + value_1);
  if (1u <= (tensorSubgroupSize>>5u)) {
    v_2[0i] = (v_2[0i] + subgroupShuffleXor(v_2[0i], 16u));
    v_3[0i] = (v_3[0i] + subgroupShuffleXor(v_3[0i], 16u));
    v_2[0i] = (v_2[0i] + subgroupShuffleXor(v_2[0i], 8u));
    v_3[0i] = (v_3[0i] + subgroupShuffleXor(v_3[0i], 8u));
    v_2[0i] = (v_2[0i] + subgroupShuffleXor(v_2[0i], 4u));
    v_3[0i] = (v_3[0i] + subgroupShuffleXor(v_3[0i], 4u));
    v_2[0i] = (v_2[0i] + subgroupShuffleXor(v_2[0i], 2u));
    v_3[0i] = (v_3[0i] + subgroupShuffleXor(v_3[0i], 2u));
    v_2[0i] = (v_2[0i] + subgroupShuffleXor(v_2[0i], 1u));
    v_3[0i] = (v_3[0i] + subgroupShuffleXor(v_3[0i], 1u));
    let value_2 : f32 = v_2[0i];
    v_4[i32(threadIdx.x)] = value_2;
    let value_3 : f32 = v_3[0i];
    v_5[i32(threadIdx.x)] = value_3;
  } else {
    let value_4 : f32 = v_2[0i];
    v_4[i32(threadIdx.x)] = value_4;
    let value_5 : f32 = v_3[0i];
    v_5[i32(threadIdx.x)] = value_5;
    workgroupBarrier();
    if ((i32(threadIdx.x) & 31i) < 16i) {
      v_4[i32(threadIdx.x)] = (v_4[i32(threadIdx.x)] + v_4[(i32(threadIdx.x) + 16i)]);
      v_5[i32(threadIdx.x)] = (v_5[i32(threadIdx.x)] + v_5[(i32(threadIdx.x) + 16i)]);
    }
    workgroupBarrier();
    if ((i32(threadIdx.x) & 31i) < 8i) {
      v_4[i32(threadIdx.x)] = (v_4[i32(threadIdx.x)] + v_4[(i32(threadIdx.x) + 8i)]);
      v_5[i32(threadIdx.x)] = (v_5[i32(threadIdx.x)] + v_5[(i32(threadIdx.x) + 8i)]);
    }
    workgroupBarrier();
    if ((i32(threadIdx.x) & 31i) < 4i) {
      v_4[i32(threadIdx.x)] = (v_4[i32(threadIdx.x)] + v_4[(i32(threadIdx.x) + 4i)]);
      v_5[i32(threadIdx.x)] = (v_5[i32(threadIdx.x)] + v_5[(i32(threadIdx.x) + 4i)]);
    }
    workgroupBarrier();
    if ((i32(threadIdx.x) & 31i) < 2i) {
      v_4[i32(threadIdx.x)] = (v_4[i32(threadIdx.x)] + v_4[(i32(threadIdx.x) + 2i)]);
      v_5[i32(threadIdx.x)] = (v_5[i32(threadIdx.x)] + v_5[(i32(threadIdx.x) + 2i)]);
    }
    workgroupBarrier();
    if ((i32(threadIdx.x) & 31i) < 1i) {
      v_4[i32(threadIdx.x)] = (v_4[i32(threadIdx.x)] + v_4[(i32(threadIdx.x) + 1i)]);
      v_5[i32(threadIdx.x)] = (v_5[i32(threadIdx.x)] + v_5[(i32(threadIdx.x) + 1i)]);
    }
    workgroupBarrier();
  }
  if (((i32(threadIdx.x) % 32i) == 0i) & bool(1)) {
    out[((v__1 * 4i) + (i32(threadIdx.x)>>5u))] = ((v_4[i32(threadIdx.x)] / (1.000000e+00f + exp((v_4[i32(threadIdx.x)] * -1.000000e+00f)))) * v_5[i32(threadIdx.x)]);
  }
}



fn tensor_byte_align_u32(lo:u32, hi:u32, byte:u32)->u32 {
  let shift=byte*8u;
  return (lo>>shift) | select(0u,hi<<((32u-shift)&31u),shift!=0u);
}

fn tensor_unpack_f16(bits:u32)->f32 {
  return unpack2x16float(bits).x;
}

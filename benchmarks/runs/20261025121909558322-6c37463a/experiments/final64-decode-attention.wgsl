// Function: kernel_kernel
//----------------------------------------
// Function: kernel_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> enabled : array<i32>;
@group(0) @binding(1) var<storage, read> kc : array<f16>;
@group(0) @binding(2) var<storage, read> ks : array<f32>;
@group(0) @binding(3) var<storage, read> mapping : array<i32>;
@group(0) @binding(4) var<storage, read_write> out : array<f32>;
@group(0) @binding(5) var<storage, read> positions : array<i32>;
@group(0) @binding(6) var<storage, read> q : array<f32>;
@group(0) @binding(7) var<storage, read> vc : array<f16>;
@group(0) @binding(8) var<storage, read> vs : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(9) var<uniform> podArgs : PODArgs;

var<workgroup> keys : array<u32, 1056>;
var<workgroup> values : array<u32, 1024>;
var<workgroup> probabilities : array<f32, 128>;
@compute @workgroup_size(256, 1, 1)
fn kernel_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var v : array<f32, 1>;
  var v_1 : array<f32, 1>;
  var v_2 : array<f32, 1>;
  var v_3 : array<f32, 1>;
  var v_4 : array<f32, 1>;
  var v_5 : array<f32, 1>;
  var v_6 : array<f32, 1>;
  var v_7 : array<f32, 1>;
  var v_8 : array<f32, 1>;
  var v_9 : array<f32, 1>;
  var v_10 : array<f32, 1>;
  var v_11 : array<f32, 1>;
  var v_12 : array<f32, 1>;
  var v_13 : array<f32, 1>;
  var v_14 : array<f32, 1>;
  var v_15 : array<f32, 1>;
  var v_16 : array<f32, 1>;
  var v_17 : array<f32, 1>;
  var v_18 : array<f32, 1>;
  var v_19 : array<f32, 1>;
  var v_20 : array<f32, 1>;
  var v_21 : array<f32, 1>;
  var v_22 : array<f32, 1>;
  var v_23 : array<f32, 1>;
  var v_24 : array<f32, 1>;
  var v_25 : array<f32, 1>;
  var v_26 : array<f32, 1>;
  var v_27 : array<f32, 1>;
  var v_28 : array<f32, 1>;
  var v_29 : array<f32, 1>;
  var v_30 : array<f32, 1>;
  var v_31 : array<f32, 1>;
  var maximum : array<f32, 1>;
  var normalizer : array<f32, 1>;
  var acc : array<f32, 1>;
  var score : array<f32, 1>;
  var probability : array<f32, 1>;
  var peak : array<f32, 1>;
  var norm : array<f32, 1>;
  var second : array<f32, 1>;
    var condval : i32;
  if ((enabled[0i] != 0i)) {
    condval = min((positions[0i] + 1i), 131072i);
} else {
    condval = 0i;
}
let end : i32 = condval;
  v[0i] = 0.000000e+00f;
  v_1[0i] = 0.000000e+00f;
  v_2[0i] = 0.000000e+00f;
  v_3[0i] = 0.000000e+00f;
  v_4[0i] = 0.000000e+00f;
  v_5[0i] = 0.000000e+00f;
  v_6[0i] = 0.000000e+00f;
  v_7[0i] = 0.000000e+00f;
  v_8[0i] = 0.000000e+00f;
  v_9[0i] = 0.000000e+00f;
  v_10[0i] = 0.000000e+00f;
  v_11[0i] = 0.000000e+00f;
  v_12[0i] = 0.000000e+00f;
  v_13[0i] = 0.000000e+00f;
  v_14[0i] = 0.000000e+00f;
  v_15[0i] = 0.000000e+00f;
  v_16[0i] = 0.000000e+00f;
  v_17[0i] = 0.000000e+00f;
  v_18[0i] = 0.000000e+00f;
  v_19[0i] = 0.000000e+00f;
  v_20[0i] = 0.000000e+00f;
  v_21[0i] = 0.000000e+00f;
  v_22[0i] = 0.000000e+00f;
  v_23[0i] = 0.000000e+00f;
  v_24[0i] = 0.000000e+00f;
  v_25[0i] = 0.000000e+00f;
  v_26[0i] = 0.000000e+00f;
  v_27[0i] = 0.000000e+00f;
  v_28[0i] = 0.000000e+00f;
  v_29[0i] = 0.000000e+00f;
  v_30[0i] = 0.000000e+00f;
  v_31[0i] = 0.000000e+00f;
  if ((i32(blockIdx.y) * 128i) < end) {
    v[0i] = q[(((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i))];
    v_1[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 1i)];
    v_2[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 2i)];
    v_3[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 3i)];
    v_4[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 4i)];
    v_5[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 5i)];
    v_6[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 6i)];
    v_7[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 7i)];
    v_8[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 8i)];
    v_9[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 9i)];
    v_10[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 10i)];
    v_11[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 11i)];
    v_12[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 12i)];
    v_13[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 13i)];
    v_14[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 14i)];
    v_15[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 15i)];
    v_16[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 16i)];
    v_17[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 17i)];
    v_18[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 18i)];
    v_19[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 19i)];
    v_20[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 20i)];
    v_21[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 21i)];
    v_22[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 22i)];
    v_23[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 23i)];
    v_24[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 24i)];
    v_25[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 25i)];
    v_26[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 26i)];
    v_27[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 27i)];
    v_28[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 28i)];
    v_29[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 29i)];
    v_30[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 30i)];
    v_31[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 1i) * 32i)) + 31i)];
  }
  let _tmp : f32 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  maximum[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  normalizer[0i] = 0.000000e+00f;
  acc[0i] = 0.000000e+00f;
  for (var tile : i32 = 0i; tile < (((max((end - (i32(blockIdx.y) * 128i)), 0i) + 8191i)>>13u) * 4i); tile++) {
    if (((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) < end) {
      for (var load : i32 = 0i; load < 4i; load++) {
                var condval_1 : f32;
        if ((((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) < end)) {
          condval_1 = (f32(kc[(((((((((((v__1 * 131072i) + ((tile>>2u) * 8192i)) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 1048575i)>>17u) * 8388608i) + ((((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 131071i) * 64i)) + ((i32(threadIdx.x) & 31i) * 2i))]) * ks[(v__1 * 131072i)]);
} else {
          condval_1 = 0.000000e+00f;
}
let a : f32 = condval_1;
                var condval_2 : f32;
        if ((((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) < end)) {
          condval_2 = (f32(kc[((((((((((((((tile>>2u) * 524288i) + (i32(blockIdx.y) * 8192i)) + ((tile & 3i) * 2048i)) + (load * 512i)) + (i32(threadIdx.x) * 2i)) + 1i)>>23u) + v__1) & 7i) * 8388608i) + ((((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 131071i) * 64i)) + ((i32(threadIdx.x) & 31i) * 2i)) + 1i)]) * ks[(v__1 * 131072i)]);
} else {
          condval_2 = 0.000000e+00f;
}
let b : f32 = condval_2;
        keys[((((i32(threadIdx.x) & 31i) * 33i) + (load * 8i)) + (i32(threadIdx.x)>>5u))] = pack2x16float(vec2<f32>(a, b));
                var condval_3 : f32;
        if ((((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) < end)) {
          condval_3 = (f32(vc[(((((((((((v__1 * 131072i) + ((tile>>2u) * 8192i)) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 1048575i)>>17u) * 8388608i) + ((((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 131071i) * 64i)) + ((i32(threadIdx.x) & 31i) * 2i))]) * vs[(v__1 * 131072i)]);
} else {
          condval_3 = 0.000000e+00f;
}
let c : f32 = condval_3;
                var condval_4 : f32;
        if ((((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) < end)) {
          condval_4 = (f32(vc[((((((((((((((tile>>2u) * 524288i) + (i32(blockIdx.y) * 8192i)) + ((tile & 3i) * 2048i)) + (load * 512i)) + (i32(threadIdx.x) * 2i)) + 1i)>>23u) + v__1) & 7i) * 8388608i) + ((((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 131071i) * 64i)) + ((i32(threadIdx.x) & 31i) * 2i)) + 1i)]) * vs[(v__1 * 131072i)]);
} else {
          condval_4 = 0.000000e+00f;
}
let e : f32 = condval_4;
        values[((load * 256i) + i32(threadIdx.x))] = pack2x16float(vec2<f32>(c, e));
      }
      workgroupBarrier();
      score[0i] = 0.000000e+00f;
      let left : vec4<f32> = vec4<f32>(v[0i], v_1[0i], v_2[0i], v_3[0i]);
      let right : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[(((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u))]>>0u)), tensor_unpack_f16((keys[(((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u))]>>16u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 33i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 33i)]>>16u)));
      score[0i] = (score[0i] + dot(left, right));
      let left_1 : vec4<f32> = vec4<f32>(v_4[0i], v_5[0i], v_6[0i], v_7[0i]);
      let right_1 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 66i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 66i)]>>16u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 99i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 99i)]>>16u)));
      score[0i] = (score[0i] + dot(left_1, right_1));
      let left_2 : vec4<f32> = vec4<f32>(v_8[0i], v_9[0i], v_10[0i], v_11[0i]);
      let right_2 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 132i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 132i)]>>16u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 165i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 165i)]>>16u)));
      score[0i] = (score[0i] + dot(left_2, right_2));
      let left_3 : vec4<f32> = vec4<f32>(v_12[0i], v_13[0i], v_14[0i], v_15[0i]);
      let right_3 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 198i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 198i)]>>16u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 231i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 231i)]>>16u)));
      score[0i] = (score[0i] + dot(left_3, right_3));
      let left_4 : vec4<f32> = vec4<f32>(v_16[0i], v_17[0i], v_18[0i], v_19[0i]);
      let right_4 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 264i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 264i)]>>16u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 297i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 297i)]>>16u)));
      score[0i] = (score[0i] + dot(left_4, right_4));
      let left_5 : vec4<f32> = vec4<f32>(v_20[0i], v_21[0i], v_22[0i], v_23[0i]);
      let right_5 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 330i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 330i)]>>16u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 363i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 363i)]>>16u)));
      score[0i] = (score[0i] + dot(left_5, right_5));
      let left_6 : vec4<f32> = vec4<f32>(v_24[0i], v_25[0i], v_26[0i], v_27[0i]);
      let right_6 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 396i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 396i)]>>16u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 429i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 429i)]>>16u)));
      score[0i] = (score[0i] + dot(left_6, right_6));
      let left_7 : vec4<f32> = vec4<f32>(v_28[0i], v_29[0i], v_30[0i], v_31[0i]);
      let right_7 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 462i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 462i)]>>16u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 495i)]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 528i) + ((i32(threadIdx.x) & 63i)>>1u)) + 495i)]>>16u)));
      score[0i] = (score[0i] + dot(left_7, right_7));
      let value : f32 = score[0i];
      score[0i] = (score[0i] + subgroupShuffleXor(value, 1u));
      if ((i32(threadIdx.x) % 2i) == 0i) {
        var condval_5 : f32;
        if (((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) < end)) {
          condval_5 = (score[0i] * 1.250000e-01f);
} else {
          condval_5 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
        probabilities[(i32(threadIdx.x)>>1u)] = condval_5;
      }
      workgroupBarrier();
      probability[0i] = probabilities[(((i32(threadIdx.x)>>6u) * 32i) + (i32(threadIdx.x) & 31i))];
      peak[0i] = probability[0i];
      for (var step : i32 = 0i; step < 5i; step++) {
        let value_1 : f32 = peak[0i];
        peak[0i] = max(peak[0i], subgroupShuffleXor(value_1, u32((32i>>u32((step + 1i))))));
      }
    }
    workgroupBarrier();
    if (((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) < end) {
      let next_max : f32 = max(maximum[0i], peak[0i]);
            var condval_6 : f32;
      if ((maximum[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
        condval_6 = 0.000000e+00f;
} else {
        condval_6 = exp((maximum[0i] - next_max));
}
let correction : f32 = condval_6;
      var condval_7 : f32;
      if (((((((tile>>2u) * 8192i) + (i32(blockIdx.y) * 128i)) + ((tile & 3i) * 32i)) + (i32(threadIdx.x) & 31i)) < end)) {
        condval_7 = exp((probability[0i] - next_max));
} else {
        condval_7 = 0.000000e+00f;
}
      probability[0i] = condval_7;
      norm[0i] = probability[0i];
      second[0i] = 0.000000e+00f;
      for (var step_1 : i32 = 0i; step_1 < 5i; step_1++) {
        let value_2 : f32 = norm[0i];
        norm[0i] = (norm[0i] + subgroupShuffleXor(value_2, u32((32i>>u32((step_1 + 1i))))));
      }
      normalizer[0i] = fma(normalizer[0i], correction, norm[0i]);
      maximum[0i] = next_max;
      if ((i32(threadIdx.x) & 63i) < 32i) {
        probabilities[(((i32(threadIdx.x)>>6u) * 32i) + (i32(threadIdx.x) & 63i))] = probability[0i];
      }
      acc[0i] = (acc[0i] * correction);
      workgroupBarrier();
      let left_8 : vec4<f32> = vec4<f32>(probabilities[((i32(threadIdx.x)>>6u) * 32i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 1i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 2i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 3i)]);
      let right_8 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[((i32(threadIdx.x) & 63i)>>1u)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 32i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 64i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 96i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_8, right_8));
      let left_9 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 4i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 5i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 6i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 7i)]);
      let right_9 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 128i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 160i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 192i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 224i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_9, right_9));
      let left_10 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 8i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 9i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 10i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 11i)]);
      let right_10 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 256i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 288i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 320i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 352i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_10, right_10));
      let left_11 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 12i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 13i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 14i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 15i)]);
      let right_11 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 384i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 416i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 448i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 480i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_11, right_11));
      let left_12 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 16i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 17i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 18i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 19i)]);
      let right_12 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 512i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 544i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 576i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 608i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_12, right_12));
      let left_13 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 20i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 21i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 22i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 23i)]);
      let right_13 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 640i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 672i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 704i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 736i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_13, right_13));
      let left_14 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 24i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 25i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 26i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 27i)]);
      let right_14 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 768i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 800i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 832i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 864i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_14, right_14));
      let left_15 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 28i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 29i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 30i)], probabilities[(((i32(threadIdx.x)>>6u) * 32i) + 31i)]);
      let right_15 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 896i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 928i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 960i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 992i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_15, right_15));
      workgroupBarrier();
    }
  }
  if (0i <= mapping[0i]) {
    out[((((v__1 * 16896i) + ((i32(threadIdx.x)>>6u) * 4224i)) + (i32(blockIdx.y) * 66i)) + (i32(threadIdx.x) & 63i))] = acc[0i];
  }
  if ((i32(threadIdx.x) % 64i) == 0i) {
    if (0i <= mapping[0i]) {
      out[((((v__1 * 16896i) + ((i32(threadIdx.x)>>6u) * 4224i)) + (i32(blockIdx.y) * 66i)) + 64i)] = maximum[0i];
    }
    if (0i <= mapping[0i]) {
      out[((((v__1 * 16896i) + ((i32(threadIdx.x)>>6u) * 4224i)) + (i32(blockIdx.y) * 66i)) + 65i)] = normalizer[0i];
    }
  }
}



fn tensor_unpack_f16(bits:u32)->f32 {
  return unpack2x16float(bits).x;
}

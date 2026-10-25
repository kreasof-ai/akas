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

var<workgroup> keys : array<u32, 2080>;
var<workgroup> values : array<u32, 2048>;
var<workgroup> scores : array<f32, 256>;
var<workgroup> probabilities : array<f32, 256>;
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
  for (var tile : i32 = 0i; tile < (((max((end - (i32(blockIdx.y) * 128i)), 0i) + 16383i)>>14u) * 2i); tile++) {
    if (((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) < end) {
      for (var load : i32 = 0i; load < 8i; load++) {
                var condval_1 : f32;
        if ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) < end)) {
          condval_1 = (f32(kc[(((((((((((v__1 * 131072i) + ((tile>>1u) * 16384i)) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 1048575i)>>17u) * 8388608i) + ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 131071i) * 64i)) + ((i32(threadIdx.x) & 31i) * 2i))]) * ks[(v__1 * 131072i)]);
} else {
          condval_1 = 0.000000e+00f;
}
let a : f32 = condval_1;
                var condval_2 : f32;
        if ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) < end)) {
          condval_2 = (f32(kc[((((((((((((((tile>>1u) * 1048576i) + (i32(blockIdx.y) * 8192i)) + ((tile & 1i) * 4096i)) + (load * 512i)) + (i32(threadIdx.x) * 2i)) + 1i)>>23u) + v__1) & 7i) * 8388608i) + ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 131071i) * 64i)) + ((i32(threadIdx.x) & 31i) * 2i)) + 1i)]) * ks[(v__1 * 131072i)]);
} else {
          condval_2 = 0.000000e+00f;
}
let b : f32 = condval_2;
        keys[((((i32(threadIdx.x) & 31i) * 65i) + (load * 8i)) + (i32(threadIdx.x)>>5u))] = pack2x16float(vec2<f32>(a, b));
                var condval_3 : f32;
        if ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) < end)) {
          condval_3 = (f32(vc[(((((((((((v__1 * 131072i) + ((tile>>1u) * 16384i)) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 1048575i)>>17u) * 8388608i) + ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 131071i) * 64i)) + ((i32(threadIdx.x) & 31i) * 2i))]) * vs[(v__1 * 131072i)]);
} else {
          condval_3 = 0.000000e+00f;
}
let c : f32 = condval_3;
                var condval_4 : f32;
        if ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) < end)) {
          condval_4 = (f32(vc[((((((((((((((tile>>1u) * 1048576i) + (i32(blockIdx.y) * 8192i)) + ((tile & 1i) * 4096i)) + (load * 512i)) + (i32(threadIdx.x) * 2i)) + 1i)>>23u) + v__1) & 7i) * 8388608i) + ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (load * 8i)) + (i32(threadIdx.x)>>5u)) & 131071i) * 64i)) + ((i32(threadIdx.x) & 31i) * 2i)) + 1i)]) * vs[(v__1 * 131072i)]);
} else {
          condval_4 = 0.000000e+00f;
}
let e : f32 = condval_4;
        values[((load * 256i) + i32(threadIdx.x))] = pack2x16float(vec2<f32>(c, e));
      }
      workgroupBarrier();
      for (var score_tile : i32 = 0i; score_tile < 2i; score_tile++) {
        score[0i] = 0.000000e+00f;
        let left : vec4<f32> = vec4<f32>(v[0i], v_1[0i], v_2[0i], v_3[0i]);
        let right : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u))]>>0u)), tensor_unpack_f16((keys[((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u))]>>16u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 65i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 65i)]>>16u)));
        score[0i] = (score[0i] + dot(left, right));
        let left_1 : vec4<f32> = vec4<f32>(v_4[0i], v_5[0i], v_6[0i], v_7[0i]);
        let right_1 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 130i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 130i)]>>16u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 195i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 195i)]>>16u)));
        score[0i] = (score[0i] + dot(left_1, right_1));
        let left_2 : vec4<f32> = vec4<f32>(v_8[0i], v_9[0i], v_10[0i], v_11[0i]);
        let right_2 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 260i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 260i)]>>16u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 325i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 325i)]>>16u)));
        score[0i] = (score[0i] + dot(left_2, right_2));
        let left_3 : vec4<f32> = vec4<f32>(v_12[0i], v_13[0i], v_14[0i], v_15[0i]);
        let right_3 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 390i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 390i)]>>16u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 455i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 455i)]>>16u)));
        score[0i] = (score[0i] + dot(left_3, right_3));
        let left_4 : vec4<f32> = vec4<f32>(v_16[0i], v_17[0i], v_18[0i], v_19[0i]);
        let right_4 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 520i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 520i)]>>16u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 585i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 585i)]>>16u)));
        score[0i] = (score[0i] + dot(left_4, right_4));
        let left_5 : vec4<f32> = vec4<f32>(v_20[0i], v_21[0i], v_22[0i], v_23[0i]);
        let right_5 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 650i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 650i)]>>16u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 715i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 715i)]>>16u)));
        score[0i] = (score[0i] + dot(left_5, right_5));
        let left_6 : vec4<f32> = vec4<f32>(v_24[0i], v_25[0i], v_26[0i], v_27[0i]);
        let right_6 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 780i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 780i)]>>16u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 845i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 845i)]>>16u)));
        score[0i] = (score[0i] + dot(left_6, right_6));
        let left_7 : vec4<f32> = vec4<f32>(v_28[0i], v_29[0i], v_30[0i], v_31[0i]);
        let right_7 : vec4<f32> = vec4<f32>(tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 910i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 910i)]>>16u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 975i)]>>0u)), tensor_unpack_f16((keys[(((((i32(threadIdx.x) & 1i) * 1040i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) + 975i)]>>16u)));
        score[0i] = (score[0i] + dot(left_7, right_7));
        let value : f32 = score[0i];
        score[0i] = (score[0i] + subgroupShuffleXor(value, 1u));
        if ((i32(threadIdx.x) % 2i) == 0i) {
          var condval_5 : f32;
          if ((((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u)) < end)) {
            condval_5 = (score[0i] * 1.250000e-01f);
} else {
            condval_5 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
          scores[((((i32(threadIdx.x)>>6u) * 64i) + (score_tile * 32i)) + ((i32(threadIdx.x) & 63i)>>1u))] = condval_5;
        }
      }
      workgroupBarrier();
      probability[0i] = scores[i32(threadIdx.x)];
      peak[0i] = probability[0i];
      peak[0i] = max(peak[0i], scores[(((i32(threadIdx.x)>>6u) * 64i) + ((i32(threadIdx.x) & 63i) ^ 32i))]);
      for (var step : i32 = 0i; step < 5i; step++) {
        let value_1 : f32 = peak[0i];
        peak[0i] = max(peak[0i], subgroupShuffleXor(value_1, u32((32i>>u32((step + 1i))))));
      }
      let next_max : f32 = max(maximum[0i], peak[0i]);
            var condval_6 : f32;
      if ((maximum[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
        condval_6 = 0.000000e+00f;
} else {
        condval_6 = exp((maximum[0i] - next_max));
}
let correction : f32 = condval_6;
      var condval_7 : f32;
      if (((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (i32(threadIdx.x) & 63i)) < end)) {
        condval_7 = exp((probability[0i] - next_max));
} else {
        condval_7 = 0.000000e+00f;
}
      probability[0i] = condval_7;
      norm[0i] = probability[0i];
      second[0i] = 0.000000e+00f;
      var condval_8 : f32;
      if (((((((tile>>1u) * 16384i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + ((i32(threadIdx.x) & 63i) ^ 32i)) < end)) {
        condval_8 = exp((scores[(((i32(threadIdx.x)>>6u) * 64i) + ((i32(threadIdx.x) & 63i) ^ 32i))] - next_max));
} else {
        condval_8 = 0.000000e+00f;
}
      norm[0i] = (norm[0i] + condval_8);
      for (var step_1 : i32 = 0i; step_1 < 5i; step_1++) {
        let value_2 : f32 = norm[0i];
        norm[0i] = (norm[0i] + subgroupShuffleXor(value_2, u32((32i>>u32((step_1 + 1i))))));
      }
      normalizer[0i] = fma(normalizer[0i], correction, norm[0i]);
      maximum[0i] = next_max;
      probabilities[i32(threadIdx.x)] = probability[0i];
      acc[0i] = (acc[0i] * correction);
      workgroupBarrier();
      let left_8 : vec4<f32> = vec4<f32>(probabilities[((i32(threadIdx.x)>>6u) * 64i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 1i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 2i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 3i)]);
      let right_8 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[((i32(threadIdx.x) & 63i)>>1u)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 32i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 64i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 96i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_8, right_8));
      let left_9 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 4i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 5i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 6i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 7i)]);
      let right_9 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 128i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 160i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 192i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 224i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_9, right_9));
      let left_10 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 8i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 9i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 10i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 11i)]);
      let right_10 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 256i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 288i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 320i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 352i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_10, right_10));
      let left_11 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 12i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 13i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 14i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 15i)]);
      let right_11 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 384i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 416i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 448i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 480i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_11, right_11));
      let left_12 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 16i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 17i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 18i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 19i)]);
      let right_12 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 512i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 544i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 576i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 608i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_12, right_12));
      let left_13 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 20i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 21i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 22i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 23i)]);
      let right_13 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 640i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 672i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 704i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 736i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_13, right_13));
      let left_14 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 24i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 25i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 26i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 27i)]);
      let right_14 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 768i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 800i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 832i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 864i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_14, right_14));
      let left_15 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 28i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 29i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 30i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 31i)]);
      let right_15 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 896i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 928i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 960i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 992i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_15, right_15));
      let left_16 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 32i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 33i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 34i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 35i)]);
      let right_16 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1024i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1056i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1088i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1120i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_16, right_16));
      let left_17 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 36i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 37i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 38i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 39i)]);
      let right_17 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1152i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1184i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1216i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1248i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_17, right_17));
      let left_18 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 40i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 41i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 42i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 43i)]);
      let right_18 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1280i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1312i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1344i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1376i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_18, right_18));
      let left_19 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 44i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 45i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 46i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 47i)]);
      let right_19 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1408i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1440i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1472i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1504i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_19, right_19));
      let left_20 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 48i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 49i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 50i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 51i)]);
      let right_20 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1536i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1568i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1600i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1632i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_20, right_20));
      let left_21 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 52i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 53i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 54i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 55i)]);
      let right_21 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1664i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1696i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1728i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1760i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_21, right_21));
      let left_22 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 56i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 57i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 58i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 59i)]);
      let right_22 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1792i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1824i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1856i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1888i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_22, right_22));
      let left_23 : vec4<f32> = vec4<f32>(probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 60i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 61i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 62i)], probabilities[(((i32(threadIdx.x)>>6u) * 64i) + 63i)]);
      let right_23 : vec4<f32> = vec4<f32>(tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1920i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1952i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 1984i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))), tensor_unpack_f16((values[(((i32(threadIdx.x) & 63i)>>1u) + 2016i)]>>u32(((i32(threadIdx.x) & 1i) * 16i)))));
      acc[0i] = (acc[0i] + dot(left_23, right_23));
      workgroupBarrier();
    }
  }
  if (0i <= mapping[0i]) {
    out[((((v__1 * 33792i) + ((i32(threadIdx.x)>>6u) * 8448i)) + (i32(blockIdx.y) * 66i)) + (i32(threadIdx.x) & 63i))] = acc[0i];
  }
  if ((i32(threadIdx.x) % 64i) == 0i) {
    if (0i <= mapping[0i]) {
      out[((((v__1 * 33792i) + ((i32(threadIdx.x)>>6u) * 8448i)) + (i32(blockIdx.y) * 66i)) + 64i)] = maximum[0i];
    }
    if (0i <= mapping[0i]) {
      out[((((v__1 * 33792i) + ((i32(threadIdx.x)>>6u) * 8448i)) + (i32(blockIdx.y) * 66i)) + 65i)] = normalizer[0i];
    }
  }
}



fn tensor_unpack_f16(bits:u32)->f32 {
  return unpack2x16float(bits).x;
}

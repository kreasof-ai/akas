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

var<workgroup> cache_tile : array<f16, 4160>;
var<workgroup> scores : array<f32, 2052>;
var<workgroup> reductions : array<f32, 8>;
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
  var maximum : array<f32, 1>;
  var normalizer : array<f32, 1>;
  var result : array<f32, 1>;
  var score : array<f32, 1>;
    var condval : i32;
  if ((enabled[0i] != 0i)) {
    condval = min((positions[0i] + 1i), 131072i);
} else {
    condval = 0i;
}
let length : i32 = condval;
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
  let _tmp : f32 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  maximum[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  normalizer[0i] = 0.000000e+00f;
  result[0i] = 0.000000e+00f;
  if ((i32(blockIdx.y) * 128i) < length) {
    v[0i] = q[(((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i))];
    v_1[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 1i)];
    v_2[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 2i)];
    v_3[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 3i)];
    v_4[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 4i)];
    v_5[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 5i)];
    v_6[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 6i)];
    v_7[0i] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>6u) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 7i)];
    for (var tile : i32 = 0i; tile < (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i); tile++) {
      for (var i : i32 = 0i; i < 16i; i++) {
        var condval_1 : f16;
        if ((((((((tile>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + ((i>>3u) * 32i)) + (i32(threadIdx.x)>>3u)) < length)) {
          condval_1 = kc[((((v__1 * 8388608i) + (min(((((((tile>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + ((i>>3u) * 32i)) + (i32(threadIdx.x)>>3u)), 131071i) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + (i & 7i))];
} else {
          condval_1 = 0.000000e+00h;
}
        cache_tile[(((((i>>3u) * 2080i) + ((i32(threadIdx.x)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + (i & 7i))] = condval_1;
      }
      workgroupBarrier();
      for (var item : i32 = 0i; item < 8i; item++) {
        score[0i] = 0.000000e+00f;
        let left : vec4<f32> = vec4<f32>(v[0i], v_1[0i], v_2[0i], v_3[0i]);
        let right : vec4<f32> = vec4<f32>(f32(cache_tile[(((item * 520i) + (((i32(threadIdx.x) & 63i)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i))]), f32(cache_tile[((((item * 520i) + (((i32(threadIdx.x) & 63i)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 1i)]), f32(cache_tile[((((item * 520i) + (((i32(threadIdx.x) & 63i)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 2i)]), f32(cache_tile[((((item * 520i) + (((i32(threadIdx.x) & 63i)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 3i)]));
        score[0i] = (score[0i] + dot(left, right));
        let left_1 : vec4<f32> = vec4<f32>(v_4[0i], v_5[0i], v_6[0i], v_7[0i]);
        let right_1 : vec4<f32> = vec4<f32>(f32(cache_tile[((((item * 520i) + (((i32(threadIdx.x) & 63i)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 4i)]), f32(cache_tile[((((item * 520i) + (((i32(threadIdx.x) & 63i)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 5i)]), f32(cache_tile[((((item * 520i) + (((i32(threadIdx.x) & 63i)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 6i)]), f32(cache_tile[((((item * 520i) + (((i32(threadIdx.x) & 63i)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + 7i)]));
        score[0i] = (score[0i] + dot(left_1, right_1));
        for (var step : i32 = 0i; step < 3i; step++) {
          let value : f32 = score[0i];
          score[0i] = (score[0i] + subgroupShuffleXor(value, u32((8i>>u32((step + 1i))))));
        }
        var condval_2 : f32;
        if ((((((((tile>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile & 1i) * 64i)) + (item * 8i)) + ((i32(threadIdx.x) & 63i)>>3u)) < length)) {
          condval_2 = ((score[0i] * 1.250000e-01f) * ks[(v__1 * 131072i)]);
} else {
          condval_2 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
        score[0i] = condval_2;
        maximum[0i] = max(maximum[0i], score[0i]);
        if ((i32(threadIdx.x) % 8i) == 0i) {
          scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile * 64i)) + (item * 8i)) + ((i32(threadIdx.x) & 63i)>>3u))] = score[0i];
        }
      }
      workgroupBarrier();
    }
    for (var step_1 : i32 = 0i; step_1 < 5i; step_1++) {
      let value_1 : f32 = maximum[0i];
      maximum[0i] = max(maximum[0i], subgroupShuffleXor(value_1, u32((16i>>u32(step_1)))));
    }
    if ((i32(threadIdx.x) % 32i) == 0i) {
      reductions[(i32(threadIdx.x)>>5u)] = maximum[0i];
    }
    workgroupBarrier();
    maximum[0i] = max(reductions[((i32(threadIdx.x)>>6u) * 2i)], reductions[(((i32(threadIdx.x)>>6u) * 2i) + 1i)]);
    for (var tile_1 : i32 = 0i; tile_1 < (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i); tile_1++) {
      let probability : f32 = exp((scores[((((i32(threadIdx.x)>>6u) * 513i) + (tile_1 * 64i)) + (i32(threadIdx.x) & 63i))] - maximum[0i]));
      scores[((((i32(threadIdx.x)>>6u) * 513i) + (tile_1 * 64i)) + (i32(threadIdx.x) & 63i))] = probability;
      normalizer[0i] = (normalizer[0i] + probability);
    }
    for (var step_2 : i32 = 0i; step_2 < 5i; step_2++) {
      let value_2 : f32 = normalizer[0i];
      normalizer[0i] = (normalizer[0i] + subgroupShuffleXor(value_2, u32((16i>>u32(step_2)))));
    }
  }
  workgroupBarrier();
  if ((i32(blockIdx.y) * 128i) < length) {
    if ((i32(threadIdx.x) % 32i) == 0i) {
      reductions[(i32(threadIdx.x)>>5u)] = normalizer[0i];
    }
    workgroupBarrier();
    normalizer[0i] = (reductions[((i32(threadIdx.x)>>6u) * 2i)] + reductions[(((i32(threadIdx.x)>>6u) * 2i) + 1i)]);
    for (var tile_2 : i32 = 0i; tile_2 < (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i); tile_2++) {
      for (var i_1 : i32 = 0i; i_1 < 16i; i_1++) {
        var condval_3 : f16;
        if ((((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + ((i_1>>3u) * 32i)) + (i32(threadIdx.x)>>3u)) < length)) {
          condval_3 = vc[((((v__1 * 8388608i) + (min(((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + ((i_1>>3u) * 32i)) + (i32(threadIdx.x)>>3u)), 131071i) * 64i)) + ((i32(threadIdx.x) & 7i) * 8i)) + (i_1 & 7i))];
} else {
          condval_3 = 0.000000e+00h;
}
        cache_tile[(((((i_1>>3u) * 2080i) + ((i32(threadIdx.x)>>3u) * 65i)) + ((i32(threadIdx.x) & 7i) * 8i)) + (i_1 & 7i))] = condval_3;
      }
      workgroupBarrier();
      for (var item_1 : i32 = 0i; item_1 < 4i; item_1++) {
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & ((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) < length)) {
          v_8[0i] = fma(scores[((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i))], f32(cache_tile[((item_1 * 1040i) + (i32(threadIdx.x) & 63i))]), v_8[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 1i) < length)) {
          v_9[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 1i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 65i)]), v_9[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 2i) < length)) {
          v_10[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 2i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 130i)]), v_10[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 3i) < length)) {
          v_11[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 3i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 195i)]), v_11[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 4i) < length)) {
          v_8[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 4i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 260i)]), v_8[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 5i) < length)) {
          v_9[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 5i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 325i)]), v_9[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 6i) < length)) {
          v_10[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 6i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 390i)]), v_10[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 7i) < length)) {
          v_11[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 7i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 455i)]), v_11[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 8i) < length)) {
          v_8[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 8i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 520i)]), v_8[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 9i) < length)) {
          v_9[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 9i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 585i)]), v_9[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 10i) < length)) {
          v_10[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 10i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 650i)]), v_10[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 11i) < length)) {
          v_11[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 11i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 715i)]), v_11[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 12i) < length)) {
          v_8[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 12i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 780i)]), v_8[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 13i) < length)) {
          v_9[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 13i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 845i)]), v_9[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 14i) < length)) {
          v_10[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 14i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 910i)]), v_10[0i]);
        }
        if (((tile_2 - (((max((length - (i32(blockIdx.y) * 128i)), 0i) + 32767i)>>15u) * 2i)) < 0i) & (((((((tile_2>>1u) * 32768i) + (i32(blockIdx.y) * 128i)) + ((tile_2 & 1i) * 64i)) + (item_1 * 16i)) + 15i) < length)) {
          v_11[0i] = fma(scores[(((((i32(threadIdx.x)>>6u) * 513i) + (tile_2 * 64i)) + (item_1 * 16i)) + 15i)], f32(cache_tile[(((item_1 * 1040i) + (i32(threadIdx.x) & 63i)) + 975i)]), v_11[0i]);
        }
      }
      workgroupBarrier();
    }
    let other : f32 = v_8[0i];
    result[0i] = (result[0i] + other);
    let other_1 : f32 = v_9[0i];
    result[0i] = (result[0i] + other_1);
    let other_2 : f32 = v_10[0i];
    result[0i] = (result[0i] + other_2);
    let other_3 : f32 = v_11[0i];
    result[0i] = (result[0i] + other_3);
    result[0i] = (result[0i] * vs[(v__1 * 131072i)]);
  }
  if (0i <= mapping[0i]) {
    out[((((v__1 * 67584i) + ((i32(threadIdx.x)>>6u) * 16896i)) + (i32(blockIdx.y) * 66i)) + (i32(threadIdx.x) & 63i))] = result[0i];
  }
  if ((i32(threadIdx.x) % 64i) == 0i) {
    if (0i <= mapping[0i]) {
      out[((((v__1 * 67584i) + ((i32(threadIdx.x)>>6u) * 16896i)) + (i32(blockIdx.y) * 66i)) + 64i)] = maximum[0i];
    }
    if (0i <= mapping[0i]) {
      out[((((v__1 * 67584i) + ((i32(threadIdx.x)>>6u) * 16896i)) + (i32(blockIdx.y) * 66i)) + 65i)] = normalizer[0i];
    }
  }
}



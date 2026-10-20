// Function: kernel_kernel
//----------------------------------------
// Function: kernel_kernel
//----------------------------------------
enable f16;

@group(0) @binding(0) var<storage, read> enabled : array<i32>;
@group(0) @binding(1) var<storage, read> kc : array<f16>;
@group(0) @binding(2) var<storage, read> ks : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;
@group(0) @binding(4) var<storage, read> positions : array<i32>;
@group(0) @binding(5) var<storage, read> q : array<f32>;
@group(0) @binding(6) var<storage, read> vc : array<f16>;
@group(0) @binding(7) var<storage, read> vs : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(8) var<uniform> podArgs : PODArgs;

var<workgroup> queries : array<f32, 2048>;
var<workgroup> keys : array<f16, 2112>;
var<workgroup> values : array<f16, 2048>;
var<workgroup> scores : array<f32, 1024>;
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
  var v_32 : array<f32, 1>;
  var v_33 : array<f32, 1>;
  var v_34 : array<f32, 1>;
  var v_35 : array<f32, 1>;
  var v_36 : array<f32, 1>;
  var v_37 : array<f32, 1>;
  var v_38 : array<f32, 1>;
  var v_39 : array<f32, 1>;
  var v_40 : array<f32, 1>;
  var v_41 : array<f32, 1>;
  var v_42 : array<f32, 1>;
  var v_43 : array<f32, 1>;
  var v_44 : array<f32, 1>;
  var v_45 : array<f32, 1>;
  var v_46 : array<f32, 1>;
  var v_47 : array<f32, 1>;
  var right : array<vec4<f32>, 1>;
  var peak : array<f32, 1>;
  var next_max : array<f32, 1>;
  var probability : array<f32, 1>;
  var norm : array<f32, 1>;
  var peak_1 : array<f32, 1>;
  var next_max_1 : array<f32, 1>;
  var probability_1 : array<f32, 1>;
  var norm_1 : array<f32, 1>;
  var peak_2 : array<f32, 1>;
  var next_max_2 : array<f32, 1>;
  var probability_2 : array<f32, 1>;
  var norm_2 : array<f32, 1>;
  var peak_3 : array<f32, 1>;
  var next_max_3 : array<f32, 1>;
  var probability_3 : array<f32, 1>;
  var norm_3 : array<f32, 1>;
  var peak_4 : array<f32, 1>;
  var next_max_4 : array<f32, 1>;
  var probability_4 : array<f32, 1>;
  var norm_4 : array<f32, 1>;
  var peak_5 : array<f32, 1>;
  var next_max_5 : array<f32, 1>;
  var probability_5 : array<f32, 1>;
  var norm_5 : array<f32, 1>;
  var peak_6 : array<f32, 1>;
  var next_max_6 : array<f32, 1>;
  var probability_6 : array<f32, 1>;
  var norm_6 : array<f32, 1>;
  var peak_7 : array<f32, 1>;
  var next_max_7 : array<f32, 1>;
  var probability_7 : array<f32, 1>;
  var norm_7 : array<f32, 1>;
  var right0 : array<vec4<f32>, 1>;
  var right1 : array<vec4<f32>, 1>;
    var condval : i32;
  if ((enabled[(i32(blockIdx.y) * 8i)] != 0i)) {
    condval = positions[(i32(blockIdx.y) * 8i)];
} else {
    condval = -1i;
}
  var condval_1 : i32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 1i)] != 0i)) {
    condval_1 = positions[((i32(blockIdx.y) * 8i) + 1i)];
} else {
    condval_1 = -1i;
}
  var condval_2 : i32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 2i)] != 0i)) {
    condval_2 = positions[((i32(blockIdx.y) * 8i) + 2i)];
} else {
    condval_2 = -1i;
}
  var condval_3 : i32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 3i)] != 0i)) {
    condval_3 = positions[((i32(blockIdx.y) * 8i) + 3i)];
} else {
    condval_3 = -1i;
}
  var condval_4 : i32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 4i)] != 0i)) {
    condval_4 = positions[((i32(blockIdx.y) * 8i) + 4i)];
} else {
    condval_4 = -1i;
}
  var condval_5 : i32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 5i)] != 0i)) {
    condval_5 = positions[((i32(blockIdx.y) * 8i) + 5i)];
} else {
    condval_5 = -1i;
}
  var condval_6 : i32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 6i)] != 0i)) {
    condval_6 = positions[((i32(blockIdx.y) * 8i) + 6i)];
} else {
    condval_6 = -1i;
}
  var condval_7 : i32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 7i)] != 0i)) {
    condval_7 = positions[((i32(blockIdx.y) * 8i) + 7i)];
} else {
    condval_7 = -1i;
}
let length : i32 = min((max(max(max(max(max(max(max(max(-1i, condval), condval_1), condval_2), condval_3), condval_4), condval_5), condval_6), condval_7) + 1i), 8192i);
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
  v_32[0i] = 0.000000e+00f;
  v_33[0i] = 0.000000e+00f;
  v_34[0i] = 0.000000e+00f;
  v_35[0i] = 0.000000e+00f;
  v_36[0i] = 0.000000e+00f;
  v_37[0i] = 0.000000e+00f;
  v_38[0i] = 0.000000e+00f;
  v_39[0i] = 0.000000e+00f;
  v[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  v_1[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  v_2[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  v_3[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  v_4[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  v_5[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  v_6[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  v_7[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  for (var i : i32 = 0i; i < 4i; i++) {
    for (var vec : i32 = 0i; vec < 4i; vec++) {
      queries[(((i * 512i) + (i32(threadIdx.x) * 4i)) + vec)] = q[((((((i32(blockIdx.y) * 16384i) + (i * 4096i)) + ((i32(threadIdx.x)>>6u) * 2048i)) + (v__1 * 256i)) + ((i32(threadIdx.x) & 63i) * 4i)) + vec)];
    }
  }
  workgroupBarrier();
  for (var tile : i32 = 0i; tile < ((length + 31i) / 32i); tile++) {
    for (var i_1 : i32 = 0i; i_1 < 16i; i_1++) {
      var condval_8 : f32;
      if (((((tile * 32i) + (i_1 * 2i)) + (i32(threadIdx.x)>>6u)) < length)) {
        var condval_9 : f16;
        if ((tile < 256i)) {
          condval_9 = kc[((((v__1 * 524288i) + (tile * 2048i)) + (i_1 * 128i)) + i32(threadIdx.x))];
} else {
          condval_9 = 0.000000e+00h;
}
        var condval_10 : f32;
        if ((tile < 256i)) {
          condval_10 = ks[((((v__1 * 8192i) + (tile * 32i)) + (i_1 * 2i)) + (i32(threadIdx.x)>>6u))];
} else {
          condval_10 = 0.000000e+00f;
}
        condval_8 = (f32(condval_9) * condval_10);
} else {
        condval_8 = 0.000000e+00f;
}
      keys[((((i32(threadIdx.x) & 63i) * 33i) + (i_1 * 2i)) + (i32(threadIdx.x)>>6u))] = f16(condval_8);
      var condval_11 : f32;
      if (((((tile * 32i) + (i_1 * 2i)) + (i32(threadIdx.x)>>6u)) < length)) {
        var condval_12 : f16;
        if ((tile < 256i)) {
          condval_12 = vc[((((v__1 * 524288i) + (tile * 2048i)) + (i_1 * 128i)) + i32(threadIdx.x))];
} else {
          condval_12 = 0.000000e+00h;
}
        var condval_13 : f32;
        if ((tile < 256i)) {
          condval_13 = vs[((((v__1 * 8192i) + (tile * 32i)) + (i_1 * 2i)) + (i32(threadIdx.x)>>6u))];
} else {
          condval_13 = 0.000000e+00f;
}
        condval_11 = (f32(condval_12) * condval_13);
} else {
        condval_11 = 0.000000e+00f;
}
      values[((i_1 * 128i) + i32(threadIdx.x))] = f16(condval_11);
    }
    workgroupBarrier();
    v_40[0i] = 0.000000e+00f;
    v_41[0i] = 0.000000e+00f;
    v_42[0i] = 0.000000e+00f;
    v_43[0i] = 0.000000e+00f;
    v_44[0i] = 0.000000e+00f;
    v_45[0i] = 0.000000e+00f;
    v_46[0i] = 0.000000e+00f;
    v_47[0i] = 0.000000e+00f;
    for (var channel : i32 = 0i; channel < 16i; channel++) {
      right[0i] = vec4<f32>(f32(keys[((channel * 132i) + (i32(threadIdx.x) & 31i))]), f32(keys[(((channel * 132i) + (i32(threadIdx.x) & 31i)) + 33i)]), f32(keys[(((channel * 132i) + (i32(threadIdx.x) & 31i)) + 66i)]), f32(keys[(((channel * 132i) + (i32(threadIdx.x) & 31i)) + 99i)]));
      let right_1 : vec4<f32> = right[0i];
      let left : vec4<f32> = vec4<f32>(queries[(((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i))], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 2i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 3i)]);
      v_40[0i] = (v_40[0i] + dot(left, right_1));
      let right_2 : vec4<f32> = right[0i];
      let left_1 : vec4<f32> = vec4<f32>(queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 256i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 257i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 258i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 259i)]);
      v_41[0i] = (v_41[0i] + dot(left_1, right_2));
      let right_3 : vec4<f32> = right[0i];
      let left_2 : vec4<f32> = vec4<f32>(queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 512i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 513i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 514i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 515i)]);
      v_42[0i] = (v_42[0i] + dot(left_2, right_3));
      let right_4 : vec4<f32> = right[0i];
      let left_3 : vec4<f32> = vec4<f32>(queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 768i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 769i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 770i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 771i)]);
      v_43[0i] = (v_43[0i] + dot(left_3, right_4));
      let right_5 : vec4<f32> = right[0i];
      let left_4 : vec4<f32> = vec4<f32>(queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1024i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1025i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1026i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1027i)]);
      v_44[0i] = (v_44[0i] + dot(left_4, right_5));
      let right_6 : vec4<f32> = right[0i];
      let left_5 : vec4<f32> = vec4<f32>(queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1280i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1281i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1282i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1283i)]);
      v_45[0i] = (v_45[0i] + dot(left_5, right_6));
      let right_7 : vec4<f32> = right[0i];
      let left_6 : vec4<f32> = vec4<f32>(queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1536i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1537i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1538i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1539i)]);
      v_46[0i] = (v_46[0i] + dot(left_6, right_7));
      let right_8 : vec4<f32> = right[0i];
      let left_7 : vec4<f32> = vec4<f32>(queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1792i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1793i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1794i)], queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + 1795i)]);
      v_47[0i] = (v_47[0i] + dot(left_7, right_8));
    }
    let valid : bool = (((enabled[(i32(blockIdx.y) * 8i)] != 0i) && (tile < 256i)) && (((tile * 32i) + (i32(threadIdx.x) & 31i)) <= positions[(i32(blockIdx.y) * 8i)]));
    var condval_14 : f32;
    if (valid) {
      condval_14 = (v_40[0i] * 1.250000e-01f);
} else {
      condval_14 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    v_40[0i] = condval_14;
    peak[0i] = v_40[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step : i32 = 0i; step < 5i; step++) {
        let other : f32 = subgroupShuffleXor(peak[0i], u32((16i>>u32(step))));
        peak[0i] = max(peak[0i], other);
      }
    } else {
      scores[i32(threadIdx.x)] = peak[0i];
      workgroupBarrier();
      for (var step_1 : i32 = 0i; step_1 < 5i; step_1++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_1))) {
          scores[i32(threadIdx.x)] = max(scores[i32(threadIdx.x)], scores[(i32(threadIdx.x) + (16i>>u32(step_1)))]);
        }
        workgroupBarrier();
      }
      peak[0i] = scores[((i32(threadIdx.x)>>5u) * 32i)];
      workgroupBarrier();
    }
    next_max[0i] = max(v[0i], peak[0i]);
    var condval_15 : f32;
    if ((v[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
      condval_15 = 0.000000e+00f;
} else {
      condval_15 = exp((v[0i] - next_max[0i]));
}
    v_16[0i] = condval_15;
    var condval_16 : f32;
    if (valid) {
      condval_16 = exp((v_40[0i] - next_max[0i]));
} else {
      condval_16 = 0.000000e+00f;
}
    probability[0i] = condval_16;
    norm[0i] = probability[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_2 : i32 = 0i; step_2 < 5i; step_2++) {
        let other_1 : f32 = subgroupShuffleXor(norm[0i], u32((16i>>u32(step_2))));
        norm[0i] = (norm[0i] + other_1);
      }
    } else {
      scores[i32(threadIdx.x)] = norm[0i];
      workgroupBarrier();
      for (var step_3 : i32 = 0i; step_3 < 5i; step_3++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_3))) {
          scores[i32(threadIdx.x)] = (scores[i32(threadIdx.x)] + scores[(i32(threadIdx.x) + (16i>>u32(step_3)))]);
        }
        workgroupBarrier();
      }
      norm[0i] = scores[((i32(threadIdx.x)>>5u) * 32i)];
      workgroupBarrier();
    }
    v_8[0i] = fma(v_8[0i], v_16[0i], norm[0i]);
    v[0i] = next_max[0i];
    scores[i32(threadIdx.x)] = probability[0i];
    let valid_1 : bool = (((enabled[((i32(blockIdx.y) * 8i) + 1i)] != 0i) && (tile < 256i)) && (((tile * 32i) + (i32(threadIdx.x) & 31i)) <= positions[((i32(blockIdx.y) * 8i) + 1i)]));
    var condval_17 : f32;
    if (valid_1) {
      condval_17 = (v_41[0i] * 1.250000e-01f);
} else {
      condval_17 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    v_41[0i] = condval_17;
    peak_1[0i] = v_41[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_4 : i32 = 0i; step_4 < 5i; step_4++) {
        let other_2 : f32 = subgroupShuffleXor(peak_1[0i], u32((16i>>u32(step_4))));
        peak_1[0i] = max(peak_1[0i], other_2);
      }
    } else {
      scores[(i32(threadIdx.x) + 128i)] = peak_1[0i];
      workgroupBarrier();
      for (var step_5 : i32 = 0i; step_5 < 5i; step_5++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_5))) {
          scores[(i32(threadIdx.x) + 128i)] = max(scores[(i32(threadIdx.x) + 128i)], scores[((i32(threadIdx.x) + (16i>>u32(step_5))) + 128i)]);
        }
        workgroupBarrier();
      }
      peak_1[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 128i)];
      workgroupBarrier();
    }
    next_max_1[0i] = max(v_1[0i], peak_1[0i]);
    var condval_18 : f32;
    if ((v_1[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
      condval_18 = 0.000000e+00f;
} else {
      condval_18 = exp((v_1[0i] - next_max_1[0i]));
}
    v_17[0i] = condval_18;
    var condval_19 : f32;
    if (valid_1) {
      condval_19 = exp((v_41[0i] - next_max_1[0i]));
} else {
      condval_19 = 0.000000e+00f;
}
    probability_1[0i] = condval_19;
    norm_1[0i] = probability_1[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_6 : i32 = 0i; step_6 < 5i; step_6++) {
        let other_3 : f32 = subgroupShuffleXor(norm_1[0i], u32((16i>>u32(step_6))));
        norm_1[0i] = (norm_1[0i] + other_3);
      }
    } else {
      scores[(i32(threadIdx.x) + 128i)] = norm_1[0i];
      workgroupBarrier();
      for (var step_7 : i32 = 0i; step_7 < 5i; step_7++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_7))) {
          scores[(i32(threadIdx.x) + 128i)] = (scores[(i32(threadIdx.x) + 128i)] + scores[((i32(threadIdx.x) + (16i>>u32(step_7))) + 128i)]);
        }
        workgroupBarrier();
      }
      norm_1[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 128i)];
      workgroupBarrier();
    }
    v_9[0i] = fma(v_9[0i], v_17[0i], norm_1[0i]);
    v_1[0i] = next_max_1[0i];
    scores[(i32(threadIdx.x) + 128i)] = probability_1[0i];
    let valid_2 : bool = (((enabled[((i32(blockIdx.y) * 8i) + 2i)] != 0i) && (tile < 256i)) && (((tile * 32i) + (i32(threadIdx.x) & 31i)) <= positions[((i32(blockIdx.y) * 8i) + 2i)]));
    var condval_20 : f32;
    if (valid_2) {
      condval_20 = (v_42[0i] * 1.250000e-01f);
} else {
      condval_20 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    v_42[0i] = condval_20;
    peak_2[0i] = v_42[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_8 : i32 = 0i; step_8 < 5i; step_8++) {
        let other_4 : f32 = subgroupShuffleXor(peak_2[0i], u32((16i>>u32(step_8))));
        peak_2[0i] = max(peak_2[0i], other_4);
      }
    } else {
      scores[(i32(threadIdx.x) + 256i)] = peak_2[0i];
      workgroupBarrier();
      for (var step_9 : i32 = 0i; step_9 < 5i; step_9++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_9))) {
          scores[(i32(threadIdx.x) + 256i)] = max(scores[(i32(threadIdx.x) + 256i)], scores[((i32(threadIdx.x) + (16i>>u32(step_9))) + 256i)]);
        }
        workgroupBarrier();
      }
      peak_2[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 256i)];
      workgroupBarrier();
    }
    next_max_2[0i] = max(v_2[0i], peak_2[0i]);
    var condval_21 : f32;
    if ((v_2[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
      condval_21 = 0.000000e+00f;
} else {
      condval_21 = exp((v_2[0i] - next_max_2[0i]));
}
    v_18[0i] = condval_21;
    var condval_22 : f32;
    if (valid_2) {
      condval_22 = exp((v_42[0i] - next_max_2[0i]));
} else {
      condval_22 = 0.000000e+00f;
}
    probability_2[0i] = condval_22;
    norm_2[0i] = probability_2[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_10 : i32 = 0i; step_10 < 5i; step_10++) {
        let other_5 : f32 = subgroupShuffleXor(norm_2[0i], u32((16i>>u32(step_10))));
        norm_2[0i] = (norm_2[0i] + other_5);
      }
    } else {
      scores[(i32(threadIdx.x) + 256i)] = norm_2[0i];
      workgroupBarrier();
      for (var step_11 : i32 = 0i; step_11 < 5i; step_11++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_11))) {
          scores[(i32(threadIdx.x) + 256i)] = (scores[(i32(threadIdx.x) + 256i)] + scores[((i32(threadIdx.x) + (16i>>u32(step_11))) + 256i)]);
        }
        workgroupBarrier();
      }
      norm_2[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 256i)];
      workgroupBarrier();
    }
    v_10[0i] = fma(v_10[0i], v_18[0i], norm_2[0i]);
    v_2[0i] = next_max_2[0i];
    scores[(i32(threadIdx.x) + 256i)] = probability_2[0i];
    let valid_3 : bool = (((enabled[((i32(blockIdx.y) * 8i) + 3i)] != 0i) && (tile < 256i)) && (((tile * 32i) + (i32(threadIdx.x) & 31i)) <= positions[((i32(blockIdx.y) * 8i) + 3i)]));
    var condval_23 : f32;
    if (valid_3) {
      condval_23 = (v_43[0i] * 1.250000e-01f);
} else {
      condval_23 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    v_43[0i] = condval_23;
    peak_3[0i] = v_43[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_12 : i32 = 0i; step_12 < 5i; step_12++) {
        let other_6 : f32 = subgroupShuffleXor(peak_3[0i], u32((16i>>u32(step_12))));
        peak_3[0i] = max(peak_3[0i], other_6);
      }
    } else {
      scores[(i32(threadIdx.x) + 384i)] = peak_3[0i];
      workgroupBarrier();
      for (var step_13 : i32 = 0i; step_13 < 5i; step_13++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_13))) {
          scores[(i32(threadIdx.x) + 384i)] = max(scores[(i32(threadIdx.x) + 384i)], scores[((i32(threadIdx.x) + (16i>>u32(step_13))) + 384i)]);
        }
        workgroupBarrier();
      }
      peak_3[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 384i)];
      workgroupBarrier();
    }
    next_max_3[0i] = max(v_3[0i], peak_3[0i]);
    var condval_24 : f32;
    if ((v_3[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
      condval_24 = 0.000000e+00f;
} else {
      condval_24 = exp((v_3[0i] - next_max_3[0i]));
}
    v_19[0i] = condval_24;
    var condval_25 : f32;
    if (valid_3) {
      condval_25 = exp((v_43[0i] - next_max_3[0i]));
} else {
      condval_25 = 0.000000e+00f;
}
    probability_3[0i] = condval_25;
    norm_3[0i] = probability_3[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_14 : i32 = 0i; step_14 < 5i; step_14++) {
        let other_7 : f32 = subgroupShuffleXor(norm_3[0i], u32((16i>>u32(step_14))));
        norm_3[0i] = (norm_3[0i] + other_7);
      }
    } else {
      scores[(i32(threadIdx.x) + 384i)] = norm_3[0i];
      workgroupBarrier();
      for (var step_15 : i32 = 0i; step_15 < 5i; step_15++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_15))) {
          scores[(i32(threadIdx.x) + 384i)] = (scores[(i32(threadIdx.x) + 384i)] + scores[((i32(threadIdx.x) + (16i>>u32(step_15))) + 384i)]);
        }
        workgroupBarrier();
      }
      norm_3[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 384i)];
      workgroupBarrier();
    }
    v_11[0i] = fma(v_11[0i], v_19[0i], norm_3[0i]);
    v_3[0i] = next_max_3[0i];
    scores[(i32(threadIdx.x) + 384i)] = probability_3[0i];
    let valid_4 : bool = (((enabled[((i32(blockIdx.y) * 8i) + 4i)] != 0i) && (tile < 256i)) && (((tile * 32i) + (i32(threadIdx.x) & 31i)) <= positions[((i32(blockIdx.y) * 8i) + 4i)]));
    var condval_26 : f32;
    if (valid_4) {
      condval_26 = (v_44[0i] * 1.250000e-01f);
} else {
      condval_26 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    v_44[0i] = condval_26;
    peak_4[0i] = v_44[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_16 : i32 = 0i; step_16 < 5i; step_16++) {
        let other_8 : f32 = subgroupShuffleXor(peak_4[0i], u32((16i>>u32(step_16))));
        peak_4[0i] = max(peak_4[0i], other_8);
      }
    } else {
      scores[(i32(threadIdx.x) + 512i)] = peak_4[0i];
      workgroupBarrier();
      for (var step_17 : i32 = 0i; step_17 < 5i; step_17++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_17))) {
          scores[(i32(threadIdx.x) + 512i)] = max(scores[(i32(threadIdx.x) + 512i)], scores[((i32(threadIdx.x) + (16i>>u32(step_17))) + 512i)]);
        }
        workgroupBarrier();
      }
      peak_4[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 512i)];
      workgroupBarrier();
    }
    next_max_4[0i] = max(v_4[0i], peak_4[0i]);
    var condval_27 : f32;
    if ((v_4[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
      condval_27 = 0.000000e+00f;
} else {
      condval_27 = exp((v_4[0i] - next_max_4[0i]));
}
    v_20[0i] = condval_27;
    var condval_28 : f32;
    if (valid_4) {
      condval_28 = exp((v_44[0i] - next_max_4[0i]));
} else {
      condval_28 = 0.000000e+00f;
}
    probability_4[0i] = condval_28;
    norm_4[0i] = probability_4[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_18 : i32 = 0i; step_18 < 5i; step_18++) {
        let other_9 : f32 = subgroupShuffleXor(norm_4[0i], u32((16i>>u32(step_18))));
        norm_4[0i] = (norm_4[0i] + other_9);
      }
    } else {
      scores[(i32(threadIdx.x) + 512i)] = norm_4[0i];
      workgroupBarrier();
      for (var step_19 : i32 = 0i; step_19 < 5i; step_19++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_19))) {
          scores[(i32(threadIdx.x) + 512i)] = (scores[(i32(threadIdx.x) + 512i)] + scores[((i32(threadIdx.x) + (16i>>u32(step_19))) + 512i)]);
        }
        workgroupBarrier();
      }
      norm_4[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 512i)];
      workgroupBarrier();
    }
    v_12[0i] = fma(v_12[0i], v_20[0i], norm_4[0i]);
    v_4[0i] = next_max_4[0i];
    scores[(i32(threadIdx.x) + 512i)] = probability_4[0i];
    let valid_5 : bool = (((enabled[((i32(blockIdx.y) * 8i) + 5i)] != 0i) && (tile < 256i)) && (((tile * 32i) + (i32(threadIdx.x) & 31i)) <= positions[((i32(blockIdx.y) * 8i) + 5i)]));
    var condval_29 : f32;
    if (valid_5) {
      condval_29 = (v_45[0i] * 1.250000e-01f);
} else {
      condval_29 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    v_45[0i] = condval_29;
    peak_5[0i] = v_45[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_20 : i32 = 0i; step_20 < 5i; step_20++) {
        let other_10 : f32 = subgroupShuffleXor(peak_5[0i], u32((16i>>u32(step_20))));
        peak_5[0i] = max(peak_5[0i], other_10);
      }
    } else {
      scores[(i32(threadIdx.x) + 640i)] = peak_5[0i];
      workgroupBarrier();
      for (var step_21 : i32 = 0i; step_21 < 5i; step_21++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_21))) {
          scores[(i32(threadIdx.x) + 640i)] = max(scores[(i32(threadIdx.x) + 640i)], scores[((i32(threadIdx.x) + (16i>>u32(step_21))) + 640i)]);
        }
        workgroupBarrier();
      }
      peak_5[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 640i)];
      workgroupBarrier();
    }
    next_max_5[0i] = max(v_5[0i], peak_5[0i]);
    var condval_30 : f32;
    if ((v_5[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
      condval_30 = 0.000000e+00f;
} else {
      condval_30 = exp((v_5[0i] - next_max_5[0i]));
}
    v_21[0i] = condval_30;
    var condval_31 : f32;
    if (valid_5) {
      condval_31 = exp((v_45[0i] - next_max_5[0i]));
} else {
      condval_31 = 0.000000e+00f;
}
    probability_5[0i] = condval_31;
    norm_5[0i] = probability_5[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_22 : i32 = 0i; step_22 < 5i; step_22++) {
        let other_11 : f32 = subgroupShuffleXor(norm_5[0i], u32((16i>>u32(step_22))));
        norm_5[0i] = (norm_5[0i] + other_11);
      }
    } else {
      scores[(i32(threadIdx.x) + 640i)] = norm_5[0i];
      workgroupBarrier();
      for (var step_23 : i32 = 0i; step_23 < 5i; step_23++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_23))) {
          scores[(i32(threadIdx.x) + 640i)] = (scores[(i32(threadIdx.x) + 640i)] + scores[((i32(threadIdx.x) + (16i>>u32(step_23))) + 640i)]);
        }
        workgroupBarrier();
      }
      norm_5[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 640i)];
      workgroupBarrier();
    }
    v_13[0i] = fma(v_13[0i], v_21[0i], norm_5[0i]);
    v_5[0i] = next_max_5[0i];
    scores[(i32(threadIdx.x) + 640i)] = probability_5[0i];
    let valid_6 : bool = (((enabled[((i32(blockIdx.y) * 8i) + 6i)] != 0i) && (tile < 256i)) && (((tile * 32i) + (i32(threadIdx.x) & 31i)) <= positions[((i32(blockIdx.y) * 8i) + 6i)]));
    var condval_32 : f32;
    if (valid_6) {
      condval_32 = (v_46[0i] * 1.250000e-01f);
} else {
      condval_32 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    v_46[0i] = condval_32;
    peak_6[0i] = v_46[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_24 : i32 = 0i; step_24 < 5i; step_24++) {
        let other_12 : f32 = subgroupShuffleXor(peak_6[0i], u32((16i>>u32(step_24))));
        peak_6[0i] = max(peak_6[0i], other_12);
      }
    } else {
      scores[(i32(threadIdx.x) + 768i)] = peak_6[0i];
      workgroupBarrier();
      for (var step_25 : i32 = 0i; step_25 < 5i; step_25++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_25))) {
          scores[(i32(threadIdx.x) + 768i)] = max(scores[(i32(threadIdx.x) + 768i)], scores[((i32(threadIdx.x) + (16i>>u32(step_25))) + 768i)]);
        }
        workgroupBarrier();
      }
      peak_6[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 768i)];
      workgroupBarrier();
    }
    next_max_6[0i] = max(v_6[0i], peak_6[0i]);
    var condval_33 : f32;
    if ((v_6[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
      condval_33 = 0.000000e+00f;
} else {
      condval_33 = exp((v_6[0i] - next_max_6[0i]));
}
    v_22[0i] = condval_33;
    var condval_34 : f32;
    if (valid_6) {
      condval_34 = exp((v_46[0i] - next_max_6[0i]));
} else {
      condval_34 = 0.000000e+00f;
}
    probability_6[0i] = condval_34;
    norm_6[0i] = probability_6[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_26 : i32 = 0i; step_26 < 5i; step_26++) {
        let other_13 : f32 = subgroupShuffleXor(norm_6[0i], u32((16i>>u32(step_26))));
        norm_6[0i] = (norm_6[0i] + other_13);
      }
    } else {
      scores[(i32(threadIdx.x) + 768i)] = norm_6[0i];
      workgroupBarrier();
      for (var step_27 : i32 = 0i; step_27 < 5i; step_27++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_27))) {
          scores[(i32(threadIdx.x) + 768i)] = (scores[(i32(threadIdx.x) + 768i)] + scores[((i32(threadIdx.x) + (16i>>u32(step_27))) + 768i)]);
        }
        workgroupBarrier();
      }
      norm_6[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 768i)];
      workgroupBarrier();
    }
    v_14[0i] = fma(v_14[0i], v_22[0i], norm_6[0i]);
    v_6[0i] = next_max_6[0i];
    scores[(i32(threadIdx.x) + 768i)] = probability_6[0i];
    let valid_7 : bool = (((enabled[((i32(blockIdx.y) * 8i) + 7i)] != 0i) && (tile < 256i)) && (((tile * 32i) + (i32(threadIdx.x) & 31i)) <= positions[((i32(blockIdx.y) * 8i) + 7i)]));
    var condval_35 : f32;
    if (valid_7) {
      condval_35 = (v_47[0i] * 1.250000e-01f);
} else {
      condval_35 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    v_47[0i] = condval_35;
    peak_7[0i] = v_47[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_28 : i32 = 0i; step_28 < 5i; step_28++) {
        let other_14 : f32 = subgroupShuffleXor(peak_7[0i], u32((16i>>u32(step_28))));
        peak_7[0i] = max(peak_7[0i], other_14);
      }
    } else {
      scores[(i32(threadIdx.x) + 896i)] = peak_7[0i];
      workgroupBarrier();
      for (var step_29 : i32 = 0i; step_29 < 5i; step_29++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_29))) {
          scores[(i32(threadIdx.x) + 896i)] = max(scores[(i32(threadIdx.x) + 896i)], scores[((i32(threadIdx.x) + (16i>>u32(step_29))) + 896i)]);
        }
        workgroupBarrier();
      }
      peak_7[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 896i)];
      workgroupBarrier();
    }
    next_max_7[0i] = max(v_7[0i], peak_7[0i]);
    var condval_36 : f32;
    if ((v_7[0i] == (bitcast<f32>(2139095040u) * -1.000000e+00f))) {
      condval_36 = 0.000000e+00f;
} else {
      condval_36 = exp((v_7[0i] - next_max_7[0i]));
}
    v_23[0i] = condval_36;
    var condval_37 : f32;
    if (valid_7) {
      condval_37 = exp((v_47[0i] - next_max_7[0i]));
} else {
      condval_37 = 0.000000e+00f;
}
    probability_7[0i] = condval_37;
    norm_7[0i] = probability_7[0i];
    if (1u <= (tensorSubgroupSize>>5u)) {
      for (var step_30 : i32 = 0i; step_30 < 5i; step_30++) {
        let other_15 : f32 = subgroupShuffleXor(norm_7[0i], u32((16i>>u32(step_30))));
        norm_7[0i] = (norm_7[0i] + other_15);
      }
    } else {
      scores[(i32(threadIdx.x) + 896i)] = norm_7[0i];
      workgroupBarrier();
      for (var step_31 : i32 = 0i; step_31 < 5i; step_31++) {
        if ((i32(threadIdx.x) & 31i) < (16i>>u32(step_31))) {
          scores[(i32(threadIdx.x) + 896i)] = (scores[(i32(threadIdx.x) + 896i)] + scores[((i32(threadIdx.x) + (16i>>u32(step_31))) + 896i)]);
        }
        workgroupBarrier();
      }
      norm_7[0i] = scores[(((i32(threadIdx.x)>>5u) * 32i) + 896i)];
      workgroupBarrier();
    }
    v_15[0i] = fma(v_15[0i], v_23[0i], norm_7[0i]);
    v_7[0i] = next_max_7[0i];
    scores[(i32(threadIdx.x) + 896i)] = probability_7[0i];
    workgroupBarrier();
    let factor : f32 = v_16[0i];
    v_24[0i] = (v_24[0i] * factor);
    let factor_1 : f32 = v_16[0i];
    v_25[0i] = (v_25[0i] * factor_1);
    let factor_2 : f32 = v_17[0i];
    v_26[0i] = (v_26[0i] * factor_2);
    let factor_3 : f32 = v_17[0i];
    v_27[0i] = (v_27[0i] * factor_3);
    let factor_4 : f32 = v_18[0i];
    v_28[0i] = (v_28[0i] * factor_4);
    let factor_5 : f32 = v_18[0i];
    v_29[0i] = (v_29[0i] * factor_5);
    let factor_6 : f32 = v_19[0i];
    v_30[0i] = (v_30[0i] * factor_6);
    let factor_7 : f32 = v_19[0i];
    v_31[0i] = (v_31[0i] * factor_7);
    let factor_8 : f32 = v_20[0i];
    v_32[0i] = (v_32[0i] * factor_8);
    let factor_9 : f32 = v_20[0i];
    v_33[0i] = (v_33[0i] * factor_9);
    let factor_10 : f32 = v_21[0i];
    v_34[0i] = (v_34[0i] * factor_10);
    let factor_11 : f32 = v_21[0i];
    v_35[0i] = (v_35[0i] * factor_11);
    let factor_12 : f32 = v_22[0i];
    v_36[0i] = (v_36[0i] * factor_12);
    let factor_13 : f32 = v_22[0i];
    v_37[0i] = (v_37[0i] * factor_13);
    let factor_14 : f32 = v_23[0i];
    v_38[0i] = (v_38[0i] * factor_14);
    let factor_15 : f32 = v_23[0i];
    v_39[0i] = (v_39[0i] * factor_15);
    for (var key : i32 = 0i; key < 8i; key++) {
      right0[0i] = vec4<f32>(f32(values[((key * 256i) + (i32(threadIdx.x) & 31i))]), f32(values[(((key * 256i) + (i32(threadIdx.x) & 31i)) + 64i)]), f32(values[(((key * 256i) + (i32(threadIdx.x) & 31i)) + 128i)]), f32(values[(((key * 256i) + (i32(threadIdx.x) & 31i)) + 192i)]));
      right1[0i] = vec4<f32>(f32(values[(((key * 256i) + (i32(threadIdx.x) & 31i)) + 32i)]), f32(values[(((key * 256i) + (i32(threadIdx.x) & 31i)) + 96i)]), f32(values[(((key * 256i) + (i32(threadIdx.x) & 31i)) + 160i)]), f32(values[(((key * 256i) + (i32(threadIdx.x) & 31i)) + 224i)]));
      let right0_1 : vec4<f32> = right0[0i];
      let right1_1 : vec4<f32> = right1[0i];
      let left_8 : vec4<f32> = vec4<f32>(scores[(((i32(threadIdx.x)>>5u) * 32i) + (key * 4i))], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 1i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 2i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 3i)]);
      v_24[0i] = (v_24[0i] + dot(left_8, right0_1));
      v_25[0i] = (v_25[0i] + dot(left_8, right1_1));
      let right0_2 : vec4<f32> = right0[0i];
      let right1_2 : vec4<f32> = right1[0i];
      let left_9 : vec4<f32> = vec4<f32>(scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 128i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 129i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 130i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 131i)]);
      v_26[0i] = (v_26[0i] + dot(left_9, right0_2));
      v_27[0i] = (v_27[0i] + dot(left_9, right1_2));
      let right0_3 : vec4<f32> = right0[0i];
      let right1_3 : vec4<f32> = right1[0i];
      let left_10 : vec4<f32> = vec4<f32>(scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 256i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 257i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 258i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 259i)]);
      v_28[0i] = (v_28[0i] + dot(left_10, right0_3));
      v_29[0i] = (v_29[0i] + dot(left_10, right1_3));
      let right0_4 : vec4<f32> = right0[0i];
      let right1_4 : vec4<f32> = right1[0i];
      let left_11 : vec4<f32> = vec4<f32>(scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 384i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 385i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 386i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 387i)]);
      v_30[0i] = (v_30[0i] + dot(left_11, right0_4));
      v_31[0i] = (v_31[0i] + dot(left_11, right1_4));
      let right0_5 : vec4<f32> = right0[0i];
      let right1_5 : vec4<f32> = right1[0i];
      let left_12 : vec4<f32> = vec4<f32>(scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 512i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 513i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 514i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 515i)]);
      v_32[0i] = (v_32[0i] + dot(left_12, right0_5));
      v_33[0i] = (v_33[0i] + dot(left_12, right1_5));
      let right0_6 : vec4<f32> = right0[0i];
      let right1_6 : vec4<f32> = right1[0i];
      let left_13 : vec4<f32> = vec4<f32>(scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 640i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 641i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 642i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 643i)]);
      v_34[0i] = (v_34[0i] + dot(left_13, right0_6));
      v_35[0i] = (v_35[0i] + dot(left_13, right1_6));
      let right0_7 : vec4<f32> = right0[0i];
      let right1_7 : vec4<f32> = right1[0i];
      let left_14 : vec4<f32> = vec4<f32>(scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 768i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 769i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 770i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 771i)]);
      v_36[0i] = (v_36[0i] + dot(left_14, right0_7));
      v_37[0i] = (v_37[0i] + dot(left_14, right1_7));
      let right0_8 : vec4<f32> = right0[0i];
      let right1_8 : vec4<f32> = right1[0i];
      let left_15 : vec4<f32> = vec4<f32>(scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 896i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 897i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 898i)], scores[((((i32(threadIdx.x)>>5u) * 32i) + (key * 4i)) + 899i)]);
      v_38[0i] = (v_38[0i] + dot(left_15, right0_8));
      v_39[0i] = (v_39[0i] + dot(left_15, right1_8));
    }
    workgroupBarrier();
  }
  let value : f32 = v_24[0i];
  let norm_8 : f32 = v_8[0i];
  var condval_38 : f32;
  if ((enabled[(i32(blockIdx.y) * 8i)] != 0i)) {
    condval_38 = (value / max(norm_8, 1.000000e-20f));
} else {
    condval_38 = 0.000000e+00f;
}
  out[((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i))] = condval_38;
  let value_1 : f32 = v_25[0i];
  let norm_9 : f32 = v_8[0i];
  var condval_39 : f32;
  if ((enabled[(i32(blockIdx.y) * 8i)] != 0i)) {
    condval_39 = (value_1 / max(norm_9, 1.000000e-20f));
} else {
    condval_39 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 32i)] = condval_39;
  let value_2 : f32 = v_26[0i];
  let norm_10 : f32 = v_9[0i];
  var condval_40 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 1i)] != 0i)) {
    condval_40 = (value_2 / max(norm_10, 1.000000e-20f));
} else {
    condval_40 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 2048i)] = condval_40;
  let value_3 : f32 = v_27[0i];
  let norm_11 : f32 = v_9[0i];
  var condval_41 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 1i)] != 0i)) {
    condval_41 = (value_3 / max(norm_11, 1.000000e-20f));
} else {
    condval_41 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 2080i)] = condval_41;
  let value_4 : f32 = v_28[0i];
  let norm_12 : f32 = v_10[0i];
  var condval_42 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 2i)] != 0i)) {
    condval_42 = (value_4 / max(norm_12, 1.000000e-20f));
} else {
    condval_42 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 4096i)] = condval_42;
  let value_5 : f32 = v_29[0i];
  let norm_13 : f32 = v_10[0i];
  var condval_43 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 2i)] != 0i)) {
    condval_43 = (value_5 / max(norm_13, 1.000000e-20f));
} else {
    condval_43 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 4128i)] = condval_43;
  let value_6 : f32 = v_30[0i];
  let norm_14 : f32 = v_11[0i];
  var condval_44 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 3i)] != 0i)) {
    condval_44 = (value_6 / max(norm_14, 1.000000e-20f));
} else {
    condval_44 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 6144i)] = condval_44;
  let value_7 : f32 = v_31[0i];
  let norm_15 : f32 = v_11[0i];
  var condval_45 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 3i)] != 0i)) {
    condval_45 = (value_7 / max(norm_15, 1.000000e-20f));
} else {
    condval_45 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 6176i)] = condval_45;
  let value_8 : f32 = v_32[0i];
  let norm_16 : f32 = v_12[0i];
  var condval_46 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 4i)] != 0i)) {
    condval_46 = (value_8 / max(norm_16, 1.000000e-20f));
} else {
    condval_46 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 8192i)] = condval_46;
  let value_9 : f32 = v_33[0i];
  let norm_17 : f32 = v_12[0i];
  var condval_47 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 4i)] != 0i)) {
    condval_47 = (value_9 / max(norm_17, 1.000000e-20f));
} else {
    condval_47 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 8224i)] = condval_47;
  let value_10 : f32 = v_34[0i];
  let norm_18 : f32 = v_13[0i];
  var condval_48 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 5i)] != 0i)) {
    condval_48 = (value_10 / max(norm_18, 1.000000e-20f));
} else {
    condval_48 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 10240i)] = condval_48;
  let value_11 : f32 = v_35[0i];
  let norm_19 : f32 = v_13[0i];
  var condval_49 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 5i)] != 0i)) {
    condval_49 = (value_11 / max(norm_19, 1.000000e-20f));
} else {
    condval_49 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 10272i)] = condval_49;
  let value_12 : f32 = v_36[0i];
  let norm_20 : f32 = v_14[0i];
  var condval_50 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 6i)] != 0i)) {
    condval_50 = (value_12 / max(norm_20, 1.000000e-20f));
} else {
    condval_50 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 12288i)] = condval_50;
  let value_13 : f32 = v_37[0i];
  let norm_21 : f32 = v_14[0i];
  var condval_51 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 6i)] != 0i)) {
    condval_51 = (value_13 / max(norm_21, 1.000000e-20f));
} else {
    condval_51 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 12320i)] = condval_51;
  let value_14 : f32 = v_38[0i];
  let norm_22 : f32 = v_15[0i];
  var condval_52 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 7i)] != 0i)) {
    condval_52 = (value_14 / max(norm_22, 1.000000e-20f));
} else {
    condval_52 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 14336i)] = condval_52;
  let value_15 : f32 = v_39[0i];
  let norm_23 : f32 = v_15[0i];
  var condval_53 : f32;
  if ((enabled[((i32(blockIdx.y) * 8i) + 7i)] != 0i)) {
    condval_53 = (value_15 / max(norm_23, 1.000000e-20f));
} else {
    condval_53 = 0.000000e+00f;
}
  out[(((((i32(blockIdx.y) * 16384i) + (v__1 * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 14368i)] = condval_53;
}



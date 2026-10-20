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

var<workgroup> queries : array<f32, 256>;
var<workgroup> keys : array<f32, 2112>;
var<workgroup> values : array<f32, 2048>;
var<workgroup> probabilities : array<f32, 128>;
@compute @workgroup_size(128, 1, 1)
fn kernel_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var maximum : array<f32, 1>;
  var normalizer : array<f32, 1>;
  var acc0 : array<f32, 1>;
  var acc1 : array<f32, 1>;
  var score : array<f32, 1>;
  var peak : array<f32, 1>;
  var norm : array<f32, 1>;
    var condval : i32;
  if ((enabled[0i] != 0i)) {
    condval = min((positions[0i] + 1i), 32i);
} else {
    condval = 0i;
}
let length : i32 = condval;
  let _tmp : f32 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  maximum[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  normalizer[0i] = 0.000000e+00f;
  acc0[0i] = 0.000000e+00f;
  acc1[0i] = 0.000000e+00f;
  queries[(((i32(threadIdx.x)>>5u) * 64i) + (i32(threadIdx.x) & 31i))] = q[(((v__1 * 256i) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i))];
  queries[((((i32(threadIdx.x)>>5u) * 64i) + (i32(threadIdx.x) & 31i)) + 32i)] = q[((((v__1 * 256i) + ((i32(threadIdx.x)>>5u) * 64i)) + (i32(threadIdx.x) & 31i)) + 32i)];
  workgroupBarrier();
  for (var tile : i32 = 0i; tile < ((max((min(((i32(blockIdx.y) + 1i) * ((length + 15i) / 16i)), length) - (i32(blockIdx.y) * ((length + 15i) / 16i))), 0i) + 31i) / 32i); tile++) {
    for (var i : i32 = 0i; i < 16i; i++) {
      var condval_1 : f32;
      if (((((((tile * 32i) + (i * 2i)) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i))) < ((i32(blockIdx.y) + 1i) * ((length + 15i) / 16i))) && (((((tile * 32i) + (i * 2i)) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i))) < length))) {
        var condval_2 : f16;
        if (((((((i * 2i) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i)))>>5u) + tile) < 1i)) {
          condval_2 = kc[(((((v__1 * 2048i) + (tile * 2048i)) + (i * 128i)) + ((i32(blockIdx.y) * ((length + 15i) / 16i)) * 64i)) + i32(threadIdx.x))];
} else {
          condval_2 = 0.000000e+00h;
}
        var condval_3 : f32;
        if (((((((i * 2i) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i)))>>5u) + tile) < 1i)) {
          condval_3 = ks[(((((v__1 * 32i) + (tile * 32i)) + (i * 2i)) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i)))];
} else {
          condval_3 = 0.000000e+00f;
}
        condval_1 = (f32(condval_2) * condval_3);
} else {
        condval_1 = 0.000000e+00f;
}
      keys[((((i32(threadIdx.x) & 63i) * 33i) + (i * 2i)) + (i32(threadIdx.x)>>6u))] = condval_1;
      var condval_4 : f32;
      if (((((((tile * 32i) + (i * 2i)) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i))) < ((i32(blockIdx.y) + 1i) * ((length + 15i) / 16i))) && (((((tile * 32i) + (i * 2i)) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i))) < length))) {
        var condval_5 : f16;
        if (((((((i * 2i) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i)))>>5u) + tile) < 1i)) {
          condval_5 = vc[(((((v__1 * 2048i) + (tile * 2048i)) + (i * 128i)) + ((i32(blockIdx.y) * ((length + 15i) / 16i)) * 64i)) + i32(threadIdx.x))];
} else {
          condval_5 = 0.000000e+00h;
}
        var condval_6 : f32;
        if (((((((i * 2i) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i)))>>5u) + tile) < 1i)) {
          condval_6 = vs[(((((v__1 * 32i) + (tile * 32i)) + (i * 2i)) + (i32(threadIdx.x)>>6u)) + (i32(blockIdx.y) * ((length + 15i) / 16i)))];
} else {
          condval_6 = 0.000000e+00f;
}
        condval_4 = (f32(condval_5) * condval_6);
} else {
        condval_4 = 0.000000e+00f;
}
      values[((i * 128i) + i32(threadIdx.x))] = condval_4;
    }
    workgroupBarrier();
    score[0i] = 0.000000e+00f;
    for (var channel : i32 = 0i; channel < 16i; channel++) {
      for (var j : i32 = 0i; j < 4i; j++) {
        score[0i] = fma(queries[((((i32(threadIdx.x)>>5u) * 64i) + (channel * 4i)) + j)], keys[(((channel * 132i) + (j * 33i)) + (i32(threadIdx.x) & 31i))], score[0i]);
      }
    }
    var condval_7 : f32;
    if ((((((tile * 32i) + (i32(blockIdx.y) * ((length + 15i) / 16i))) + (i32(threadIdx.x) & 31i)) < ((i32(blockIdx.y) + 1i) * ((length + 15i) / 16i))) && ((((tile * 32i) + (i32(blockIdx.y) * ((length + 15i) / 16i))) + (i32(threadIdx.x) & 31i)) < length))) {
      condval_7 = (score[0i] * 1.250000e-01f);
} else {
      condval_7 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
}
    score[0i] = condval_7;
    peak[0i] = score[0i];
    for (var step : i32 = 0i; step < 5i; step++) {
      let value : f32 = peak[0i];
      peak[0i] = max(peak[0i], subgroupShuffleXor(value, u32((16i>>u32(step)))));
    }
    let next_max : f32 = max(maximum[0i], peak[0i]);
    let correction : f32 = exp((maximum[0i] - next_max));
        var condval_8 : f32;
    if ((((((tile * 32i) + (i32(blockIdx.y) * ((length + 15i) / 16i))) + (i32(threadIdx.x) & 31i)) < ((i32(blockIdx.y) + 1i) * ((length + 15i) / 16i))) && ((((tile * 32i) + (i32(blockIdx.y) * ((length + 15i) / 16i))) + (i32(threadIdx.x) & 31i)) < length))) {
      condval_8 = exp((score[0i] - next_max));
} else {
      condval_8 = 0.000000e+00f;
}
let probability : f32 = condval_8;
    norm[0i] = probability;
    for (var step_1 : i32 = 0i; step_1 < 5i; step_1++) {
      let value_1 : f32 = norm[0i];
      norm[0i] = (norm[0i] + subgroupShuffleXor(value_1, u32((16i>>u32(step_1)))));
    }
    normalizer[0i] = fma(normalizer[0i], correction, norm[0i]);
    maximum[0i] = next_max;
    probabilities[i32(threadIdx.x)] = probability;
    let _tmp_1 : f32 = (acc0[0i] * correction);
    let _tmp_2 : f32 = (acc1[0i] * correction);
    acc0[0i] = _tmp_1;
    acc1[0i] = _tmp_2;
    workgroupBarrier();
    for (var key : i32 = 0i; key < 32i; key++) {
      acc0[0i] = fma(probabilities[(((i32(threadIdx.x)>>5u) * 32i) + key)], values[((key * 64i) + (i32(threadIdx.x) & 31i))], acc0[0i]);
      acc1[0i] = fma(probabilities[(((i32(threadIdx.x)>>5u) * 32i) + key)], values[(((key * 64i) + (i32(threadIdx.x) & 31i)) + 32i)], acc1[0i]);
    }
    workgroupBarrier();
  }
  out[((((v__1 * 4224i) + ((i32(threadIdx.x)>>5u) * 1056i)) + (i32(blockIdx.y) * 66i)) + (i32(threadIdx.x) & 31i))] = acc0[0i];
  out[(((((v__1 * 4224i) + ((i32(threadIdx.x)>>5u) * 1056i)) + (i32(blockIdx.y) * 66i)) + (i32(threadIdx.x) & 31i)) + 32i)] = acc1[0i];
  if ((i32(threadIdx.x) % 32i) == 0i) {
    out[((((v__1 * 4224i) + ((i32(threadIdx.x)>>5u) * 1056i)) + (i32(blockIdx.y) * 66i)) + 64i)] = maximum[0i];
    out[((((v__1 * 4224i) + ((i32(threadIdx.x)>>5u) * 1056i)) + (i32(blockIdx.y) * 66i)) + 65i)] = normalizer[0i];
  }
}



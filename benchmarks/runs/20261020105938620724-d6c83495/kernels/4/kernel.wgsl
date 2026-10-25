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

var<workgroup> values : array<f32, 256>;
var<workgroup> maxima : array<f32, 4>;
var<workgroup> sums : array<f32, 4>;
@compute @workgroup_size(128, 1, 1)
fn kernel_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>,
  @builtin(subgroup_size) tensorSubgroupSize : u32
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var maximum : array<f32, 1>;
  var normalizer : array<f32, 1>;
  var a0 : array<f32, 1>;
  var a1 : array<f32, 1>;
  var score : array<f32, 1>;
  var total : array<f32, 1>;
  var result0 : array<f32, 1>;
  var result1 : array<f32, 1>;
    var condval : i32;
  if ((enabled[0i] != 0i)) {
    condval = min((positions[0i] + 1i), 8192i);
} else {
    condval = 0i;
}
let length : i32 = condval;
  let _tmp : f32 = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  maximum[0i] = (bitcast<f32>(2139095040u) * -1.000000e+00f);
  normalizer[0i] = 0.000000e+00f;
  a0[0i] = 0.000000e+00f;
  a1[0i] = 0.000000e+00f;
  let q0 : f32 = q[((v__1 * 64i) + (i32(threadIdx.x) & 31i))];
  let q1 : f32 = q[(((v__1 * 64i) + (i32(threadIdx.x) & 31i)) + 32i)];
  for (var offset : i32 = 0i; offset < ((max(((min(((i32(blockIdx.y) + 1i) * ((length + 15i)>>4u)), length) - (i32(threadIdx.x)>>5u)) - (i32(blockIdx.y) * ((length + 15i)>>4u))), 0i) + 3i)>>2u); offset++) {
    var condval_1 : f16;
    if ((((((i32(threadIdx.x)>>5u) + (i32(blockIdx.y) * ((length + 15i)>>4u)))>>2u) + offset) < 2048i)) {
      condval_1 = kc[(((((((v__1>>2u) * 524288i) + (offset * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + ((i32(blockIdx.y) * ((length + 15i)>>4u)) * 64i)) + (i32(threadIdx.x) & 31i)) + 32i)];
} else {
      condval_1 = 0.000000e+00h;
}
    var condval_2 : f32;
    if ((((((i32(threadIdx.x)>>5u) + (i32(blockIdx.y) * ((length + 15i)>>4u)))>>2u) + offset) < 2048i)) {
      condval_2 = ks[(((((v__1>>2u) * 8192i) + (offset * 4i)) + (i32(threadIdx.x)>>5u)) + (i32(blockIdx.y) * ((length + 15i)>>4u)))];
} else {
      condval_2 = 0.000000e+00f;
}
    var condval_3 : f16;
    if ((((((i32(threadIdx.x)>>5u) + (i32(blockIdx.y) * ((length + 15i)>>4u)))>>2u) + offset) < 2048i)) {
      condval_3 = kc[((((((v__1>>2u) * 524288i) + (offset * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + ((i32(blockIdx.y) * ((length + 15i)>>4u)) * 64i)) + (i32(threadIdx.x) & 31i))];
} else {
      condval_3 = 0.000000e+00h;
}
    var condval_4 : f32;
    if ((((((i32(threadIdx.x)>>5u) + (i32(blockIdx.y) * ((length + 15i)>>4u)))>>2u) + offset) < 2048i)) {
      condval_4 = ks[(((((v__1>>2u) * 8192i) + (offset * 4i)) + (i32(threadIdx.x)>>5u)) + (i32(blockIdx.y) * ((length + 15i)>>4u)))];
} else {
      condval_4 = 0.000000e+00f;
}
    score[0i] = fma(q1, (f32(condval_1) * condval_2), (q0 * (f32(condval_3) * condval_4)));
    for (var step : i32 = 0i; step < 5i; step++) {
      let value : f32 = score[0i];
      score[0i] = (score[0i] + subgroupShuffleXor(value, u32((16i>>u32(step)))));
    }
    let value_1 : f32 = score[0i];
    let size : u32 = tensorSubgroupSize;
    score[0i] = (subgroupShuffle(value_1, (((u32(i32(threadIdx.x)) % size)>>5u) * 32u)) * 1.250000e-01f);
    let next_max : f32 = max(maximum[0i], score[0i]);
    let correction : f32 = exp((maximum[0i] - next_max));
    let probability : f32 = exp((score[0i] - next_max));
    var condval_5 : f16;
    if ((((((i32(threadIdx.x)>>5u) + (i32(blockIdx.y) * ((length + 15i)>>4u)))>>2u) + offset) < 2048i)) {
      condval_5 = vc[((((((v__1>>2u) * 524288i) + (offset * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + ((i32(blockIdx.y) * ((length + 15i)>>4u)) * 64i)) + (i32(threadIdx.x) & 31i))];
} else {
      condval_5 = 0.000000e+00h;
}
    var condval_6 : f32;
    if ((((((i32(threadIdx.x)>>5u) + (i32(blockIdx.y) * ((length + 15i)>>4u)))>>2u) + offset) < 2048i)) {
      condval_6 = vs[(((((v__1>>2u) * 8192i) + (offset * 4i)) + (i32(threadIdx.x)>>5u)) + (i32(blockIdx.y) * ((length + 15i)>>4u)))];
} else {
      condval_6 = 0.000000e+00f;
}
    a0[0i] = fma(probability, (f32(condval_5) * condval_6), (a0[0i] * correction));
    var condval_7 : f16;
    if ((((((i32(threadIdx.x)>>5u) + (i32(blockIdx.y) * ((length + 15i)>>4u)))>>2u) + offset) < 2048i)) {
      condval_7 = vc[(((((((v__1>>2u) * 524288i) + (offset * 256i)) + ((i32(threadIdx.x)>>5u) * 64i)) + ((i32(blockIdx.y) * ((length + 15i)>>4u)) * 64i)) + (i32(threadIdx.x) & 31i)) + 32i)];
} else {
      condval_7 = 0.000000e+00h;
}
    var condval_8 : f32;
    if ((((((i32(threadIdx.x)>>5u) + (i32(blockIdx.y) * ((length + 15i)>>4u)))>>2u) + offset) < 2048i)) {
      condval_8 = vs[(((((v__1>>2u) * 8192i) + (offset * 4i)) + (i32(threadIdx.x)>>5u)) + (i32(blockIdx.y) * ((length + 15i)>>4u)))];
} else {
      condval_8 = 0.000000e+00f;
}
    a1[0i] = fma(probability, (f32(condval_7) * condval_8), (a1[0i] * correction));
    normalizer[0i] = fma(normalizer[0i], correction, probability);
    maximum[0i] = next_max;
  }
  let _tmp_1 : f32 = a0[0i];
  let _tmp_2 : f32 = a1[0i];
  values[(((i32(threadIdx.x)>>5u) * 64i) + (i32(threadIdx.x) & 31i))] = _tmp_1;
  values[((((i32(threadIdx.x)>>5u) * 64i) + (i32(threadIdx.x) & 31i)) + 32i)] = _tmp_2;
  if ((i32(threadIdx.x) % 32i) == 0i) {
    maxima[(i32(threadIdx.x)>>5u)] = maximum[0i];
    sums[(i32(threadIdx.x)>>5u)] = normalizer[0i];
  }
  workgroupBarrier();
  if ((i32(threadIdx.x)>>5u) == 0i) {
    total[0i] = 0.000000e+00f;
    result0[0i] = 0.000000e+00f;
    result1[0i] = 0.000000e+00f;
    for (var other : i32 = 0i; other < 4i; other++) {
            var condval_9 : f32;
      if ((((i32(blockIdx.y) * ((length + 15i)>>4u)) < ((i32(blockIdx.y) + 1i) * ((length + 15i)>>4u))) && ((i32(blockIdx.y) * ((length + 15i)>>4u)) < length))) {
        condval_9 = exp((maxima[other] - max(max(maxima[0i], maxima[1i]), max(maxima[2i], maxima[3i]))));
} else {
        condval_9 = 0.000000e+00f;
}
let factor : f32 = condval_9;
      total[0i] = fma(sums[other], factor, total[0i]);
      result0[0i] = fma(values[((other * 64i) + (i32(threadIdx.x) & 31i))], factor, result0[0i]);
      result1[0i] = fma(values[(((other * 64i) + (i32(threadIdx.x) & 31i)) + 32i)], factor, result1[0i]);
    }
    let _tmp_3 : f32 = result0[0i];
    let _tmp_4 : f32 = result1[0i];
    out[(((v__1 * 1056i) + (i32(blockIdx.y) * 66i)) + (i32(threadIdx.x) & 31i))] = _tmp_3;
    out[((((v__1 * 1056i) + (i32(blockIdx.y) * 66i)) + (i32(threadIdx.x) & 31i)) + 32i)] = _tmp_4;
    if ((i32(threadIdx.x) % 32i) == 0i) {
      out[(((v__1 * 1056i) + (i32(blockIdx.y) * 66i)) + 64i)] = max(max(maxima[0i], maxima[1i]), max(maxima[2i], maxima[3i]));
      out[(((v__1 * 1056i) + (i32(blockIdx.y) * 66i)) + 65i)] = total[0i];
    }
  }
}



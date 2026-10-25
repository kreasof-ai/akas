# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def kernel(q_handle: T.handle, kc_handle: T.handle, vc_handle: T.handle, ks_handle: T.handle, vs_handle: T.handle, out_handle: T.handle, positions_handle: T.handle, enabled_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["webgpu", "gpu"], "kind": "webgpu", "max_num_threads": 256, "supports_subgroups": False, "tag": "", "thread_warp_size": 1}), "tl.has_tma": T.bool(False)})
        q = T.match_buffer(q_handle, (1, 2048), strides=(2048, 1))
        kc = T.match_buffer(kc_handle, (1, 8, 131072, 64), "float16", strides=(67108864, 8388608, 64, 1))
        vc = T.match_buffer(vc_handle, (1, 8, 131072, 64), "float16", strides=(67108864, 8388608, 64, 1))
        ks = T.match_buffer(ks_handle, (1, 8, 131072, 1), strides=(1048576, 131072, 1, 1))
        vs = T.match_buffer(vs_handle, (1, 8, 131072, 1), strides=(1048576, 131072, 1, 1))
        out = T.match_buffer(out_handle, (270336,), strides=(1,))
        positions = T.match_buffer(positions_handle, (1,), "int32", strides=(1,))
        enabled = T.match_buffer(enabled_handle, (1,), "int32", strides=(1,))
        head = T.launch_thread("blockIdx.x", 8)
        wgpu_batch_head = T.launch_thread("blockIdx.y", 128)
        tx = T.launch_thread("threadIdx.x", 256)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        scores = T.alloc_buffer((4100,), scope="shared")
        scores_1 = T.decl_buffer((4100,), data=scores.data, scope="shared")
        reductions = T.alloc_buffer((8,), scope="shared")
        reductions_1 = T.decl_buffer((8,), data=reductions.data, scope="shared")
        buffer = T.alloc_buffer((1,), scope="local.var")
        buffer_1 = T.decl_buffer((1,), data=buffer.data, scope="local.var")
        buffer_2 = T.alloc_buffer((1,), scope="local.var")
        buffer_3 = T.decl_buffer((1,), data=buffer_2.data, scope="local.var")
        buffer_4 = T.alloc_buffer((1,), scope="local.var")
        buffer_5 = T.decl_buffer((1,), data=buffer_4.data, scope="local.var")
        buffer_6 = T.alloc_buffer((1,), scope="local.var")
        buffer_7 = T.decl_buffer((1,), data=buffer_6.data, scope="local.var")
        buffer_8 = T.alloc_buffer((1,), scope="local.var")
        buffer_9 = T.decl_buffer((1,), data=buffer_8.data, scope="local.var")
        buffer_10 = T.alloc_buffer((1,), scope="local.var")
        buffer_11 = T.decl_buffer((1,), data=buffer_10.data, scope="local.var")
        buffer_12 = T.alloc_buffer((1,), scope="local.var")
        buffer_13 = T.decl_buffer((1,), data=buffer_12.data, scope="local.var")
        buffer_14 = T.alloc_buffer((1,), scope="local.var")
        buffer_15 = T.decl_buffer((1,), data=buffer_14.data, scope="local.var")
        buffer_16 = T.alloc_buffer((1,), scope="local.var")
        buffer_17 = T.decl_buffer((1,), data=buffer_16.data, scope="local.var")
        buffer_18 = T.alloc_buffer((1,), scope="local.var")
        buffer_19 = T.decl_buffer((1,), data=buffer_18.data, scope="local.var")
        buffer_20 = T.alloc_buffer((1,), scope="local.var")
        buffer_21 = T.decl_buffer((1,), data=buffer_20.data, scope="local.var")
        buffer_22 = T.alloc_buffer((1,), scope="local.var")
        buffer_23 = T.decl_buffer((1,), data=buffer_22.data, scope="local.var")
        maximum = T.alloc_buffer((1,), scope="local.var")
        maximum_1 = T.decl_buffer((1,), data=maximum.data, scope="local.var")
        normalizer = T.alloc_buffer((1,), scope="local.var")
        normalizer_1 = T.decl_buffer((1,), data=normalizer.data, scope="local.var")
        result = T.alloc_buffer((1,), scope="local.var")
        result_1 = T.decl_buffer((1,), data=result.data, scope="local.var")
        enabled_1 = T.Buffer((1,), "int32", data=enabled.data)
        positions_1 = T.Buffer((1,), "int32", data=positions.data)
        length: T.int32 = T.if_then_else(enabled_1[0] != 0, T.min(positions_1[0] + 1, 131072), 0)
        buffer_1[0] = T.float32(0.0)
        buffer_3[0] = T.float32(0.0)
        buffer_5[0] = T.float32(0.0)
        buffer_7[0] = T.float32(0.0)
        buffer_9[0] = T.float32(0.0)
        buffer_11[0] = T.float32(0.0)
        buffer_13[0] = T.float32(0.0)
        buffer_15[0] = T.float32(0.0)
        buffer_17[0] = T.float32(0.0)
        buffer_19[0] = T.float32(0.0)
        buffer_21[0] = T.float32(0.0)
        buffer_23[0] = T.float32(0.0)
        _tmp: T.float32 = T.reinterpret("float32", T.uint32(2139095040)) * T.float32(-1.0)
        maximum_1[0] = T.reinterpret("float32", T.uint32(2139095040)) * T.float32(-1.0)
        normalizer_1[0] = T.float32(0.0)
        result_1[0] = T.float32(0.0)
        q_1 = T.Buffer((2048,), data=q.data)
        if wgpu_batch_head * 128 < length:
            buffer_1[0] = q_1[head * 256 + tx // 64 * 64 + tx % 8 * 8]
        if wgpu_batch_head * 128 < length:
            buffer_3[0] = q_1[head * 256 + tx // 64 * 64 + tx % 8 * 8 + 1]
        if wgpu_batch_head * 128 < length:
            buffer_5[0] = q_1[head * 256 + tx // 64 * 64 + tx % 8 * 8 + 2]
        if wgpu_batch_head * 128 < length:
            buffer_7[0] = q_1[head * 256 + tx // 64 * 64 + tx % 8 * 8 + 3]
        if wgpu_batch_head * 128 < length:
            buffer_9[0] = q_1[head * 256 + tx // 64 * 64 + tx % 8 * 8 + 4]
        if wgpu_batch_head * 128 < length:
            buffer_11[0] = q_1[head * 256 + tx // 64 * 64 + tx % 8 * 8 + 5]
        if wgpu_batch_head * 128 < length:
            buffer_13[0] = q_1[head * 256 + tx // 64 * 64 + tx % 8 * 8 + 6]
        if wgpu_batch_head * 128 < length:
            buffer_15[0] = q_1[head * 256 + tx // 64 * 64 + tx % 8 * 8 + 7]
        if wgpu_batch_head * 128 < length:
            for tile in range((T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16):
                score = T.alloc_buffer((1,), scope="local.var")
                score_1 = T.decl_buffer((1,), data=score.data, scope="local.var")
                score_1[0] = T.float32(0.0)
                if tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + tx % 64 // 8 < length:
                    left: T.float32x4 = T.call_extern("float32x4", "vec4<f32>", buffer_1[0], buffer_3[0], buffer_5[0], buffer_7[0])
                    kc_1 = T.Buffer((67108864,), "float16", data=kc.data)
                    ks_1 = T.Buffer((1048576,), data=ks.data)
                    right: T.float32x4 = T.call_extern("float32x4", "vec4<f32>", T.Cast("float32", T.if_then_else(tile < 128, kc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) * T.int64(8)], T.float16(0.0))) * ks_1[head * 131072], T.Cast("float32", T.if_then_else(tile < 128, kc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) * T.int64(8) + T.int64(1)], T.float16(0.0))) * ks_1[head * 131072], T.Cast("float32", T.if_then_else(tile < 128, kc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) * T.int64(8) + T.int64(2)], T.float16(0.0))) * ks_1[head * 131072], T.Cast("float32", T.if_then_else(tile < 128, kc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) * T.int64(8) + T.int64(3)], T.float16(0.0))) * ks_1[head * 131072])
                    score_1[0] = score_1[0] + T.call_extern("float32", "dot", left, right)
                    left_1: T.float32x4 = T.call_extern("float32x4", "vec4<f32>", buffer_9[0], buffer_11[0], buffer_13[0], buffer_15[0])
                    right_1: T.float32x4 = T.call_extern("float32x4", "vec4<f32>", T.Cast("float32", T.if_then_else(tile < 128, kc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) * T.int64(8) + T.int64(4)], T.float16(0.0))) * ks_1[head * 131072], T.Cast("float32", T.if_then_else(tile < 128, kc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) * T.int64(8) + T.int64(5)], T.float16(0.0))) * ks_1[head * 131072], T.Cast("float32", T.if_then_else(tile < 128, kc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) * T.int64(8) + T.int64(6)], T.float16(0.0))) * ks_1[head * 131072], T.Cast("float32", T.if_then_else(tile < 128, kc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) * T.int64(8) + T.int64(7)], T.float16(0.0))) * ks_1[head * 131072])
                    score_1[0] = score_1[0] + T.call_extern("float32", "dot", left_1, right_1)
                for step in T.unroll(3):
                    value: T.float32 = score_1[0]
                    score_1[0] = score_1[0] + T.call_extern("float32", "subgroupShuffleXor", value, T.Cast("uint32", T.shift_right(8, step + 1)))
                score_1[0] = T.if_then_else(tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + tx % 64 // 8 < length, score_1[0] * T.float32(0.125), T.reinterpret("float32", T.uint32(2139095040)) * T.float32(-1.0))
                maximum_1[0] = T.max(maximum_1[0], score_1[0])
                if tx % 8 == 0:
                    scores_1[tx // 64 * 1025 + tile * 8 + tx % 64 // 8] = score_1[0]
        if wgpu_batch_head * 128 < length:
            for step in T.unroll(5):
                value: T.float32 = maximum_1[0]
                maximum_1[0] = T.max(maximum_1[0], T.call_extern("float32", "subgroupShuffleXor", value, T.Cast("uint32", T.shift_right(16, step))))
        if wgpu_batch_head * 128 < length:
            if tx % 32 == 0:
                reductions_1[tx // 32] = maximum_1[0]
        if wgpu_batch_head * 128 < length:
            T.tvm_storage_sync("shared")
        if wgpu_batch_head * 128 < length:
            maximum_1[0] = T.max(reductions_1[tx // 64 * 2], reductions_1[tx // 64 * 2 + 1])
        if wgpu_batch_head * 128 < length:
            for tile in range((T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 2):
                probability: T.float32 = T.exp(scores_1[tx // 64 * 1025 + tile * 64 + tx % 64] - maximum_1[0])
                scores_1[tx // 64 * 1025 + tile * 64 + tx % 64] = probability
                normalizer_1[0] = normalizer_1[0] + probability
        if wgpu_batch_head * 128 < length:
            for step in T.unroll(5):
                value: T.float32 = normalizer_1[0]
                normalizer_1[0] = normalizer_1[0] + T.call_extern("float32", "subgroupShuffleXor", value, T.Cast("uint32", T.shift_right(16, step)))
        if wgpu_batch_head * 128 < length:
            if tx % 32 == 0:
                reductions_1[tx // 32] = normalizer_1[0]
        if wgpu_batch_head * 128 < length:
            T.tvm_storage_sync("shared")
        if wgpu_batch_head * 128 < length:
            normalizer_1[0] = reductions_1[tx // 64 * 2] + reductions_1[tx // 64 * 2 + 1]
        if wgpu_batch_head * 128 < length:
            vc_1 = T.Buffer((67108864,), "float16", data=vc.data)
            vs_1 = T.Buffer((1048576,), data=vs.data)
            for tile in range((T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16):
                if T.bitwise_and(tile - (T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16 < 0, tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 < length):
                    buffer_17[0] = buffer_17[0] + scores_1[tx // 64 * 1025 + tile * 8] * (T.Cast("float32", T.if_then_else(tile < 128, vc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64)], T.float16(0.0))) * vs_1[head * 131072])
                if T.bitwise_and(tile - (T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16 < 0, tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + 1 < length):
                    buffer_19[0] = buffer_19[0] + scores_1[tx // 64 * 1025 + tile * 8 + 1] * (T.Cast("float32", T.if_then_else(tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 < 131071, vc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) + T.int64(64)], T.float16(0.0))) * vs_1[head * 131072])
                if T.bitwise_and(tile - (T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16 < 0, tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + 2 < length):
                    buffer_21[0] = buffer_21[0] + scores_1[tx // 64 * 1025 + tile * 8 + 2] * (T.Cast("float32", T.if_then_else(tile // 16 * 8192 + wgpu_batch_head * 64 + tile % 16 * 4 < 65535, vc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) + T.int64(128)], T.float16(0.0))) * vs_1[head * 131072])
                if T.bitwise_and(tile - (T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16 < 0, tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + 3 < length):
                    buffer_23[0] = buffer_23[0] + scores_1[tx // 64 * 1025 + tile * 8 + 3] * (T.Cast("float32", T.if_then_else(tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 < 131069, vc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) + T.int64(192)], T.float16(0.0))) * vs_1[head * 131072])
                if T.bitwise_and(tile - (T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16 < 0, tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + 4 < length):
                    buffer_17[0] = buffer_17[0] + scores_1[tx // 64 * 1025 + tile * 8 + 4] * (T.Cast("float32", T.if_then_else(tile // 16 * 4096 + wgpu_batch_head * 32 + tile % 16 * 2 < 32767, vc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) + T.int64(256)], T.float16(0.0))) * vs_1[head * 131072])
                if T.bitwise_and(tile - (T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16 < 0, tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + 5 < length):
                    buffer_19[0] = buffer_19[0] + scores_1[tx // 64 * 1025 + tile * 8 + 5] * (T.Cast("float32", T.if_then_else(tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 < 131067, vc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) + T.int64(320)], T.float16(0.0))) * vs_1[head * 131072])
                if T.bitwise_and(tile - (T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16 < 0, tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + 6 < length):
                    buffer_21[0] = buffer_21[0] + scores_1[tx // 64 * 1025 + tile * 8 + 6] * (T.Cast("float32", T.if_then_else(tile // 16 * 8192 + wgpu_batch_head * 64 + tile % 16 * 4 < 65533, vc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) + T.int64(384)], T.float16(0.0))) * vs_1[head * 131072])
                if T.bitwise_and(tile - (T.max(length - wgpu_batch_head * 128, 0) + 16383) // 16384 * 16 < 0, tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 + 7 < length):
                    buffer_23[0] = buffer_23[0] + scores_1[tx // 64 * 1025 + tile * 8 + 7] * (T.Cast("float32", T.if_then_else(tile // 16 * 16384 + wgpu_batch_head * 128 + tile % 16 * 8 < 131065, vc_1[T.Cast("int64", head) * T.int64(8388608) + T.Cast("int64", tile) // T.int64(16) * T.int64(1048576) + T.Cast("int64", wgpu_batch_head) * T.int64(8192) + T.Cast("int64", tile) % T.int64(16) * T.int64(512) + T.Cast("int64", tx) % T.int64(64) + T.int64(448)], T.float16(0.0))) * vs_1[head * 131072])
        if wgpu_batch_head * 128 < length:
            other: T.float32 = buffer_17[0]
            result_1[0] = result_1[0] + other
            other_1: T.float32 = buffer_19[0]
            result_1[0] = result_1[0] + other_1
            other_2: T.float32 = buffer_21[0]
            result_1[0] = result_1[0] + other_2
            other_3: T.float32 = buffer_23[0]
            result_1[0] = result_1[0] + other_3
        out_1 = T.Buffer((270336,), data=out.data)
        out_1[head * 33792 + tx // 64 * 8448 + wgpu_batch_head * 66 + tx % 64] = result_1[0]
        if tx % 64 == 0:
            out_1[head * 33792 + tx // 64 * 8448 + wgpu_batch_head * 66 + 64] = maximum_1[0]
        if tx % 64 == 0:
            out_1[head * 33792 + tx // 64 * 8448 + wgpu_batch_head * 66 + 65] = normalizer_1[0]
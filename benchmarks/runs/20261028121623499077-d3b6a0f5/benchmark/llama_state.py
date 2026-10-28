"""Pinned b11946 Windows C API for real-prefill, state-restored decode tests.

The ABI is from include/llama.h at 8ca4060b7e30e76baaa5670ee1ec1b760b7f8552.
Refuse other libraries before calling functions that return structs by value.
"""

import ctypes as C
import hashlib
import os
from pathlib import Path
import time
import numpy as np

LLAMA_DLL_SHA256 = "5032cde7cafd0787e3287ec05c67a6c510f635e85b99e4ff077847da9027a9d9"
P, I, U, F, B, Z = C.c_void_p, C.c_int32, C.c_uint32, C.c_float, C.c_bool, C.c_size_t


class ModelParams(C.Structure):
    _fields_ = (
        [(n, P) for n in ("devices", "tensor_buft_overrides")]
        + [
            (n, I)
            for n in (
                "n_gpu_layers",
                "split_mode",
                "load_mode",
                "lazy_mode",
                "main_gpu",
            )
        ]
        + [
            (n, P)
            for n in (
                "tensor_split",
                "progress_callback",
                "progress_callback_user_data",
                "kv_overrides",
            )
        ]
        + [
            (n, B)
            for n in (
                "vocab_only",
                "check_tensors",
                "use_extra_bufts",
                "no_host",
                "no_alloc",
                "load_mtp",
            )
        ]
    )


class ContextParams(C.Structure):
    _fields_ = (
        [
            (n, U)
            for n in (
                "n_ctx",
                "n_batch",
                "n_ubatch",
                "n_seq_max",
                "n_rs_seq",
                "n_outputs_max",
                "n_outputs_max_per_seq",
            )
        ]
        + [
            (n, I)
            for n in (
                "n_threads",
                "n_threads_batch",
                "ctx_type",
                "rope_scaling_type",
                "pooling_type",
                "attention_type",
                "flash_attn_type",
            )
        ]
        + [
            (n, F)
            for n in (
                "rope_freq_base",
                "rope_freq_scale",
                "yarn_ext_factor",
                "yarn_attn_factor",
                "yarn_beta_fast",
                "yarn_beta_slow",
            )
        ]
        + [
            ("yarn_orig_ctx", U),
            ("defrag_thold", F),
            ("cb_eval", P),
            ("cb_eval_user_data", P),
            ("type_k", I),
            ("type_v", I),
            ("moe_cache_size", Z),
            ("abort_callback", P),
            ("abort_callback_data", P),
        ]
        + [
            (n, B)
            for n in (
                "embeddings",
                "offload_kqv",
                "no_perf",
                "op_offload",
                "swa_full",
                "kv_unified",
            )
        ]
        + [("samplers", P), ("n_samplers", Z), ("ctx_other", P)]
    )


class Batch(C.Structure):
    _fields_ = [("n_tokens", I)] + [
        (n, P) for n in ("token", "embd", "pos", "n_seq_id", "seq_id", "logits")
    ]


class Perf(C.Structure):
    _fields_ = [
        (n, C.c_double) for n in ("t_start_ms", "t_load_ms", "t_p_eval_ms", "t_eval_ms")
    ] + [
        (n, I)
        for n in (
            "n_p_eval",
            "n_eval",
            "n_reused",
            "n_reallocs_same",
            "n_reallocs_diff",
        )
    ]


class LlamaState:
    def __init__(
        self,
        checkpoint,
        binary_directory,
        *,
        context=131072,
        batch=2048,
        ubatch=512,
        threads=2,
    ):
        directory = Path(binary_directory).resolve()
        library = directory / "llama.dll"
        if os.name != "nt" or C.sizeof(P) != 8:
            raise RuntimeError("this pinned ABI targets 64-bit Windows")
        if hashlib.sha256(library.read_bytes()).hexdigest() != LLAMA_DLL_SHA256:
            raise ValueError(
                "llama.dll differs from the pinned official b11946 Vulkan binary"
            )
        assert (
            C.sizeof(ModelParams),
            C.sizeof(ContextParams),
            C.sizeof(Batch),
            C.sizeof(Perf),
        ) == (80, 168, 56, 56)
        self._dll_directory = os.add_dll_directory(str(directory))
        self.libs = [
            C.CDLL(str(directory / n))
            for n in ("ggml-base.dll", "ggml.dll", "llama.dll")
        ]
        self.functions = {}
        declarations = [
            ("ggml_backend_load_all_from_path", None, [C.c_char_p]),
            ("ggml_backend_dev_count", Z, []),
            ("ggml_backend_dev_get", P, [Z]),
            ("ggml_backend_dev_name", C.c_char_p, [P]),
            ("ggml_backend_dev_description", C.c_char_p, [P]),
            ("llama_backend_init", None, []),
            ("llama_backend_free", None, []),
            ("llama_model_default_params", ModelParams, []),
            ("llama_context_default_params", ContextParams, []),
            ("llama_model_load_from_file", P, [C.c_char_p, ModelParams]),
            ("llama_init_from_model", P, [P, ContextParams]),
            ("llama_free", None, [P]),
            ("llama_model_free", None, [P]),
            ("llama_n_ctx", U, [P]),
            ("llama_get_memory", P, [P]),
            ("llama_memory_clear", None, [P, B]),
            ("llama_batch_get_one", Batch, [C.POINTER(I), I]),
            ("llama_decode", I, [P, Batch]),
            ("llama_synchronize", None, [P]),
            ("llama_sampler_init_greedy", P, []),
            ("llama_sampler_sample", I, [P, P, I]),
            ("llama_sampler_reset", None, [P]),
            ("llama_sampler_free", None, [P]),
            ("llama_state_get_size", Z, [P]),
            ("llama_state_get_data", Z, [P, P, Z]),
            ("llama_state_set_data", Z, [P, P, Z]),
            ("llama_perf_context", Perf, [P]),
            ("llama_perf_context_reset", None, [P]),
            ("llama_get_logits_ith", C.POINTER(F), [P, I]),
            ("llama_model_get_vocab", P, [P]),
            ("llama_vocab_n_tokens", I, [P]),
        ]
        for name, restype, argtypes in declarations:
            function = next(
                (getattr(lib, name) for lib in self.libs if hasattr(lib, name)), None
            )
            if function is None:
                raise RuntimeError("missing pinned C API symbol " + name)
            function.restype, function.argtypes = restype, argtypes
            self.functions[name] = function
        self.call("ggml_backend_load_all_from_path", str(directory).encode())
        self.call("llama_backend_init")
        devices = [
            self.call("ggml_backend_dev_get", i)
            for i in range(self.call("ggml_backend_dev_count"))
        ]
        device = next(
            (d for d in devices if self.call("ggml_backend_dev_name", d) == b"Vulkan0"),
            None,
        )
        if device is None:
            raise RuntimeError("Vulkan0 absent from pinned C API")
        self.device = self.call("ggml_backend_dev_description", device).decode()
        self._devices = (P * 2)(device, None)
        mp = self.call("llama_model_default_params")
        mp.devices = C.cast(self._devices, P)
        mp.n_gpu_layers, mp.split_mode, mp.main_gpu = 99, 0, 0
        self.model = self.call(
            "llama_model_load_from_file", str(Path(checkpoint).resolve()).encode(), mp
        )
        if not self.model:
            raise RuntimeError("llama model load failed")
        cp = self.call("llama_context_default_params")
        cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max = context, batch, ubatch, 1
        cp.n_rs_seq, cp.n_threads, cp.n_threads_batch = 0, threads, threads
        cp.flash_attn_type, cp.type_k, cp.type_v = 1, 1, 1
        cp.no_perf, cp.offload_kqv, cp.op_offload = False, True, True
        self.configuration = {
            n: getattr(cp, n)
            for n in (
                "n_ctx",
                "n_batch",
                "n_ubatch",
                "n_seq_max",
                "n_rs_seq",
                "n_threads",
                "n_threads_batch",
                "flash_attn_type",
                "type_k",
                "type_v",
                "offload_kqv",
                "op_offload",
                "kv_unified",
                "swa_full",
            )
        }
        self.context = self.call("llama_init_from_model", self.model, cp)
        if not self.context or self.call("llama_n_ctx", self.context) != context:
            raise RuntimeError("llama context allocation failed or differs")
        self.sampler = self.call("llama_sampler_init_greedy")
        self.vocab = self.call(
            "llama_vocab_n_tokens", self.call("llama_model_get_vocab", self.model)
        )

    def call(self, name, *args):
        return self.functions[name](*args)

    def clear(self):
        self.call(
            "llama_memory_clear", self.call("llama_get_memory", self.context), True
        )
        self.call("llama_sampler_reset", self.sampler)
        self.call("llama_perf_context_reset", self.context)

    def decode(self, tokens):
        values = np.ascontiguousarray(tokens, dtype="int32")
        batch = self.call(
            "llama_batch_get_one", values.ctypes.data_as(C.POINTER(I)), len(values)
        )
        status = self.call("llama_decode", self.context, batch)
        if status:
            raise RuntimeError(f"llama_decode returned {status}")

    def sample(self):
        return int(self.call("llama_sampler_sample", self.sampler, self.context, -1))

    def prefill(self, tokens, *, progress=None):
        started = time.perf_counter()
        chunk = self.configuration["n_batch"]
        for begin in range(0, len(tokens), chunk):
            self.decode(tokens[begin : begin + chunk])
            if progress:
                progress(min(begin + chunk, len(tokens)), time.perf_counter() - started)
        first = self.sample()
        self.call("llama_synchronize", self.context)
        return first, time.perf_counter() - started

    def capture(self):
        self.call("llama_synchronize", self.context)
        size = self.call("llama_state_get_size", self.context)
        data = C.create_string_buffer(size)
        used = self.call("llama_state_get_data", self.context, data, size)
        if not 0 < used <= size:
            raise RuntimeError("invalid llama state capture")
        return data, used, self.sample()

    def restore(self, state):
        data, size, first = state
        if self.call("llama_state_set_data", self.context, data, size) != size:
            raise RuntimeError("incomplete llama state restore")
        self.call("llama_sampler_reset", self.sampler)
        self.call("llama_perf_context_reset", self.context)
        return first

    def generate(self, count=128, *, first=None):
        # b11946 restores recurrent/KV memory but leaves the last output row
        # unchanged. Keep the prefill's greedy seed explicitly with the state.
        output = [self.sample() if first is None else int(first)]
        for _ in range(count - 1):
            self.decode([output[-1]])
            output.append(self.sample())
        self.call("llama_synchronize", self.context)
        perf = self.call("llama_perf_context", self.context)
        return output, {n: getattr(perf, n) for n, _ in Perf._fields_}

    def finite_logits(self):
        values = np.ctypeslib.as_array(
            self.call("llama_get_logits_ith", self.context, -1), shape=(self.vocab,)
        )
        return bool(np.isfinite(values).all())

    def close(self):
        self.call("llama_sampler_free", self.sampler)
        self.call("llama_free", self.context)
        self.call("llama_model_free", self.model)
        self.call("llama_backend_free")
        self._dll_directory.close()

// CPU-only llama.cpp token-step baseline. Compile against the pinned headers.
#include "llama.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static std::vector<llama_token> read_tokens(const std::string & directory, int prefix) {
    std::ifstream input(directory + "/tokens-" + std::to_string(prefix) + ".bin", std::ios::binary);
    std::vector<llama_token> tokens(prefix);
    if (!input.read(reinterpret_cast<char *>(tokens.data()), prefix * sizeof(llama_token))) {
        throw std::runtime_error("cannot read complete prompt token file");
    }
    return tokens;
}

struct Sample {
    double prefill, decode;
    std::vector<llama_token> output;
};

static Sample measure(llama_context * ctx, llama_sampler * sampler,
                      const std::vector<llama_token> & input, int generated,
                      const std::string & logits_file = "") {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_sampler_reset(sampler);
    llama_synchronize(ctx);
    auto start = Clock::now();
    const int chunk = 512;
    llama_batch batch = llama_batch_init(chunk, 0, 1);
    for (int begin = 0; begin < static_cast<int>(input.size()); begin += chunk) {
        batch.n_tokens = std::min(chunk, static_cast<int>(input.size()) - begin);
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = input[begin + i];
            batch.pos[i] = begin + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = begin + i + 1 == static_cast<int>(input.size());
        }
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("prefill failed");
    }
    llama_synchronize(ctx);
    Sample result;
    result.output.push_back(llama_sampler_sample(sampler, ctx, -1));
    result.prefill = elapsed(start);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits = llama_get_logits_ith(ctx, -1);
    for (int i = 0; i < vocab; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite prefill logits");
    }
    if (!logits_file.empty()) {
        std::ofstream out(logits_file, std::ios::binary);
        out.write(reinterpret_cast<const char *>(logits), vocab * sizeof(float));
    }
    start = Clock::now();
    for (int step = 0; step < generated - 1; ++step) {
        batch.n_tokens = 1;
        batch.token[0] = result.output.back();
        batch.pos[0] = static_cast<int>(input.size()) + step;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("decode failed");
        result.output.push_back(llama_sampler_sample(sampler, ctx, -1));
    }
    llama_synchronize(ctx);
    result.decode = elapsed(start);
    llama_batch_free(batch);
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s MODEL INPUT_DIR THREADS BATCH_THREADS OUTPUT PREFIX...\n", argv[0]);
        return 2;
    }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    ggml_threadpool_t pool = nullptr, pool_batch = nullptr;
    try {
        const int threads = std::stoi(argv[3]), batch_threads = std::stoi(argv[4]);
        const int generated = std::stoi(argv[5]);
        if (threads < 1 || batch_threads < 1 || generated < 2) throw std::runtime_error("invalid arguments");
        llama_backend_init();
        std::fprintf(stderr, "CPU FEATURES: %s\n", llama_print_system_info());
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 8320;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        cp.n_threads = threads;
        cp.n_threads_batch = batch_threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = false;
        cp.op_offload = false;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        auto pool_params = ggml_threadpool_params_default(threads);
        auto batch_params = ggml_threadpool_params_default(batch_threads);
        for (auto item : {std::make_pair("AKAS_LLAMA_DECODE_CPUS", &pool_params),
                          std::make_pair("AKAS_LLAMA_BATCH_CPUS", &batch_params)}) {
            if (const char * value = std::getenv(item.first)) {
                auto mask = std::strtoull(value, nullptr, 16);
                for (int cpu = 0; cpu < 64; ++cpu) item.second->cpumask[cpu] = (mask >> cpu) & 1;
                item.second->strict_cpu = true;
            }
        }
        pool = ggml_threadpool_new(&pool_params);
        pool_batch = ggml_threadpool_new(&batch_params);
        if (!pool || !pool_batch) throw std::runtime_error("thread pool creation failed");
        llama_attach_threadpool(ctx, pool, pool_batch);
        sampler = llama_sampler_init_greedy();
        measure(ctx, sampler, read_tokens(argv[2], 32), 3);
        std::fprintf(stderr, "WARMUP COMPLETE\n");
        for (int argument = 6; argument < argc; ++argument) {
            const int prefix = std::stoi(argv[argument]);
            if (prefix + generated > static_cast<int>(cp.n_ctx)) throw std::runtime_error("context exhausted");
            std::fprintf(stderr, "MEASURE CPU %d t=%d tb=%d\n", prefix, threads, batch_threads);
            auto tokens = read_tokens(argv[2], prefix);
            auto result = measure(ctx, sampler, tokens, generated,
                std::string(argv[2]) + "/logits-" + std::to_string(prefix) + ".f32");
            std::printf("RESULT {\"prefix_tokens\":%d,\"generated_tokens\":%d,\"threads\":%d,\"batch_threads\":%d,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,\"prefill_tokens_per_second\":%.6f,\"decode_tokens_per_second\":%.6f,\"output_ids\":[",
                prefix, generated, threads, batch_threads, result.prefill, result.decode,
                prefix / result.prefill, (generated - 1) / result.decode);
            for (size_t i = 0; i < result.output.size(); ++i) {
                std::printf("%s%d", i ? "," : "", result.output[i]);
            }
            std::printf("]}\n");
            std::fflush(stdout);
        }
        llama_sampler_free(sampler);
        llama_free(ctx);
        ggml_threadpool_free(pool);
        ggml_threadpool_free(pool_batch);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        if (sampler) llama_sampler_free(sampler);
        if (ctx) llama_free(ctx);
        if (pool) ggml_threadpool_free(pool);
        if (pool_batch) ggml_threadpool_free(pool_batch);
        if (model) llama_model_free(model);
        llama_backend_free();
        return 1;
    }
}

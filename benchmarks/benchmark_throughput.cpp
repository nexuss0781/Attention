#include "smao_phase1/core/exact_decomposition.h"
#include "smao_phase1/core/phase1_forward.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace smao {
namespace {

template <typename T>
std::vector<T> generate_random_vector(size_t count, unsigned int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<T> distribution(0.0, 1.0);
    std::vector<T> result(count);
    for (T& value : result) value = distribution(gen);
    return result;
}

bool benchmark_decomposition(bool full) {
    std::cout << "\n=== Decomposition Throughput Benchmark ===\n";
    bool valid_run = true;
    const auto configs = full
        ? std::vector<std::pair<size_t, size_t>>{{10000, 64}, {100000, 64}, {1000000, 64}}
        : std::vector<std::pair<size_t, size_t>>{{10000, 64}, {100000, 64}};

    for (const auto& [n, d] : configs) {
        auto q = generate_random_vector<f32>(n * d, 1);
        auto k = generate_random_vector<f32>(n * d, 2);
        std::vector<f32> query_scales(n), key_weights(n);
        f32 sigma_squared = 0.0f;

        const auto start = std::chrono::steady_clock::now();
        const Status status = exact_decomposition(
            q.data(), k.data(), n, d, d, 1e-6f, -80.0f, 80.0f,
            query_scales.data(), key_weights.data(), &sigma_squared);
        const auto end = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration<double>(end - start).count();
        const double tokens_per_second = static_cast<double>(n) / std::max(seconds, 1e-12);
        const bool valid = status == Status::OK && std::isfinite(sigma_squared) &&
                           query_scales.front() > 0.0f && key_weights.front() > 0.0f;
        const bool target_met = tokens_per_second >= 5.0e6;
        valid_run = valid_run && valid;
        std::cout << "n=" << n << ", d=" << d << ": " << seconds * 1000.0
                  << " ms, " << tokens_per_second / 1.0e6
                  << " M tokens/sec [" << (valid ? "VALID" : "INVALID")
                  << ", target=" << (target_met ? "met" : "not-met") << "]\n";
    }
    return valid_run;
}

bool benchmark_end_to_end(bool full) {
    std::cout << "\n=== End-to-End Phase 1 Benchmark ===\n";
    bool valid_run = true;
    const auto configs = full
        ? std::vector<std::pair<size_t, size_t>>{{10000, 64}, {100000, 64}, {1000000, 64}}
        : std::vector<std::pair<size_t, size_t>>{{10000, 64}, {100000, 64}};

    for (const auto& [n, d] : configs) {
        auto q = generate_random_vector<f32>(n * d, 1);
        auto k = generate_random_vector<f32>(n * d, 2);
        auto l_values = generate_random_vector<f32>(d * d, 3);
        std::vector<f32> v = q;
        std::vector<f32> l(d * d, 0.0f);
        std::mt19937 gen(4);
        std::uniform_real_distribution<f32> diagonal(0.5f, 1.5f);
        for (size_t i = 0; i < d; ++i) {
            l[i * d + i] = diagonal(gen);
            for (size_t j = 0; j < i; ++j) l[i * d + j] = l_values[i * d + j] * 0.3f;
        }

        Phase1Input input;
        input.q = q.data(); input.k = k.data(); input.v = v.data(); input.l = l.data();
        input.n = n; input.d = d; input.d_v = d; input.precision = Precision::F32;

        Phase1Output output;
        phase1_forward_into(input, output);
        const auto start = std::chrono::steady_clock::now();
        phase1_forward_into(input, output);
        const auto end = std::chrono::steady_clock::now();
        const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
        const double tokens_per_second = static_cast<double>(n) / std::max(milliseconds / 1000.0, 1e-12);
        const bool valid = output.status == Status::OK && check_frozen_gate_criteria(output);
        const bool target_met = milliseconds <= 200.0;
        valid_run = valid_run && valid;
        std::cout << "n=" << n << ", d=" << d << ": " << milliseconds << " ms, "
                  << tokens_per_second / 1.0e6 << " M tokens/sec ["
                  << (valid ? "VALID" : "INVALID") << ", target="
                  << (target_met ? "met" : "not-met") << "]\n";
        phase1_output_release(output);
    }
    return valid_run;
}

} // namespace
} // namespace smao

int main(int argc, char** argv) {
    bool full = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--full") {
            full = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: attention_benchmarks [--full]\n"
                         "  --full  include the n=1,000,000 end-to-end case\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argument << "\n";
            return 2;
        }
    }

    std::cout << "Attention Phase 1 Performance Benchmarks\n";
    const bool decomposition_valid = smao::benchmark_decomposition(full);
    const bool end_to_end_valid = smao::benchmark_end_to_end(full);
    std::cout << "\nBenchmarks complete\n";
    return decomposition_valid && end_to_end_valid ? 0 : 1;
}

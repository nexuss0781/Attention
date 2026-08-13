#include "smao_phase1/core/exact_decomposition.h"
#include "smao_phase1/core/phase1_forward.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace smao {
namespace {

struct BenchmarkOptions {
    bool full = false;
    bool strict = false;
    int repeats = 5;
};

struct SampleStats {
    double minimum_ms = 0.0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double maximum_ms = 0.0;
};

template <typename T>
std::vector<T> generate_random_vector(size_t count, unsigned int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<T> distribution(0.0, 1.0);
    std::vector<T> result(count);
    for (T& value : result) value = distribution(gen);
    return result;
}

double percentile(const std::vector<double>& sorted_values, double fraction) {
    if (sorted_values.empty()) return 0.0;
    const double position = fraction * static_cast<double>(sorted_values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sorted_values.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return sorted_values[lower] + weight * (sorted_values[upper] - sorted_values[lower]);
}

SampleStats summarize(std::vector<double> samples_ms) {
    std::sort(samples_ms.begin(), samples_ms.end());
    SampleStats result;
    if (samples_ms.empty()) return result;
    result.minimum_ms = samples_ms.front();
    result.maximum_ms = samples_ms.back();
    result.mean_ms = 0.0;
    for (double sample : samples_ms) result.mean_ms += sample;
    result.mean_ms /= static_cast<double>(samples_ms.size());
    result.p50_ms = percentile(samples_ms, 0.50);
    result.p95_ms = percentile(samples_ms, 0.95);
    return result;
}

void print_stats(const SampleStats& stats, double tokens_per_second, const char* unit) {
    std::cout << std::fixed << std::setprecision(3)
              << "mean=" << stats.mean_ms << " ms, "
              << "p50=" << stats.p50_ms << " ms, "
              << "p95=" << stats.p95_ms << " ms, "
              << "min=" << stats.minimum_ms << " ms, "
              << "max=" << stats.maximum_ms << " ms, "
              << "p95-throughput=" << tokens_per_second / 1.0e6 << " " << unit;
}

bool target_applies(size_t n, size_t d) {
    return n == 1000000 && d == 64;
}

bool benchmark_decomposition(const BenchmarkOptions& options) {
    std::cout << "\n=== Decomposition Throughput Benchmark ===\n";
    bool valid_run = true;
    bool strict_targets_pass = true;
    const auto configs = options.full
        ? std::vector<std::pair<size_t, size_t>>{{10000, 64}, {100000, 64}, {1000000, 64}}
        : std::vector<std::pair<size_t, size_t>>{{10000, 64}, {100000, 64}};

    for (const auto& [n, d] : configs) {
        auto q = generate_random_vector<f32>(n * d, 1);
        auto k = generate_random_vector<f32>(n * d, 2);
        std::vector<f32> query_scales(n), key_weights(n);
        f32 sigma_squared = 0.0f;
        bool samples_valid = true;

        exact_decomposition(q.data(), k.data(), n, d, d, 1e-6f, -80.0f, 80.0f,
                            query_scales.data(), key_weights.data(), &sigma_squared);
        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<size_t>(options.repeats));
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            const Status status = exact_decomposition(
                q.data(), k.data(), n, d, d, 1e-6f, -80.0f, 80.0f,
                query_scales.data(), key_weights.data(), &sigma_squared);
            const auto end = std::chrono::steady_clock::now();
            samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            samples_valid = samples_valid && status == Status::OK &&
                            std::isfinite(sigma_squared) && query_scales.front() > 0.0f &&
                            key_weights.front() > 0.0f;
        }

        const SampleStats stats = summarize(std::move(samples_ms));
        const double p95_tokens_per_second = static_cast<double>(n) /
            std::max(stats.p95_ms / 1000.0, 1e-12);
        const bool target_relevant = target_applies(n, d);
        const bool target_met = !target_relevant || p95_tokens_per_second >= 5.0e6;
        valid_run = valid_run && samples_valid;
        strict_targets_pass = strict_targets_pass && target_met;

        std::cout << "n=" << n << ", d=" << d << ": ";
        print_stats(stats, p95_tokens_per_second, "M tokens/sec");
        std::cout << " [" << (samples_valid ? "VALID" : "INVALID")
                  << ", target=" << (target_relevant ? (target_met ? "met" : "not-met") : "n/a")
                  << "]\n";
    }
    return valid_run && (!options.strict || strict_targets_pass);
}

bool benchmark_end_to_end(const BenchmarkOptions& options) {
    std::cout << "\n=== End-to-End Phase 1 Benchmark ===\n";
    bool valid_run = true;
    bool strict_targets_pass = true;
    const auto configs = options.full
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
        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<size_t>(options.repeats));
        bool samples_valid = true;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            const Status status = phase1_forward_into(input, output);
            const auto end = std::chrono::steady_clock::now();
            samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            samples_valid = samples_valid && status == Status::OK && check_frozen_gate_criteria(output);
        }

        const SampleStats stats = summarize(std::move(samples_ms));
        const bool target_relevant = target_applies(n, d);
        const bool target_met = !target_relevant || stats.p95_ms <= 200.0;
        const double p95_tokens_per_second = static_cast<double>(n) /
            std::max(stats.p95_ms / 1000.0, 1e-12);
        valid_run = valid_run && samples_valid;
        strict_targets_pass = strict_targets_pass && target_met;

        std::cout << "n=" << n << ", d=" << d << ": ";
        print_stats(stats, p95_tokens_per_second, "M tokens/sec");
        std::cout << " [" << (samples_valid ? "VALID" : "INVALID")
                  << ", target=" << (target_relevant ? (target_met ? "met" : "not-met") : "n/a")
                  << "]\n";
        phase1_output_release(output);
    }
    return valid_run && (!options.strict || strict_targets_pass);
}

} // namespace
} // namespace smao

int main(int argc, char** argv) {
    smao::BenchmarkOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--full") {
            options.full = true;
        } else if (argument == "--strict") {
            options.strict = true;
        } else if (argument == "--repeats") {
            if (i + 1 >= argc) {
                std::cerr << "--repeats requires a positive integer\n";
                return 2;
            }
            try {
                options.repeats = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "--repeats requires a positive integer\n";
                return 2;
            }
            if (options.repeats <= 0 || options.repeats > 100) {
                std::cerr << "--repeats must be between 1 and 100\n";
                return 2;
            }
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: attention_benchmarks [--full] [--strict] [--repeats N]\n"
                         "  --full       include the n=1,000,000 end-to-end case\n"
                         "  --strict     return failure when the documented p95 target is missed\n"
                         "  --repeats N  collect N warmed samples, default 5\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argument << "\n";
            return 2;
        }
    }

    std::cout << "Attention Phase 1 Performance Benchmarks\n"
              << "repeats=" << options.repeats << ", strict=" << (options.strict ? "on" : "off") << "\n";
    const bool decomposition_valid = smao::benchmark_decomposition(options);
    const bool end_to_end_valid = smao::benchmark_end_to_end(options);
    std::cout << "\nBenchmarks complete\n";
    return decomposition_valid && end_to_end_valid ? 0 : 1;
}

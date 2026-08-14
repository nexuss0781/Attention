#include "attention/linear_attention.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {

struct Statistics {
    double mean_ms = 0.0;
    double p95_ms = 0.0;
};

Statistics summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    Statistics result;
    if (samples.empty()) return result;
    for (double sample : samples) result.mean_ms += sample;
    result.mean_ms /= static_cast<double>(samples.size());
    const double position = 0.95 * static_cast<double>(samples.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, samples.size() - 1);
    const double weight = position - static_cast<double>(lower);
    result.p95_ms = samples[lower] + weight * (samples[upper] - samples[lower]);
    return result;
}

bool run_case(std::size_t sequence_length, std::size_t hidden_size) {
    attention::Tensor query;
    attention::Tensor key;
    attention::Tensor value;
    if (!query.reset({1, sequence_length, hidden_size}) ||
        !key.reset({1, sequence_length, hidden_size}) ||
        !value.reset({1, sequence_length, hidden_size})) {
        return false;
    }
    std::mt19937 generator(1234);
    std::normal_distribution<float> distribution(0.0f, 0.1f);
    for (std::size_t index = 0; index < query.size(); ++index) {
        query.data()[index] = distribution(generator);
        key.data()[index] = distribution(generator);
        value.data()[index] = distribution(generator);
    }

    attention::LinearCausalAttention attention;
    if (!attention.reset(sequence_length, hidden_size)) return false;
    attention::Tensor output;
    if (!attention.forward(query, key, value, output) || !output.all_finite()) return false;

    std::vector<double> samples;
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        if (!attention.forward(query, key, value, output)) return false;
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    const Statistics statistics = summarize(std::move(samples));
    const double tokens_per_second = static_cast<double>(sequence_length) /
        std::max(statistics.p95_ms / 1000.0, 1e-12);
    std::cout << "n=" << sequence_length << ", d=" << hidden_size
              << ": mean=" << std::fixed << std::setprecision(3) << statistics.mean_ms
              << " ms, p95=" << statistics.p95_ms
              << " ms, p95-throughput=" << tokens_per_second / 1.0e6
              << " M tokens/sec, state-bytes=" << attention.state_bytes(1) << '\n';
    return true;
}

} // namespace

int main() {
    constexpr std::size_t hidden_size = 8;
    for (const std::size_t sequence_length : {10000u, 100000u, 1000000u}) {
        if (!run_case(sequence_length, hidden_size)) return 1;
    }
    return 0;
}

#include "attention/linear_attention.h"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr std::size_t total_tokens = 1000000;
    constexpr std::size_t chunk_tokens = 4096;
    constexpr std::size_t hidden_size = 8;

    attention::LinearCausalAttention operator_state;
    if (!operator_state.reset(total_tokens, hidden_size)) return 1;
    attention::LinearAttentionState stream;
    if (!operator_state.create_stream(1, stream)) return 1;

    attention::Tensor query;
    attention::Tensor key;
    attention::Tensor value;
    if (!query.reset({1, chunk_tokens, hidden_size}) ||
        !key.reset({1, chunk_tokens, hidden_size}) ||
        !value.reset({1, chunk_tokens, hidden_size})) return 1;
    std::mt19937 generator(7);
    std::normal_distribution<float> distribution(0.0f, 0.1f);
    for (std::size_t index = 0; index < query.size(); ++index) {
        query.data()[index] = distribution(generator);
        key.data()[index] = distribution(generator);
        value.data()[index] = distribution(generator);
    }

    attention::Tensor output;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t offset = 0; offset < total_tokens; offset += chunk_tokens) {
        const std::size_t length = std::min(chunk_tokens, total_tokens - offset);
        if (length == chunk_tokens) {
            if (!stream.append(query, key, value, output)) return 1;
            continue;
        }
        attention::Tensor tail_query;
        attention::Tensor tail_key;
        attention::Tensor tail_value;
        if (!tail_query.reset({1, length, hidden_size}) ||
            !tail_key.reset({1, length, hidden_size}) ||
            !tail_value.reset({1, length, hidden_size})) return 1;
        for (std::size_t index = 0; index < length * hidden_size; ++index) {
            tail_query.data()[index] = query.data()[index];
            tail_key.data()[index] = key.data()[index];
            tail_value.data()[index] = value.data()[index];
        }
        if (!stream.append(tail_query, tail_key, tail_value, output)) return 1;
    }
    const auto end = std::chrono::steady_clock::now();
    if (!output.all_finite() || stream.tokens_processed() != total_tokens) return 1;
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    const double tokens_per_second = static_cast<double>(total_tokens) /
        std::max(milliseconds / 1000.0, 1e-12);
    std::cout << "logical_tokens=" << total_tokens
              << ", chunk_tokens=" << chunk_tokens
              << ", hidden=" << hidden_size
              << ", elapsed_ms=" << std::fixed << std::setprecision(3) << milliseconds
              << ", throughput=" << tokens_per_second / 1.0e6
              << " M tokens/sec, state_bytes=" << stream.state_bytes() << '\n';
    return 0;
}

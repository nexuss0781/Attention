/**
 * @file benchmark_throughput.cpp
 * @brief Performance benchmarks for Phase 1
 *
 * Metrics Specifications:
 * - Decomposition throughput: >= 5 x 10^6 tok/s
 * - End-to-end latency: <= 200 ms for n=10^6, d=64
 * - Memory allocation: zero O(n^2) allocations
 */

#include "smao_phase1/core/phase1_forward.h"
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

namespace smao {

// Helper to generate random data
template<typename T>
std::vector<T> generate_random_vector(size_t n, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(0.0, 1.0);
    
    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = static_cast<T>(dist(gen));
    }
    return result;
}

// Benchmark exact decomposition
void benchmark_decomposition() {
    std::cout << "\n=== Decomposition Throughput Benchmark ===\n";
    
    std::vector<std::pair<size_t, size_t>> configs = {
        {10000, 64},
        {100000, 64},
        {1000000, 64},
    };
    
    for (const auto& [n, d] : configs) {
        // Generate data
        auto q = generate_random_vector<f32>(n * d, 1);
        auto k = generate_random_vector<f32>(n * d, 2);
        
        std::vector<f32> query_scales(n);
        std::vector<f32> key_weights(n);
        f32 sigma_squared;
        
        // Warmup
        exact_decomposition(
            q.data(), k.data(), n, d, d,
            1e-6f, -80.0f, 80.0f,
            query_scales.data(), key_weights.data(), &sigma_squared
        );
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        const int num_iterations = (n >= 1000000) ? 1 : 10;
        for (int iter = 0; iter < num_iterations; ++iter) {
            exact_decomposition(
                q.data(), k.data(), n, d, d,
                1e-6f, -80.0f, 80.0f,
                query_scales.data(), key_weights.data(), &sigma_squared
            );
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        f32 tokens_per_sec = static_cast<f32>(n * num_iterations) / (duration / 1e6f);
        
        std::cout << "n=" << n << ", d=" << d << ": "
                  << duration / num_iterations / 1000.0f << " ms/iter, "
                  << tokens_per_sec / 1e6f << " M tokens/sec"
                  << (tokens_per_sec >= 5e6f ? " [PASS]" : " [FAIL]")
                  << std::endl;
    }
}

// Benchmark end-to-end Phase 1
void benchmark_end_to_end() {
    std::cout << "\n=== End-to-End Phase 1 Benchmark ===\n";
    
    std::vector<std::pair<size_t, size_t>> configs = {
        {10000, 64},
        {100000, 64},
        {500000, 64},
    };
    
    for (const auto& [n, d] : configs) {
        // Generate data
        auto q = generate_random_vector<f32>(n * d, 1);
        auto k = generate_random_vector<f32>(n * d, 2);
        auto l_vec = generate_random_vector<f32>(d * d, 3);
        
        // Make L lower triangular with positive diagonal
        std::vector<f32> l(d * d, 0.0f);
        std::mt19937 gen(4);
        std::uniform_real_distribution<f32> diag_dist(0.5f, 1.5f);
        
        for (size_t i = 0; i < d; ++i) {
            l[i * d + i] = diag_dist(gen);
            for (size_t j = 0; j < i; ++j) {
                l[i * d + j] = l_vec[i * d + j] * 0.3f;
            }
        }
        
        // Setup input
        Phase1Input input;
        input.q = q.data();
        input.k = k.data();
        input.v = q.data();
        input.l = l.data();
        input.n = n;
        input.d = d;
        input.d_v = d;
        input.precision = Precision::F32;
        
        // Warmup
        Phase1Output output = phase1_forward(input);
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        const int num_iterations = (n >= 500000) ? 1 : 3;
        for (int iter = 0; iter < num_iterations; ++iter) {
            output = phase1_forward(input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        f32 latency_ms = static_cast<f32>(duration) / num_iterations;
        f32 tokens_per_sec = static_cast<f32>(n * num_iterations) / (duration / 1000.0f);
        
        std::cout << "n=" << n << ", d=" << d << ": "
                  << latency_ms << " ms, "
                  << tokens_per_sec / 1e6f << " M tokens/sec"
                  << (latency_ms <= 200.0f ? " [PASS]" : " [CHECK]")
                  << std::endl;
        
        // Verify output
        EXPECT_EQ(output.status, Status::OK);
        EXPECT_LE(output.condition_number, 1e4f);
    }
}

} // namespace

// Main benchmark runner
int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "SMAO Phase 1 Performance Benchmarks\n";
    std::cout << "========================================\n";
    
    smao::benchmark_decomposition();
    smao::benchmark_end_to_end();
    
    std::cout << "\n========================================\n";
    std::cout << "Benchmarks Complete\n";
    std::cout << "========================================\n";
    
    return 0;
}

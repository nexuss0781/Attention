#include "phase1_forward.h"
#include "exact_decomposition.h"
#include "metric_assembly.h"
#include "whiten_coordinates.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>

using namespace smao::phase1;

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_benchmark(const std::string& name, double elapsed_ms, size_t tokens, int d) {
    double throughput = (tokens / elapsed_ms) * 1000.0;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << name << ": " << elapsed_ms << " ms";
    std::cout << " | Throughput: " << throughput / 1e6 << " M tokens/sec";
    std::cout << " | Tokens: " << tokens << ", Dim: " << d << std::endl;
}

int main() {
    print_header("Phase 1 Performance Benchmarks");

    // ========================================================================
    // Benchmark 1: Decomposition Throughput
    // ========================================================================
    {
        print_header("Benchmark 1: ExactDecomposition Throughput");

        int d = 64;
        size_t n = 1000000;

        std::cout << "Parameters: n=" << n << ", d=" << d << std::endl;

        MatrixXf Q = MatrixXf::Random(n, d);
        MatrixXf K = MatrixXf::Random(n, d);

        VectorXf a, w;
        float sigma2;

        auto start = std::chrono::high_resolution_clock::now();

        Status status = exact_decomposition(Q, K, d, a, w, sigma2);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        if (status != Status::OK) {
            std::cout << "Error: " << static_cast<int>(status) << std::endl;
            return 1;
        }

        print_benchmark("ExactDecomposition", elapsed, n, d);

        double target_throughput = 5e6;  // 5M tokens/sec
        double achieved = (n / elapsed) * 1000.0;
        std::cout << "Target throughput: " << target_throughput / 1e6 << " M tokens/sec" << std::endl;
        std::cout << "Status: " << (achieved >= target_throughput ? "PASS" : "FAIL") << std::endl;
    }

    // ========================================================================
    // Benchmark 2: Whitening Throughput
    // ========================================================================
    {
        print_header("Benchmark 2: WhitenCoordinates Throughput");

        int d = 64;
        size_t n = 1000000;

        std::cout << "Parameters: n=" << n << ", d=" << d << std::endl;

        MatrixXf X = MatrixXf::Random(n, d);
        MatrixXf W = MatrixXf::Random(d, d);
        MatrixXf X_whitened;

        auto start = std::chrono::high_resolution_clock::now();

        Status status = whiten_coordinates(X, W, X_whitened);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        if (status != Status::OK) {
            std::cout << "Error: " << static_cast<int>(status) << std::endl;
            return 1;
        }

        print_benchmark("WhitenCoordinates", elapsed, n, d);

        double target_throughput = 2e6;  // 2M tokens/sec
        double achieved = (n / elapsed) * 1000.0;
        std::cout << "Target throughput: " << target_throughput / 1e6 << " M tokens/sec" << std::endl;
        std::cout << "Status: " << (achieved >= target_throughput ? "PASS" : "FAIL") << std::endl;
    }

    // ========================================================================
    // Benchmark 3: Metric Assembly
    // ========================================================================
    {
        print_header("Benchmark 3: MetricAssembly");

        int d = 64;

        std::cout << "Parameters: d=" << d << std::endl;

        MatrixXf L = MatrixXf::Random(d, d).triangularView<Eigen::Lower>();
        for (int i = 0; i < d; ++i) {
            L(i, i) = std::abs(L(i, i)) + 1.0f;
        }

        MatrixXf M, W;
        float kappa;

        // Warm-up
        metric_assembly(L, M, W, kappa);

        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < 1000; ++iter) {
            metric_assembly(L, M, W, kappa);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_elapsed = total_elapsed / 1000.0;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Average time per call: " << avg_elapsed << " ms (over 1000 iterations)" << std::endl;
        std::cout << "Total time: " << total_elapsed << " ms" << std::endl;

        double target_ms = 2.0;
        std::cout << "Target: < " << target_ms << " ms" << std::endl;
        std::cout << "Status: " << (avg_elapsed <= target_ms ? "PASS" : "FAIL") << std::endl;
    }

    // ========================================================================
    // Benchmark 4: End-to-End Phase 1
    // ========================================================================
    {
        print_header("Benchmark 4: End-to-End Phase1Forward");

        int d = 64;
        size_t n = 1000000;
        int d_v = 64;

        std::cout << "Parameters: n=" << n << ", d=" << d << ", d_v=" << d_v << std::endl;

        MatrixXf Q = MatrixXf::Random(n, d);
        MatrixXf K = MatrixXf::Random(n, d);
        MatrixXf V = MatrixXf::Random(n, d_v);
        MatrixXf L = MatrixXf::Identity(d, d);

        auto start = std::chrono::high_resolution_clock::now();

        Phase1Output output = phase1_forward(Q, K, V, L, d);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        if (output.status != Status::OK) {
            std::cout << "Error: " << static_cast<int>(output.status) << std::endl;
            return 1;
        }

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Total time: " << elapsed << " ms" << std::endl;

        double target_ms = 200.0;
        std::cout << "Target: < " << target_ms << " ms" << std::endl;
        std::cout << "Status: " << (elapsed <= target_ms ? "PASS" : "FAIL") << std::endl;

        // Output diagnostics
        std::cout << "\nOutput diagnostics:" << std::endl;
        std::cout << "  Condition number: " << output.condition_number << std::endl;
        std::cout << "  Bandwidth: " << output.bandwidth << std::endl;
        std::cout << "  Query scales range: [" 
                  << output.query_scales.minCoeff() << ", " 
                  << output.query_scales.maxCoeff() << "]" << std::endl;
    }

    print_header("Benchmarks Complete");

    return 0;
}

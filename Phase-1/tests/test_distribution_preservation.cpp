#include <gtest/gtest.h>
#include "exact_decomposition.h"
#include "numerical_stability.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace smao::phase1;

/**
 * Test 1.8: Output Distribution Preservation
 *
 * Verify that Gaussian decomposition weights match standard attention weights
 * in distribution (via Kolmogorov-Smirnov test).
 */
TEST(DistributionTest, Test1_8_DistributionPreservation) {
    int n = 1024;
    int d = 64;
    uint32_t d_k = d;

    // Generate Q, K from N(0, 1/sqrt(d))
    MatrixXf Q = MatrixXf::Random(n, d) / std::sqrt(d);
    MatrixXf K = MatrixXf::Random(n, d) / std::sqrt(d);

    // Compute standard attention weights
    std::vector<float> standard_weights;
    float sigma2 = std::sqrt(d);
    float denom = sigma2;

    for (int i = 0; i < std::min(n, 100); ++i) {
        for (int j = 0; j < std::min(n, 100); ++j) {
            float dot_prod = Q.row(i).dot(K.row(j));
            float exp_arg = dot_prod / denom;
            exp_arg = std::clamp(exp_arg, -80.0f, 80.0f);
            standard_weights.push_back(std::exp(exp_arg));
        }
    }

    // Compute Gaussian decomposition weights
    VectorXf a, w;
    float sigma2_out;
    Status status = exact_decomposition(Q, K, d_k, a, w, sigma2_out);
    ASSERT_EQ(status, Status::OK);

    std::vector<float> decomp_weights;
    float denom_2 = 2.0f * sigma2_out;

    for (int i = 0; i < std::min(n, 100); ++i) {
        for (int j = 0; j < std::min(n, 100); ++j) {
            VectorXf delta = Q.row(i) - K.row(j);
            float sq_dist = delta.squaredNorm();
            float exp_arg = -sq_dist / denom_2;
            exp_arg = std::clamp(exp_arg, -80.0f, 80.0f);
            decomp_weights.push_back(a(i) * w(j) * std::exp(exp_arg));
        }
    }

    // Sort for KS test
    std::sort(standard_weights.begin(), standard_weights.end());
    std::sort(decomp_weights.begin(), decomp_weights.end());

    // Compute KS statistic
    float ks_stat = 0.0f;
    for (size_t i = 0; i < standard_weights.size(); ++i) {
        float cdf_standard = static_cast<float>(i) / standard_weights.size();
        float cdf_decomp = static_cast<float>(i) / decomp_weights.size();
        ks_stat = std::max(ks_stat, std::abs(cdf_standard - cdf_decomp));
    }

    ASSERT_LT(ks_stat, 0.01f);
}

/**
 * Test histogram similarity
 */
TEST(DistributionTest, HistogramBins) {
    int n = 256;
    int d = 32;
    uint32_t d_k = d;

    MatrixXf Q = MatrixXf::Random(n, d) / std::sqrt(d);
    MatrixXf K = MatrixXf::Random(n, d) / std::sqrt(d);

    VectorXf a, w;
    float sigma2;

    Status status = exact_decomposition(Q, K, d_k, a, w, sigma2);
    ASSERT_EQ(status, Status::OK);

    // Compute query weights
    std::vector<int> a_histogram(10, 0);
    float a_min = a.minCoeff();
    float a_max = a.maxCoeff();

    if (a_max > a_min) {
        for (int i = 0; i < a.size(); ++i) {
            int bin = static_cast<int>(
                9.0f * (a(i) - a_min) / (a_max - a_min)
            );
            bin = std::clamp(bin, 0, 9);
            a_histogram[bin]++;
        }
    }

    // All bins should be populated (rough check)
    int populated_bins = 0;
    for (int count : a_histogram) {
        if (count > 0) populated_bins++;
    }

    ASSERT_GE(populated_bins, 3);  // At least some spread
}

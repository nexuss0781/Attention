#include "anisotropic_distance.h"
#include "numerical_stability.h"

namespace smao::phase1 {

f32 anisotropic_squared_distance(
    const VectorXf& q,
    const VectorXf& k,
    const MatrixXf& M
) {
    // Algorithm 4: AnisotropicDistancePrimitive
    // d_M^2(q, k) = (q - k)^T * M * (q - k)
    
    VectorXf delta = q - k;
    
    // Compute (q - k)^T * M
    VectorXf M_delta = M * delta;
    
    // Compute inner product
    f32 dist_sq = kahan_dot_product(delta, M_delta);
    
    return dist_sq;
}

f32 anisotropic_squared_distance_whitened(
    const VectorXf& q_whitened,
    const VectorXf& k_whitened
) {
    // Equivalent via whitened coordinates
    // d_M^2(q, k) = ||W*q - W*k||_2^2
    
    VectorXf delta_whitened = q_whitened - k_whitened;
    
    // Squared Euclidean distance
    f32 dist_sq = kahan_squared_norm(delta_whitened);
    
    return dist_sq;
}

}  // namespace smao::phase1

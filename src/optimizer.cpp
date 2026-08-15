#include "attention/optimizer.h"

#include <cmath>
#include <limits>

namespace attention {

bool SgdOptimizer::step(ParameterStore& parameters, std::string* error) const {
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0f) {
        if (error != nullptr) *error = "learning rate must be finite and positive";
        return false;
    }
    if (!std::isfinite(gradient_clip_norm_) || gradient_clip_norm_ < 0.0f) {
        if (error != nullptr) *error = "gradient clip norm must be finite and nonnegative";
        return false;
    }
    float gradient_scale = 1.0f;
    if (gradient_clip_norm_ > 0.0f) {
        const float norm = parameters.gradient_l2_norm();
        if (!std::isfinite(norm)) {
            if (error != nullptr) *error = "gradient norm is nonfinite";
            return false;
        }
        if (norm > gradient_clip_norm_) {
            gradient_scale = gradient_clip_norm_ / norm;
        }
    }
    for (Parameter& parameter : parameters.parameters()) {
        if (!parameter.value.all_finite() || !parameter.gradient.all_finite()) {
            if (error != nullptr) *error = "parameter or gradient is nonfinite: " + parameter.name;
            return false;
        }
        for (std::size_t index = 0; index < parameter.value.size(); ++index) {
            const float updated = parameter.value.data()[index] -
                learning_rate_ * gradient_scale * parameter.gradient.data()[index];
            if (!std::isfinite(updated)) {
                if (error != nullptr) *error = "SGD update became nonfinite: " + parameter.name;
                return false;
            }
            parameter.value.data()[index] = updated;
        }
    }
    return true;
}

} // namespace attention

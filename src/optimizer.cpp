#include "attention/optimizer.h"

#include <cmath>
#include <limits>
#include <new>

namespace attention {
namespace {

bool valid_clip(float clip) {
    return std::isfinite(clip) && clip >= 0.0f;
}

float gradient_scale_for(const ParameterStore& parameters, float clip, std::string* error) {
    if (!valid_clip(clip)) {
        if (error != nullptr) *error = "gradient clip norm must be finite and nonnegative";
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (clip == 0.0f) return 1.0f;
    const float norm = parameters.gradient_l2_norm();
    if (!std::isfinite(norm)) {
        if (error != nullptr) *error = "gradient norm is nonfinite";
        return std::numeric_limits<float>::quiet_NaN();
    }
    return norm > clip ? clip / norm : 1.0f;
}

bool validate_parameters(const ParameterStore& parameters, std::string* error) {
    for (const Parameter& parameter : parameters.parameters()) {
        if (!parameter.value.all_finite() || !parameter.gradient.all_finite()) {
            if (error != nullptr) *error = "parameter or gradient is nonfinite: " + parameter.name;
            return false;
        }
    }
    return true;
}

} // namespace

bool SgdOptimizer::step(ParameterStore& parameters, std::string* error) const {
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0f) {
        if (error != nullptr) *error = "learning rate must be finite and positive";
        return false;
    }
    if (!validate_parameters(parameters, error)) return false;
    const float gradient_scale = gradient_scale_for(parameters, gradient_clip_norm_, error);
    if (!std::isfinite(gradient_scale)) return false;
    for (Parameter& parameter : parameters.parameters()) {
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
    if (error != nullptr) error->clear();
    return true;
}

bool AdamWOptimizer::initialize_moments(const ParameterStore& parameters, std::string* error) const {
    if (first_moments_.size() == parameters.parameters().size() &&
        second_moments_.size() == parameters.parameters().size()) {
        bool shapes_match = true;
        for (std::size_t index = 0; index < parameters.parameters().size(); ++index) {
            const std::size_t size = parameters.parameters()[index].value.size();
            shapes_match = shapes_match && first_moments_[index].size() == size &&
                           second_moments_[index].size() == size;
        }
        if (shapes_match) return true;
    }
    try {
        first_moments_.clear();
        second_moments_.clear();
        first_moments_.reserve(parameters.parameters().size());
        second_moments_.reserve(parameters.parameters().size());
        for (const Parameter& parameter : parameters.parameters()) {
            first_moments_.emplace_back(parameter.value.size(), 0.0f);
            second_moments_.emplace_back(parameter.value.size(), 0.0f);
        }
    } catch (const std::bad_alloc&) {
        if (error != nullptr) *error = "AdamW moment allocation failed";
        return false;
    }
    return true;
}

bool AdamWOptimizer::step(ParameterStore& parameters, std::string* error) const {
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0f) {
        if (error != nullptr) *error = "learning rate must be finite and positive";
        return false;
    }
    if (!std::isfinite(beta1_) || beta1_ < 0.0f || beta1_ >= 1.0f ||
        !std::isfinite(beta2_) || beta2_ < 0.0f || beta2_ >= 1.0f ||
        !std::isfinite(epsilon_) || epsilon_ <= 0.0f ||
        !std::isfinite(weight_decay_) || weight_decay_ < 0.0f) {
        if (error != nullptr) *error = "AdamW hyperparameters are invalid";
        return false;
    }
    if (step_count_ == std::numeric_limits<std::uint64_t>::max()) {
        if (error != nullptr) *error = "AdamW step counter overflow";
        return false;
    }
    if (!validate_parameters(parameters, error)) return false;
    const float gradient_scale = gradient_scale_for(parameters, gradient_clip_norm_, error);
    if (!std::isfinite(gradient_scale)) return false;
    if (!initialize_moments(parameters, error)) return false;

    const float next_beta1_power = beta1_power_ * beta1_;
    const float next_beta2_power = beta2_power_ * beta2_;
    if (!std::isfinite(next_beta1_power) || !std::isfinite(next_beta2_power)) {
        if (error != nullptr) *error = "AdamW bias-correction state became nonfinite";
        return false;
    }
    const float correction1 = 1.0f - next_beta1_power;
    const float correction2 = 1.0f - next_beta2_power;
    if (!(correction1 > 0.0f) || !(correction2 > 0.0f)) {
        if (error != nullptr) *error = "AdamW bias-correction denominator is invalid";
        return false;
    }
    const float decay_factor = 1.0f - learning_rate_ * weight_decay_;
    if (!std::isfinite(decay_factor)) {
        if (error != nullptr) *error = "AdamW decay factor is nonfinite";
        return false;
    }

    for (std::size_t parameter_index = 0; parameter_index < parameters.parameters().size(); ++parameter_index) {
        Parameter& parameter = parameters.parameters()[parameter_index];
        auto& first = first_moments_[parameter_index];
        auto& second = second_moments_[parameter_index];
        for (std::size_t index = 0; index < parameter.value.size(); ++index) {
            const float gradient = gradient_scale * parameter.gradient.data()[index];
            first[index] = beta1_ * first[index] + (1.0f - beta1_) * gradient;
            second[index] = beta2_ * second[index] + (1.0f - beta2_) * gradient * gradient;
            const float first_hat = first[index] / correction1;
            const float second_hat = second[index] / correction2;
            const float updated = decay_factor * parameter.value.data()[index] -
                learning_rate_ * first_hat / (std::sqrt(second_hat) + epsilon_);
            if (!std::isfinite(updated)) {
                if (error != nullptr) *error = "AdamW update became nonfinite: " + parameter.name;
                return false;
            }
            parameter.value.data()[index] = updated;
        }
    }
    step_count_ += 1;
    beta1_power_ = next_beta1_power;
    beta2_power_ = next_beta2_power;
    if (error != nullptr) error->clear();
    return true;
}

} // namespace attention

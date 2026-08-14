#include "attention/normalization.h"

#include <cmath>
#include <new>
#include <utility>

namespace attention {
namespace {

constexpr double kEpsilon = 1e-5;

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_parameter_tensor(const Tensor& tensor) {
    return tensor.valid() && tensor.rank() == 1 &&
           tensor.data_type() == TensorDataType::F32 &&
           tensor.device() == TensorDevice::CPU &&
           tensor.layout() == TensorLayout::RowMajor;
}

} // namespace

Normalization::Normalization(std::string prefix) : prefix_(std::move(prefix)) {}

bool Normalization::register_parameters(const TransformerConfig& config,
                                       ParameterStore& parameters,
                                       std::string* error) {
    if (!config.validate(error)) return false;
    if (config.hidden_size == 0) {
        set_error(error, "normalization hidden size must be positive");
        return false;
    }
    const std::string weight_name = prefix_ + ".weight";
    const std::string bias_name = prefix_ + ".bias";
    if (parameters.find(weight_name) != nullptr || parameters.find(bias_name) != nullptr) {
        set_error(error, "normalization parameter name already exists");
        return false;
    }
    if (!parameters.add(weight_name, {config.hidden_size}, error) ||
        !parameters.add(bias_name, {config.hidden_size}, error)) {
        return false;
    }
    hidden_size_ = config.hidden_size;
    if (error != nullptr) error->clear();
    return true;
}

bool Normalization::forward(const Tensor& input,
                            const ParameterStore& parameters,
                            Tensor& output,
                            std::string* error) const {
    if (hidden_size_ == 0) {
        set_error(error, "normalization is not initialized");
        return false;
    }
    if (!input.valid() || input.rank() != 3 ||
        input.data_type() != TensorDataType::F32 ||
        input.device() != TensorDevice::CPU ||
        input.layout() != TensorLayout::RowMajor) {
        set_error(error, "normalization input must be a valid F32 CPU row-major rank-3 tensor");
        return false;
    }
    const auto& shape = input.shape();
    if (shape[0] == 0 || shape[1] == 0 || shape[2] != hidden_size_) {
        set_error(error, "normalization input shape does not match hidden size");
        return false;
    }
    if (!input.all_finite()) {
        set_error(error, "normalization input contains NaN or infinity");
        return false;
    }

    const Parameter* weight = parameters.find(prefix_ + ".weight");
    const Parameter* bias = parameters.find(prefix_ + ".bias");
    if (weight == nullptr || bias == nullptr ||
        weight->value.shape() != std::vector<std::size_t>{hidden_size_} ||
        bias->value.shape() != std::vector<std::size_t>{hidden_size_} ||
        !valid_parameter_tensor(weight->value) || !valid_parameter_tensor(bias->value) ||
        !weight->value.all_finite() || !bias->value.all_finite()) {
        set_error(error, "normalization parameters are missing, invalid, or nonfinite");
        return false;
    }

    if (!output.reset(shape, TensorDataType::F32, TensorDevice::CPU, error)) return false;
    const std::size_t token_count = shape[0] * shape[1];
    for (std::size_t token = 0; token < token_count; ++token) {
        const std::size_t offset = token * hidden_size_;
        double mean = 0.0;
        for (std::size_t hidden = 0; hidden < hidden_size_; ++hidden) {
            mean += static_cast<double>(input.data()[offset + hidden]);
        }
        mean /= static_cast<double>(hidden_size_);
        double variance = 0.0;
        for (std::size_t hidden = 0; hidden < hidden_size_; ++hidden) {
            const double centered = static_cast<double>(input.data()[offset + hidden]) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(hidden_size_);
        const double inverse_std = 1.0 / std::sqrt(variance + kEpsilon);
        for (std::size_t hidden = 0; hidden < hidden_size_; ++hidden) {
            const double normalized =
                (static_cast<double>(input.data()[offset + hidden]) - mean) * inverse_std;
            output.data()[offset + hidden] = static_cast<float>(
                normalized * static_cast<double>(weight->value.data()[hidden]) +
                static_cast<double>(bias->value.data()[hidden]));
        }
    }
    if (!output.all_finite()) {
        set_error(error, "normalization output contains NaN or infinity");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

std::size_t Normalization::hidden_size() const noexcept { return hidden_size_; }
float Normalization::epsilon() const noexcept { return static_cast<float>(kEpsilon); }
const std::string& Normalization::prefix() const noexcept { return prefix_; }

} // namespace attention

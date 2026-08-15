#include "attention/feed_forward.h"

#include <Eigen/Dense>

#include <cmath>
#include <limits>
#include <new>
#include <vector>
#include <utility>

namespace attention {
namespace {

constexpr double kSqrtTwoOverPi = 0.7978845608028654;
constexpr double kGeluCoefficient = 0.044715;

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_parameter_tensor(const Tensor& tensor) {
    return tensor.valid() && (tensor.rank() == 1 || tensor.rank() == 2) &&
           tensor.data_type() == TensorDataType::F32 &&
           tensor.device() == TensorDevice::CPU &&
           tensor.layout() == TensorLayout::RowMajor;
}

float activate(float value, Activation activation) {
    const double x = static_cast<double>(value);
    if (activation == Activation::SiLU) {
        if (x >= 0.0) {
            return static_cast<float>(x / (1.0 + std::exp(-x)));
        }
        const double exponential = std::exp(x);
        return static_cast<float>(x * exponential / (1.0 + exponential));
    }
    const double cubic = x * x * x;
    const double inner = kSqrtTwoOverPi * (x + kGeluCoefficient * cubic);
    return static_cast<float>(0.5 * x * (1.0 + std::tanh(inner)));
}

} // namespace

FeedForward::FeedForward(std::string prefix) : prefix_(std::move(prefix)) {}

bool FeedForward::register_parameters(const TransformerConfig& config,
                                      ParameterStore& parameters,
                                      std::string* error) {
    if (!config.validate(error)) return false;
    if (config.hidden_size == 0 || config.feed_forward_size == 0) {
        set_error(error, "feed-forward dimensions must be positive");
        return false;
    }
    const std::string up_weight = prefix_ + ".up.weight";
    const std::string up_bias = prefix_ + ".up.bias";
    const std::string down_weight = prefix_ + ".down.weight";
    const std::string down_bias = prefix_ + ".down.bias";
    if (parameters.find(up_weight) != nullptr || parameters.find(up_bias) != nullptr ||
        parameters.find(down_weight) != nullptr || parameters.find(down_bias) != nullptr) {
        set_error(error, "feed-forward parameter name already exists");
        return false;
    }
    if (!parameters.add(up_weight, {config.hidden_size, config.feed_forward_size}, error) ||
        !parameters.add(up_bias, {config.feed_forward_size}, error) ||
        !parameters.add(down_weight, {config.feed_forward_size, config.hidden_size}, error) ||
        !parameters.add(down_bias, {config.hidden_size}, error)) {
        return false;
    }
    hidden_size_ = config.hidden_size;
    feed_forward_size_ = config.feed_forward_size;
    activation_ = config.activation;
    if (error != nullptr) error->clear();
    return true;
}

bool FeedForward::forward(const Tensor& input,
                          const ParameterStore& parameters,
                          Tensor& output,
                          std::string* error) const {
    if (hidden_size_ == 0 || feed_forward_size_ == 0) {
        set_error(error, "feed-forward is not initialized");
        return false;
    }
    if (!input.valid() || input.rank() != 3 ||
        input.data_type() != TensorDataType::F32 ||
        input.device() != TensorDevice::CPU ||
        input.layout() != TensorLayout::RowMajor) {
        set_error(error, "feed-forward input must be a valid F32 CPU row-major rank-3 tensor");
        return false;
    }
    const auto& shape = input.shape();
    const std::size_t batch_size = shape[0];
    const std::size_t sequence_length = shape[1];
    if (batch_size == 0 || sequence_length == 0 || shape[2] != hidden_size_) {
        set_error(error, "feed-forward input shape does not match hidden size");
        return false;
    }
    if (!input.all_finite()) {
        set_error(error, "feed-forward input contains NaN or infinity");
        return false;
    }

    const Parameter* up_weight = parameters.find(prefix_ + ".up.weight");
    const Parameter* up_bias = parameters.find(prefix_ + ".up.bias");
    const Parameter* down_weight = parameters.find(prefix_ + ".down.weight");
    const Parameter* down_bias = parameters.find(prefix_ + ".down.bias");
    if (up_weight == nullptr || up_bias == nullptr || down_weight == nullptr || down_bias == nullptr ||
        up_weight->value.shape() != std::vector<std::size_t>{hidden_size_, feed_forward_size_} ||
        up_bias->value.shape() != std::vector<std::size_t>{feed_forward_size_} ||
        down_weight->value.shape() != std::vector<std::size_t>{feed_forward_size_, hidden_size_} ||
        down_bias->value.shape() != std::vector<std::size_t>{hidden_size_}) {
        set_error(error, "feed-forward parameters are missing or have invalid shapes");
        return false;
    }
    if (!valid_parameter_tensor(up_weight->value) || !valid_parameter_tensor(up_bias->value) ||
        !valid_parameter_tensor(down_weight->value) || !valid_parameter_tensor(down_bias->value) ||
        !up_weight->value.all_finite() || !up_bias->value.all_finite() ||
        !down_weight->value.all_finite() || !down_bias->value.all_finite()) {
        set_error(error, "feed-forward parameters are invalid or nonfinite");
        return false;
    }

    if (!output.reset(shape, TensorDataType::F32, TensorDevice::CPU, error)) return false;
    try {
        using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using RowMajorVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;
        const auto token_count = batch_size * sequence_length;
        Eigen::Map<const RowMajorMatrix> input_map(
            input.data(), static_cast<Eigen::Index>(token_count), static_cast<Eigen::Index>(hidden_size_));
        Eigen::Map<const RowMajorMatrix> up_weight_map(
            up_weight->value.data(), static_cast<Eigen::Index>(hidden_size_), static_cast<Eigen::Index>(feed_forward_size_));
        Eigen::Map<const RowMajorVector> up_bias_map(
            up_bias->value.data(), static_cast<Eigen::Index>(feed_forward_size_));
        RowMajorMatrix expanded(static_cast<Eigen::Index>(token_count),
                                static_cast<Eigen::Index>(feed_forward_size_));
        expanded.noalias() = input_map * up_weight_map;
        expanded.rowwise() += up_bias_map.transpose();
        for (Eigen::Index index = 0; index < expanded.size(); ++index) {
            expanded.data()[index] = activate(expanded.data()[index], activation_);
        }
        Eigen::Map<const RowMajorMatrix> down_weight_map(
            down_weight->value.data(), static_cast<Eigen::Index>(feed_forward_size_), static_cast<Eigen::Index>(hidden_size_));
        Eigen::Map<RowMajorMatrix> output_map(
            output.data(), static_cast<Eigen::Index>(token_count), static_cast<Eigen::Index>(hidden_size_));
        Eigen::Map<const RowMajorVector> down_bias_map(
            down_bias->value.data(), static_cast<Eigen::Index>(hidden_size_));
        output_map.noalias() = expanded * down_weight_map;
        output_map.rowwise() += down_bias_map.transpose();
    } catch (const std::bad_alloc&) {
        set_error(error, "feed-forward workspace allocation failed");
        return false;
    }
    if (!output.all_finite()) {
        set_error(error, "feed-forward output contains NaN or infinity");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

std::size_t FeedForward::hidden_size() const noexcept { return hidden_size_; }
std::size_t FeedForward::feed_forward_size() const noexcept { return feed_forward_size_; }
Activation FeedForward::activation() const noexcept { return activation_; }
const std::string& FeedForward::prefix() const noexcept { return prefix_; }

} // namespace attention

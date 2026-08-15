#include "attention/qkv_projection.h"

#include <Eigen/Dense>

#include <cmath>
#include <limits>

namespace attention {
namespace {

bool multiply_fits(std::size_t a, std::size_t b) noexcept {
    return a == 0 || b <= std::numeric_limits<std::size_t>::max() / a;
}

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_input_tensor(const Tensor& input) {
    return input.valid() && input.rank() == 3 &&
           input.data_type() == TensorDataType::F32 &&
           input.device() == TensorDevice::CPU &&
           input.layout() == TensorLayout::RowMajor;
}

} // namespace

bool QKVProjection::register_parameters(const TransformerConfig& config,
                                        std::size_t layer_index,
                                        ParameterStore& parameters,
                                        std::string* error) {
    if (!config.validate(error)) return false;
    hidden_size_ = config.hidden_size;
    layer_index_ = layer_index;
    const std::string prefix = "layers." + std::to_string(layer_index_) + ".attention.";
    query_weight_name_ = prefix + "q_proj.weight";
    query_bias_name_ = prefix + "q_proj.bias";
    key_weight_name_ = prefix + "k_proj.weight";
    key_bias_name_ = prefix + "k_proj.bias";
    value_weight_name_ = prefix + "v_proj.weight";
    value_bias_name_ = prefix + "v_proj.bias";

    const std::vector<std::pair<std::string, std::vector<std::size_t>>> declarations = {
        {query_weight_name_, {hidden_size_, hidden_size_}},
        {query_bias_name_, {hidden_size_}},
        {key_weight_name_, {hidden_size_, hidden_size_}},
        {key_bias_name_, {hidden_size_}},
        {value_weight_name_, {hidden_size_, hidden_size_}},
        {value_bias_name_, {hidden_size_}},
    };
    for (const auto& [name, shape] : declarations) {
        (void)shape;
        if (parameters.find(name) != nullptr) {
            set_error(error, "QKV projection parameter already exists");
            hidden_size_ = 0;
            initialized_ = false;
            return false;
        }
    }
    for (const auto& [name, shape] : declarations) {
        if (!parameters.add(name, shape, error)) {
            hidden_size_ = 0;
            initialized_ = false;
            return false;
        }
    }
    initialized_ = true;
    if (error != nullptr) error->clear();
    return true;
}

bool QKVProjection::forward(const Tensor& input,
                            const ParameterStore& parameters,
                            QKVOutput& output,
                            std::string* error) const {
    if (!initialized_ || hidden_size_ == 0) {
        set_error(error, "QKV projection is not initialized");
        return false;
    }
    if (!valid_input_tensor(input)) {
        set_error(error, "QKV projection input must be a valid F32 CPU row-major rank-3 tensor");
        return false;
    }
    const auto& shape = input.shape();
    const std::size_t batch_size = shape[0];
    const std::size_t sequence_length = shape[1];
    if (batch_size == 0 || sequence_length == 0 || shape[2] != hidden_size_) {
        set_error(error, "QKV projection input shape does not match hidden size");
        return false;
    }
    if (!input.all_finite()) {
        set_error(error, "QKV projection input contains NaN or infinity");
        return false;
    }
    if (!multiply_fits(batch_size, sequence_length) ||
        !multiply_fits(batch_size * sequence_length, hidden_size_)) {
        set_error(error, "QKV projection output shape overflows size_t");
        return false;
    }

    const std::vector<std::size_t> matrix_shape{hidden_size_, hidden_size_};
    const std::vector<std::size_t> bias_shape{hidden_size_};
    const Parameter* query_weight = parameters.find(query_weight_name_);
    const Parameter* query_bias = parameters.find(query_bias_name_);
    const Parameter* key_weight = parameters.find(key_weight_name_);
    const Parameter* key_bias = parameters.find(key_bias_name_);
    const Parameter* value_weight = parameters.find(value_weight_name_);
    const Parameter* value_bias = parameters.find(value_bias_name_);
    if (query_weight == nullptr || query_bias == nullptr || key_weight == nullptr ||
        key_bias == nullptr || value_weight == nullptr || value_bias == nullptr ||
        query_weight->value.shape() != matrix_shape || key_weight->value.shape() != matrix_shape ||
        value_weight->value.shape() != matrix_shape || query_bias->value.shape() != bias_shape ||
        key_bias->value.shape() != bias_shape || value_bias->value.shape() != bias_shape) {
        set_error(error, "QKV projection parameters are missing or have the wrong shape");
        return false;
    }
    if (!query_weight->value.all_finite() || !query_bias->value.all_finite() ||
        !key_weight->value.all_finite() || !key_bias->value.all_finite() ||
        !value_weight->value.all_finite() || !value_bias->value.all_finite()) {
        set_error(error, "QKV projection parameters contain NaN or infinity");
        return false;
    }
    if (!output.query.reset(shape) || !output.key.reset(shape) || !output.value.reset(shape)) {
        set_error(error, "QKV projection output allocation failed");
        return false;
    }

    const std::size_t token_count = batch_size * sequence_length;
    using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowMajorVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    const auto project = [this, token_count](const Tensor& source,
                                             const Parameter& weight,
                                             const Parameter& bias,
                                             Tensor& destination) {
        Eigen::Map<const RowMajorMatrix> input_map(
            source.data(), static_cast<Eigen::Index>(token_count), static_cast<Eigen::Index>(hidden_size_));
        Eigen::Map<const RowMajorMatrix> weight_map(
            weight.value.data(), static_cast<Eigen::Index>(hidden_size_), static_cast<Eigen::Index>(hidden_size_));
        Eigen::Map<RowMajorMatrix> output_map(
            destination.data(), static_cast<Eigen::Index>(token_count), static_cast<Eigen::Index>(hidden_size_));
        output_map.noalias() = input_map * weight_map.transpose();
        const Eigen::Map<const RowMajorVector> bias_map(
            bias.value.data(), static_cast<Eigen::Index>(hidden_size_));
        output_map.rowwise() += bias_map.transpose();
    };
    project(input, *query_weight, *query_bias, output.query);
    project(input, *key_weight, *key_bias, output.key);
    project(input, *value_weight, *value_bias, output.value);
    if (!output.query.all_finite() || !output.key.all_finite() || !output.value.all_finite()) {
        set_error(error, "QKV projection output is not finite");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

std::size_t QKVProjection::hidden_size() const noexcept {
    return hidden_size_;
}

std::size_t QKVProjection::layer_index() const noexcept {
    return layer_index_;
}

bool QKVProjection::initialized() const noexcept {
    return initialized_;
}

} // namespace attention

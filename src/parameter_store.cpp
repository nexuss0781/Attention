#include "attention/parameter_store.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <utility>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_name(const std::string& name) {
    if (name.empty()) return false;
    for (const unsigned char character : name) {
        if (std::isspace(character)) return false;
    }
    return true;
}

float initialization_stddev(const Tensor& tensor) {
    const auto& shape = tensor.shape();
    if (shape.size() >= 2) {
        const double fan_in = static_cast<double>(shape[shape.size() - 2]);
        const double fan_out = static_cast<double>(shape.back());
        return static_cast<float>(std::sqrt(2.0 / (fan_in + fan_out)));
    }
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(shape.back())));
}

} // namespace

bool ParameterStore::add(std::string name,
                         const std::vector<std::size_t>& shape,
                         std::string* error) {
    if (!valid_name(name)) {
        set_error(error, "parameter name must be nonempty and contain no whitespace");
        return false;
    }
    if (find(name) != nullptr) {
        set_error(error, "parameter name already exists");
        return false;
    }

    Parameter parameter;
    parameter.name = std::move(name);
    std::string tensor_error;
    if (!parameter.value.reset(shape, TensorDataType::F32, TensorDevice::CPU, &tensor_error) ||
        !parameter.gradient.reset(shape, TensorDataType::F32, TensorDevice::CPU, &tensor_error)) {
        if (error != nullptr) *error = tensor_error;
        return false;
    }
    try {
        parameters_.push_back(std::move(parameter));
    } catch (...) {
        set_error(error, "parameter registration allocation failed");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool ParameterStore::initialize(std::uint64_t seed, std::string* error) {
    std::mt19937_64 generator(seed);
    for (Parameter& parameter : parameters_) {
        const float standard_deviation = initialization_stddev(parameter.value);
        if (!std::isfinite(standard_deviation) || !(standard_deviation > 0.0f)) {
            set_error(error, "parameter initialization scale is invalid");
            return false;
        }
        std::normal_distribution<float> distribution(0.0f, standard_deviation);
        for (std::size_t index = 0; index < parameter.value.size(); ++index) {
            parameter.value.data()[index] = distribution(generator);
            parameter.gradient.data()[index] = 0.0f;
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

void ParameterStore::clear_gradients() noexcept {
    for (Parameter& parameter : parameters_) parameter.gradient.fill(0.0f);
}

const Parameter* ParameterStore::find(const std::string& name) const noexcept {
    for (const Parameter& parameter : parameters_) {
        if (parameter.name == name) return &parameter;
    }
    return nullptr;
}

Parameter* ParameterStore::find(const std::string& name) noexcept {
    for (Parameter& parameter : parameters_) {
        if (parameter.name == name) return &parameter;
    }
    return nullptr;
}

std::vector<std::string> ParameterStore::names() const {
    std::vector<std::string> result;
    result.reserve(parameters_.size());
    for (const Parameter& parameter : parameters_) result.push_back(parameter.name);
    std::sort(result.begin(), result.end());
    return result;
}

std::size_t ParameterStore::size() const noexcept {
    return parameters_.size();
}

bool ParameterStore::all_finite() const noexcept {
    for (const Parameter& parameter : parameters_) {
        if (!parameter.value.all_finite() || !parameter.gradient.all_finite()) return false;
    }
    return true;
}

std::vector<Parameter>& ParameterStore::parameters() noexcept {
    return parameters_;
}

const std::vector<Parameter>& ParameterStore::parameters() const noexcept {
    return parameters_;
}

} // namespace attention

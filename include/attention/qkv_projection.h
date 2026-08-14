#ifndef ATTENTION_QKV_PROJECTION_H
#define ATTENTION_QKV_PROJECTION_H

#include "attention/parameter_store.h"
#include "attention/tensor.h"
#include "attention/transformer_config.h"

#include <cstddef>
#include <string>

namespace attention {

struct QKVOutput {
    Tensor query;
    Tensor key;
    Tensor value;
};

class QKVProjection {
public:
    bool register_parameters(const TransformerConfig& config,
                             std::size_t layer_index,
                             ParameterStore& parameters,
                             std::string* error = nullptr);

    bool forward(const Tensor& input,
                 const ParameterStore& parameters,
                 QKVOutput& output,
                 std::string* error = nullptr) const;

    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] std::size_t layer_index() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    std::size_t hidden_size_ = 0;
    std::size_t layer_index_ = 0;
    bool initialized_ = false;
    std::string query_weight_name_;
    std::string query_bias_name_;
    std::string key_weight_name_;
    std::string key_bias_name_;
    std::string value_weight_name_;
    std::string value_bias_name_;
};

} // namespace attention

#endif // ATTENTION_QKV_PROJECTION_H

#ifndef ATTENTION_LINEAR_ATTENTION_H
#define ATTENTION_LINEAR_ATTENTION_H

#include "attention/causal_mask.h"
#include "attention/tensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace attention {

class LinearAttentionState {
public:
    bool reset(std::size_t context_length,
               std::size_t batch_size,
               std::size_t hidden_size,
               float epsilon = 1e-6f,
               std::string* error = nullptr) noexcept;

    bool append(const Tensor& query,
                const Tensor& key,
                const Tensor& value,
                Tensor& output,
                std::string* error = nullptr);

    void clear() noexcept;
    [[nodiscard]] std::size_t context_length() const noexcept;
    [[nodiscard]] std::size_t batch_size() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] std::size_t tokens_processed() const noexcept;
    [[nodiscard]] std::size_t state_bytes() const noexcept;

private:
    std::size_t context_length_ = 0;
    std::size_t batch_size_ = 0;
    std::size_t hidden_size_ = 0;
    std::size_t tokens_processed_ = 0;
    float epsilon_ = 1e-6f;
    std::vector<double> state_;
    std::vector<double> normalizer_;
    std::vector<float> query_features_;
    std::vector<float> key_features_;
};

class LinearCausalAttention {
public:
    bool reset(std::size_t context_length,
               std::size_t hidden_size,
               float epsilon = 1e-6f,
               std::string* error = nullptr) noexcept;

    bool forward(const Tensor& query,
                 const Tensor& key,
                 const Tensor& value,
                 Tensor& output,
                 std::string* error = nullptr) const;

    [[nodiscard]] std::size_t context_length() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] std::size_t state_bytes(std::size_t batch_size) const noexcept;
    bool create_stream(std::size_t batch_size,
                       LinearAttentionState& state,
                       std::string* error = nullptr) const noexcept;

private:
    std::size_t context_length_ = 0;
    std::size_t hidden_size_ = 0;
    float epsilon_ = 1e-6f;
    CausalMask mask_;
};

} // namespace attention

#endif // ATTENTION_LINEAR_ATTENTION_H

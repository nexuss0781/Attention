#ifndef ATTENTION_TRANSFORMER_CONFIG_H
#define ATTENTION_TRANSFORMER_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace attention {

enum class Activation : std::uint8_t {
    GELU = 0,
    SiLU = 1,
};

enum class ModelPrecision : std::uint8_t {
    F32 = 0,
};

struct TransformerConfig {
    std::size_t vocabulary_size = 0;
    std::size_t context_length = 0;
    std::size_t layer_count = 0;
    std::size_t hidden_size = 0;
    std::size_t attention_head_count = 0;
    std::size_t feed_forward_size = 0;
    Activation activation = Activation::GELU;
    ModelPrecision precision = ModelPrecision::F32;
    bool tie_embeddings = true;
    bool causal = true;
    float dropout_probability = 0.0f;

    [[nodiscard]] bool validate(std::string* error = nullptr) const noexcept;
    [[nodiscard]] std::size_t head_size() const noexcept;
    [[nodiscard]] std::uint64_t parameter_count() const noexcept;
    [[nodiscard]] std::uint64_t parameter_bytes() const noexcept;
    [[nodiscard]] std::uint64_t estimated_activation_bytes(std::size_t batch_size = 1) const noexcept;
    [[nodiscard]] std::uint64_t estimated_inference_memory_bytes(
        std::size_t batch_size = 1,
        std::size_t resident_sequence_length = 0) const noexcept;
    bool serialize(std::string& output, std::string* error = nullptr) const noexcept;
    static bool deserialize(const std::string& serialized,
                            TransformerConfig& output,
                            std::string* error = nullptr) noexcept;
};

} // namespace attention

#endif // ATTENTION_TRANSFORMER_CONFIG_H

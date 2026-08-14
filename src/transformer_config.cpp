#include "attention/transformer_config.h"

#include <cmath>
#include <limits>

namespace attention {
namespace {

bool multiply_fits(std::uint64_t a, std::uint64_t b) noexcept {
    return a == 0 || b <= std::numeric_limits<std::uint64_t>::max() / a;
}

bool add_fits(std::uint64_t a, std::uint64_t b) noexcept {
    return b <= std::numeric_limits<std::uint64_t>::max() - a;
}

void set_error(std::string* error, const char* message) noexcept {
    if (error != nullptr) *error = message;
}

} // namespace

bool TransformerConfig::validate(std::string* error) const noexcept {
    if (vocabulary_size == 0) {
        set_error(error, "vocabulary_size must be greater than zero");
        return false;
    }
    if (context_length == 0) {
        set_error(error, "context_length must be greater than zero");
        return false;
    }
    if (layer_count == 0) {
        set_error(error, "layer_count must be greater than zero");
        return false;
    }
    if (hidden_size == 0) {
        set_error(error, "hidden_size must be greater than zero");
        return false;
    }
    if (attention_head_count == 0 || hidden_size % attention_head_count != 0) {
        set_error(error, "hidden_size must be divisible by attention_head_count");
        return false;
    }
    if (feed_forward_size == 0) {
        set_error(error, "feed_forward_size must be greater than zero");
        return false;
    }
    if (!causal) {
        set_error(error, "the first Attention transformer foundation requires causal attention");
        return false;
    }
    if (!std::isfinite(dropout_probability) || dropout_probability < 0.0f || dropout_probability >= 1.0f) {
        set_error(error, "dropout_probability must be finite and in [0, 1)");
        return false;
    }
    if (precision != ModelPrecision::F32) {
        set_error(error, "only F32 is implemented in the first transformer foundation");
        return false;
    }
    const std::uint64_t parameters = parameter_count();
    if (parameters == 0) {
        set_error(error, "configuration parameter count overflows or is zero");
        return false;
    }
    return true;
}

std::size_t TransformerConfig::head_size() const noexcept {
    return attention_head_count == 0 ? 0 : hidden_size / attention_head_count;
}

std::uint64_t TransformerConfig::parameter_count() const noexcept {
    const std::uint64_t v = static_cast<std::uint64_t>(vocabulary_size);
    const std::uint64_t c = static_cast<std::uint64_t>(context_length);
    const std::uint64_t l = static_cast<std::uint64_t>(layer_count);
    const std::uint64_t h = static_cast<std::uint64_t>(hidden_size);
    const std::uint64_t f = static_cast<std::uint64_t>(feed_forward_size);
    if (v == 0 || c == 0 || l == 0 || h == 0 || f == 0) return 0;

    // Token and learned positional embeddings.
    std::uint64_t total = 0;
    if (!multiply_fits(v, h) || !add_fits(total, v * h)) return 0;
    total += v * h;
    if (!multiply_fits(c, h) || !add_fits(total, c * h)) return 0;
    total += c * h;

    // Each block: QKV projection, output projection, two-layer feed-forward,
    // two affine normalizations, and all associated biases.
    std::uint64_t per_layer = 0;
    if (!multiply_fits(4, h) || !multiply_fits(4 * h, h) || !add_fits(per_layer, 4 * h * h)) return 0;
    per_layer += 4 * h * h;
    if (!multiply_fits(2, h) || !multiply_fits(2 * h, f) || !add_fits(per_layer, 2 * h * f)) return 0;
    per_layer += 2 * h * f;
    if (!add_fits(per_layer, 2 * f + 6 * h)) return 0;
    per_layer += 2 * f + 6 * h;
    if (!multiply_fits(l, per_layer) || !add_fits(total, l * per_layer)) return 0;
    total += l * per_layer;

    // Final normalization and an optional untied output projection.
    if (!add_fits(total, 2 * h)) return 0;
    total += 2 * h;
    if (!tie_embeddings) {
        if (!multiply_fits(v, h) || !add_fits(total, v * h + v)) return 0;
        total += v * h + v;
    } else if (!add_fits(total, v)) {
        return 0;
    } else {
        total += v; // output bias remains trainable when embeddings are tied
    }
    return total;
}

std::uint64_t TransformerConfig::parameter_bytes() const noexcept {
    const std::uint64_t parameters = parameter_count();
    return parameters == 0 || !multiply_fits(parameters, sizeof(float)) ? 0 : parameters * sizeof(float);
}

std::uint64_t TransformerConfig::estimated_activation_bytes(std::size_t batch_size) const noexcept {
    if (batch_size == 0 || context_length == 0 || hidden_size == 0 || layer_count == 0) return 0;
    const std::uint64_t b = static_cast<std::uint64_t>(batch_size);
    const std::uint64_t t = static_cast<std::uint64_t>(context_length);
    const std::uint64_t h = static_cast<std::uint64_t>(hidden_size);
    const std::uint64_t l = static_cast<std::uint64_t>(layer_count);
    if (!multiply_fits(b, t) || !multiply_fits(b * t, h) || !multiply_fits(b * t * h, 12) ||
        !multiply_fits(b * t * h * 12, l) || !multiply_fits(b * t * h * 12 * l, sizeof(float))) return 0;
    return b * t * h * 12 * l * sizeof(float);
}

} // namespace attention

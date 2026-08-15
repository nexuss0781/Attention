#include "attention/transformer_config.h"

#include <charconv>
#include <cmath>
#include <limits>
#include <new>
#include <utility>
#include <sstream>
#include <string_view>
#include <system_error>

namespace attention {
namespace {

bool multiply_fits(std::uint64_t a, std::uint64_t b) noexcept {
    return a == 0 || b <= std::numeric_limits<std::uint64_t>::max() / a;
}

bool add_fits(std::uint64_t a, std::uint64_t b) noexcept {
    return b <= std::numeric_limits<std::uint64_t>::max() - a;
}

bool add_product(std::uint64_t& total,
                 std::uint64_t first,
                 std::uint64_t second,
                 std::uint64_t third = 1) noexcept {
    if (!multiply_fits(first, second)) return false;
    const std::uint64_t product = first * second;
    if (!multiply_fits(product, third)) return false;
    const std::uint64_t term = product * third;
    if (!add_fits(total, term)) return false;
    total += term;
    return true;
}

void set_error(std::string* error, const char* message) noexcept {
    if (error != nullptr) *error = message;
}

bool parse_uint64(std::string_view text, std::uint64_t& value) noexcept {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_size(std::string_view text, std::size_t& value) noexcept {
    std::uint64_t parsed = 0;
    if (!parse_uint64(text, parsed) || parsed > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_bool(std::string_view text, bool& value) noexcept {
    if (text == "0") {
        value = false;
        return true;
    }
    if (text == "1") {
        value = true;
        return true;
    }
    return false;
}

bool parse_activation(std::string_view text, Activation& value) noexcept {
    std::uint64_t parsed = 0;
    if (!parse_uint64(text, parsed) || parsed > 1) return false;
    value = static_cast<Activation>(parsed);
    return true;
}

bool parse_precision(std::string_view text, ModelPrecision& value) noexcept {
    std::uint64_t parsed = 0;
    if (!parse_uint64(text, parsed) || parsed != 0) return false;
    value = ModelPrecision::F32;
    return true;
}

bool parse_float(std::string_view text, float& value) noexcept {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value,
                                        std::chars_format::general);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
           std::isfinite(value);
}

bool append_uint(std::string& output, std::uint64_t value) {
    char buffer[32]{};
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec != std::errc{}) return false;
    output.append(buffer, result.ptr);
    return true;
}

bool append_float(std::string& output, float value) {
    char buffer[32]{};
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                      std::chars_format::general,
                                      std::numeric_limits<float>::max_digits10);
    if (result.ec != std::errc{}) return false;
    output.append(buffer, result.ptr);
    return true;
}

bool append_field(std::string& output, const char* name, std::uint64_t value) {
    output.append(name);
    output.push_back('=');
    if (!append_uint(output, value)) return false;
    output.push_back('\n');
    return true;
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

    std::uint64_t total = 0;
    if (!add_product(total, v, h)) return 0;
    if (!tie_embeddings && !add_product(total, v, h)) return 0;

    // Three Q/K/V matrices and biases, two FFN matrices and biases, and
    // two affine layer-normalization parameter pairs are registered per layer.
    std::uint64_t per_layer = 0;
    if (!add_product(per_layer, 3, h, h) || !add_product(per_layer, 2, h, f) ||
        !add_fits(per_layer, f)) return 0;
    per_layer += f;
    if (!add_product(per_layer, 8, h)) return 0;
    if (!add_product(total, l, per_layer)) return 0;

    // Final normalization and the vocabulary bias are always registered.
    if (!add_product(total, 2, h) || !add_fits(total, v)) return 0;
    total += v;
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
    std::uint64_t total = b;
    if (!multiply_fits(total, t)) return 0;
    total *= t;
    if (!multiply_fits(total, h)) return 0;
    total *= h;
    if (!multiply_fits(total, 12)) return 0;
    total *= 12;
    if (!multiply_fits(total, l)) return 0;
    total *= l;
    if (!multiply_fits(total, sizeof(float))) return 0;
    return total * sizeof(float);
}

std::uint64_t TransformerConfig::estimated_inference_memory_bytes(
    std::size_t batch_size,
    std::size_t resident_sequence_length) const noexcept {
    if (!validate() || batch_size == 0) return 0;
    if (resident_sequence_length == 0) resident_sequence_length = context_length;
    if (resident_sequence_length == 0 || resident_sequence_length > context_length) return 0;

    const std::uint64_t b = static_cast<std::uint64_t>(batch_size);
    const std::uint64_t r = static_cast<std::uint64_t>(resident_sequence_length);
    const std::uint64_t h = static_cast<std::uint64_t>(hidden_size);
    const std::uint64_t f = static_cast<std::uint64_t>(feed_forward_size);
    const std::uint64_t v = static_cast<std::uint64_t>(vocabulary_size);
    std::uint64_t total = parameter_bytes();
    if (total == 0) return 0;

    std::uint64_t tokens = b;
    if (!multiply_fits(tokens, r)) return 0;
    tokens *= r;
    auto add_scaled = [&total](std::uint64_t value, std::uint64_t factor) {
        if (!multiply_fits(value, factor)) return false;
        return add_fits(total, value * factor) ? (total += value * factor, true) : false;
    };
    // Ten resident hidden-width buffers cover the block pipeline and output.
    std::uint64_t hidden_workspace = tokens;
    if (!multiply_fits(hidden_workspace, h)) return 0;
    hidden_workspace *= h;
    if (!add_scaled(hidden_workspace, 10 * sizeof(float))) return 0;
    // One expanded feed-forward workspace, token IDs, direct last-token logits,
    // and the dimension-bounded double-precision linear-attention state.
    std::uint64_t feed_forward_workspace = tokens;
    if (!multiply_fits(feed_forward_workspace, f)) return 0;
    feed_forward_workspace *= f;
    if (!add_scaled(feed_forward_workspace, sizeof(float))) return 0;
    if (!add_scaled(tokens, sizeof(std::uint32_t))) return 0;
    std::uint64_t logits = b;
    if (!multiply_fits(logits, v)) return 0;
    logits *= v;
    if (!add_scaled(logits, sizeof(float))) return 0;
    std::uint64_t attention_state = b;
    if (!multiply_fits(attention_state, h)) return 0;
    attention_state *= h;
    if (!multiply_fits(h, sizeof(double))) return 0;
    const std::uint64_t double_hidden_bytes = h * sizeof(double);
    if (!add_scaled(attention_state, double_hidden_bytes)) return 0;
    if (!add_scaled(b, double_hidden_bytes)) return 0;
    return total;
}

bool TransformerConfig::serialize(std::string& output, std::string* error) const noexcept {
    if (!validate(error)) return false;
    try {
        output.clear();
        output.append("attention.transformer_config.v1\n");
        if (!append_field(output, "vocabulary_size", vocabulary_size) ||
            !append_field(output, "context_length", context_length) ||
            !append_field(output, "layer_count", layer_count) ||
            !append_field(output, "hidden_size", hidden_size) ||
            !append_field(output, "attention_head_count", attention_head_count) ||
            !append_field(output, "feed_forward_size", feed_forward_size) ||
            !append_field(output, "activation", static_cast<std::uint64_t>(activation)) ||
            !append_field(output, "precision", static_cast<std::uint64_t>(precision)) ||
            !append_field(output, "tie_embeddings", tie_embeddings ? 1 : 0) ||
            !append_field(output, "causal", causal ? 1 : 0)) {
            set_error(error, "configuration serialization failed");
            return false;
        }
        output.append("dropout_probability=");
        if (!append_float(output, dropout_probability)) {
            set_error(error, "configuration serialization failed");
            return false;
        }
        output.push_back('\n');
    } catch (const std::bad_alloc&) {
        set_error(error, "configuration serialization allocation failed");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool TransformerConfig::deserialize(const std::string& serialized,
                                     TransformerConfig& output,
                                     std::string* error) noexcept {
    try {
        std::istringstream stream(serialized);
        std::string line;
        if (!std::getline(stream, line) || line != "attention.transformer_config.v1") {
            set_error(error, "unsupported or missing configuration serialization header");
            return false;
        }
        auto read_value = [&stream, &line](const char* key, std::string& value) {
            if (!std::getline(stream, line)) return false;
            const std::string prefix = std::string(key) + "=";
            if (line.rfind(prefix, 0) != 0) return false;
            value = line.substr(prefix.size());
            return !value.empty();
        };
        TransformerConfig parsed;
        std::string value;
        if (!read_value("vocabulary_size", value) || !parse_size(value, parsed.vocabulary_size) ||
            !read_value("context_length", value) || !parse_size(value, parsed.context_length) ||
            !read_value("layer_count", value) || !parse_size(value, parsed.layer_count) ||
            !read_value("hidden_size", value) || !parse_size(value, parsed.hidden_size) ||
            !read_value("attention_head_count", value) || !parse_size(value, parsed.attention_head_count) ||
            !read_value("feed_forward_size", value) || !parse_size(value, parsed.feed_forward_size) ||
            !read_value("activation", value) || !parse_activation(value, parsed.activation) ||
            !read_value("precision", value) || !parse_precision(value, parsed.precision) ||
            !read_value("tie_embeddings", value) || !parse_bool(value, parsed.tie_embeddings) ||
            !read_value("causal", value) || !parse_bool(value, parsed.causal) ||
            !read_value("dropout_probability", value) || !parse_float(value, parsed.dropout_probability)) {
            set_error(error, "malformed configuration serialization field");
            return false;
        }
        if (std::getline(stream, line)) {
            set_error(error, "unexpected trailing configuration serialization data");
            return false;
        }
        if (!parsed.validate(error)) return false;
        output = parsed;
    } catch (const std::bad_alloc&) {
        set_error(error, "configuration deserialization allocation failed");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

} // namespace attention

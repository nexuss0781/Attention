#include "attention/training_checkpoint.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>

namespace attention {
namespace {

constexpr std::string_view kMagicV1 = "attention.training_checkpoint.v1";
constexpr std::string_view kMagicV2 = "attention.training_checkpoint.v2";

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_line_text(const std::string& value) {
    return !value.empty() && value.find('\n') == std::string::npos && value.find('\r') == std::string::npos;
}

void append_size(std::string& output, std::uint64_t value) {
    char buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    output.append(buffer, result.ptr);
}

void append_float(std::string& output, float value) {
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                      std::chars_format::general,
                                      std::numeric_limits<float>::max_digits10);
    output.append(buffer, result.ptr);
}

bool parse_size(std::string_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool parse_float(std::string_view text, float& value) {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() && std::isfinite(value);
}

bool read_line(std::string_view input, std::size_t& cursor, std::string_view& line) {
    if (cursor > input.size()) return false;
    const std::size_t end = input.find('\n', cursor);
    if (end == std::string_view::npos) return false;
    line = input.substr(cursor, end - cursor);
    cursor = end + 1;
    return true;
}

bool read_bytes(std::string_view input, std::size_t& cursor, std::uint64_t length,
                std::string_view& bytes) {
    if (length > static_cast<std::uint64_t>(input.size()) ||
        cursor > input.size() - static_cast<std::size_t>(length)) return false;
    bytes = input.substr(cursor, static_cast<std::size_t>(length));
    cursor += static_cast<std::size_t>(length);
    return cursor < input.size() && input[cursor++] == '\n';
}

bool append_text_field(std::string& output, std::string_view name, const std::string& value) {
    if (!valid_line_text(value)) return false;
    output.append(name);
    output.push_back('\n');
    append_size(output, value.size());
    output.push_back('\n');
    output.append(value);
    output.push_back('\n');
    return true;
}

bool read_text_field(std::string_view input, std::size_t& cursor, std::string_view name,
                     std::string& output, std::string* error) {
    std::string_view line;
    if (!read_line(input, cursor, line) || line != name || !read_line(input, cursor, line)) {
        set_error(error, "training checkpoint text field header is invalid");
        return false;
    }
    std::uint64_t length = 0;
    if (!parse_size(line, length)) {
        set_error(error, "training checkpoint text field length is invalid");
        return false;
    }
    std::string_view bytes;
    if (!read_bytes(input, cursor, length, bytes) || bytes.empty() ||
        bytes.find('\n') != std::string_view::npos || bytes.find('\r') != std::string_view::npos) {
        set_error(error, "training checkpoint text field payload is invalid");
        return false;
    }
    output.assign(bytes);
    return true;
}

bool read_uint_field(std::string_view input, std::size_t& cursor, std::string_view name,
                     std::uint64_t& output, std::string* error) {
    std::string_view line;
    if (!read_line(input, cursor, line) || line != name || !read_line(input, cursor, line) ||
        !parse_size(line, output)) {
        set_error(error, "training checkpoint integer field is invalid");
        return false;
    }
    return true;
}

bool read_float_field(std::string_view input, std::size_t& cursor, std::string_view name,
                      float& output, std::string* error) {
    std::string_view line;
    if (!read_line(input, cursor, line) || line != name || !read_line(input, cursor, line) ||
        !parse_float(line, output)) {
        set_error(error, "training checkpoint float field is invalid");
        return false;
    }
    return true;
}

void append_optimizer_state(std::string& output, const OptimizerState& state) {
    output.append("optimizer_kind\n");
    append_size(output, static_cast<std::uint64_t>(state.kind));
    output.push_back('\n');
    output.append("optimizer_learning_rate\n");
    append_float(output, state.learning_rate);
    output.push_back('\n');
    output.append("optimizer_beta1\n");
    append_float(output, state.beta1);
    output.push_back('\n');
    output.append("optimizer_beta2\n");
    append_float(output, state.beta2);
    output.push_back('\n');
    output.append("optimizer_epsilon\n");
    append_float(output, state.epsilon);
    output.push_back('\n');
    output.append("optimizer_weight_decay\n");
    append_float(output, state.weight_decay);
    output.push_back('\n');
    output.append("optimizer_gradient_clip_norm\n");
    append_float(output, state.gradient_clip_norm);
    output.push_back('\n');
    output.append("optimizer_step_count\n");
    append_size(output, state.step_count);
    output.push_back('\n');
    output.append("optimizer_first_count\n");
    append_size(output, state.first_moments.size());
    output.push_back('\n');
    for (const auto& moment : state.first_moments) {
        output.append("optimizer_moment_size\n");
        append_size(output, moment.size());
        output.push_back('\n');
        for (float value : moment) {
            append_float(output, value);
            output.push_back('\n');
        }
    }
    output.append("optimizer_second_count\n");
    append_size(output, state.second_moments.size());
    output.push_back('\n');
    for (const auto& moment : state.second_moments) {
        output.append("optimizer_moment_size\n");
        append_size(output, moment.size());
        output.push_back('\n');
        for (float value : moment) {
            append_float(output, value);
            output.push_back('\n');
        }
    }
}

bool read_optimizer_state(std::string_view input, std::size_t& cursor,
                          OptimizerState& state, std::string* error) {
    std::uint64_t kind = 0;
    if (!read_uint_field(input, cursor, "optimizer_kind", kind, error) || kind > 1 ||
        !read_float_field(input, cursor, "optimizer_learning_rate", state.learning_rate, error) ||
        !read_float_field(input, cursor, "optimizer_beta1", state.beta1, error) ||
        !read_float_field(input, cursor, "optimizer_beta2", state.beta2, error) ||
        !read_float_field(input, cursor, "optimizer_epsilon", state.epsilon, error) ||
        !read_float_field(input, cursor, "optimizer_weight_decay", state.weight_decay, error) ||
        !read_float_field(input, cursor, "optimizer_gradient_clip_norm", state.gradient_clip_norm, error) ||
        !read_uint_field(input, cursor, "optimizer_step_count", state.step_count, error)) return false;
    state.kind = static_cast<OptimizerKind>(kind);
    std::uint64_t first_count = 0;
    if (!read_uint_field(input, cursor, "optimizer_first_count", first_count, error)) return false;
    state.first_moments.clear();
    state.first_moments.resize(static_cast<std::size_t>(first_count));
    for (auto& moment : state.first_moments) {
        std::uint64_t size = 0;
        if (!read_uint_field(input, cursor, "optimizer_moment_size", size, error)) return false;
        moment.resize(static_cast<std::size_t>(size));
        for (float& value : moment) {
            std::string_view line;
            if (!read_line(input, cursor, line) || !parse_float(line, value)) {
                set_error(error, "optimizer first moment value is invalid");
                return false;
            }
        }
    }
    std::uint64_t second_count = 0;
    if (!read_uint_field(input, cursor, "optimizer_second_count", second_count, error)) return false;
    state.second_moments.clear();
    state.second_moments.resize(static_cast<std::size_t>(second_count));
    for (auto& moment : state.second_moments) {
        std::uint64_t size = 0;
        if (!read_uint_field(input, cursor, "optimizer_moment_size", size, error)) return false;
        moment.resize(static_cast<std::size_t>(size));
        for (float& value : moment) {
            std::string_view line;
            if (!read_line(input, cursor, line) || !parse_float(line, value)) {
                set_error(error, "optimizer second moment value is invalid");
                return false;
            }
        }
    }
    return true;
}

} // namespace

bool TrainingCheckpoint::serialize(const TransformerConfig& config,
                                   const ParameterStore& parameters,
                                   const TrainingProgress& progress,
                                   std::string& output,
                                   std::string* error,
                                   const TokenizerMetadata& tokenizer,
                                   const OptimizerState* optimizer_state) {
    if (!valid_line_text(progress.run_id) || !valid_line_text(progress.dataset_id) ||
        !valid_line_text(progress.dataset_revision) || !std::isfinite(progress.learning_rate) ||
        progress.learning_rate <= 0.0f) {
        set_error(error, "training checkpoint progress is invalid");
        return false;
    }
    std::string model_checkpoint;
    if (!TransformerCheckpoint::serialize(config, parameters, model_checkpoint, error, tokenizer)) return false;
    output.clear();
    output.append(optimizer_state != nullptr ? kMagicV2 : kMagicV1);
    output.push_back('\n');
    if (!append_text_field(output, "run_id", progress.run_id) ||
        !append_text_field(output, "dataset_id", progress.dataset_id) ||
        !append_text_field(output, "dataset_revision", progress.dataset_revision)) {
        set_error(error, "training checkpoint progress text is invalid");
        return false;
    }
    output.append("global_step\n");
    append_size(output, progress.global_step);
    output.push_back('\n');
    output.append("tokens_processed\n");
    append_size(output, progress.tokens_processed);
    output.push_back('\n');
    output.append("next_batch_index\n");
    append_size(output, progress.next_batch_index);
    output.push_back('\n');
    output.append("learning_rate\n");
    append_float(output, progress.learning_rate);
    output.push_back('\n');
    if (optimizer_state != nullptr) append_optimizer_state(output, *optimizer_state);
    output.append("model_checkpoint_bytes\n");
    append_size(output, model_checkpoint.size());
    output.push_back('\n');
    output.append(model_checkpoint);
    output.push_back('\n');
    if (error != nullptr) error->clear();
    return true;
}

bool TrainingCheckpoint::load(std::string_view input,
                              TransformerModel& model,
                              ParameterStore& parameters,
                              TrainingProgress& progress,
                              std::string* error,
                              const TokenizerMetadata& expected_tokenizer,
                              OptimizerState* optimizer_state) {
    if (model.initialized() || parameters.size() != 0) {
        set_error(error, "training checkpoint load requires a fresh model and parameter store");
        return false;
    }
    std::size_t cursor = 0;
    std::string_view line;
    if (!read_line(input, cursor, line)) {
        set_error(error, "training checkpoint magic is invalid");
        return false;
    }
    const bool has_optimizer_state = line == kMagicV2;
    if (!has_optimizer_state && line != kMagicV1) {
        set_error(error, "training checkpoint magic is invalid");
        return false;
    }
    TrainingProgress loaded;
    if (!read_text_field(input, cursor, "run_id", loaded.run_id, error) ||
        !read_text_field(input, cursor, "dataset_id", loaded.dataset_id, error) ||
        !read_text_field(input, cursor, "dataset_revision", loaded.dataset_revision, error) ||
        !read_uint_field(input, cursor, "global_step", loaded.global_step, error) ||
        !read_uint_field(input, cursor, "tokens_processed", loaded.tokens_processed, error) ||
        !read_uint_field(input, cursor, "next_batch_index", loaded.next_batch_index, error) ||
        !read_float_field(input, cursor, "learning_rate", loaded.learning_rate, error)) return false;
    OptimizerState loaded_optimizer;
    if (has_optimizer_state && !read_optimizer_state(input, cursor, loaded_optimizer, error)) return false;
    if (!read_line(input, cursor, line) || line != "model_checkpoint_bytes" ||
        !read_line(input, cursor, line)) {
        set_error(error, "training checkpoint model payload header is invalid");
        return false;
    }
    std::uint64_t length = 0;
    if (!parse_size(line, length)) {
        set_error(error, "training checkpoint model payload length is invalid");
        return false;
    }
    std::string_view model_payload;
    if (!read_bytes(input, cursor, length, model_payload) || cursor != input.size()) {
        set_error(error, "training checkpoint model payload is truncated or has trailing data");
        return false;
    }
    if (!TransformerCheckpoint::load(model_payload, model, parameters, error, expected_tokenizer)) return false;
    progress = std::move(loaded);
    if (optimizer_state != nullptr) {
        *optimizer_state = has_optimizer_state ? std::move(loaded_optimizer) : OptimizerState{};
    }
    if (error != nullptr) error->clear();
    return true;
}

} // namespace attention

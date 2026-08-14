#include "attention/checkpoint.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <system_error>
#include <vector>

namespace attention {
namespace {

constexpr std::string_view kMagic = "attention.checkpoint.v1";

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool parse_size(std::string_view text, std::size_t& value) {
    if (text.empty()) return false;
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        parsed > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_float(std::string_view text, float& value) {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() && std::isfinite(value);
}

void append_size(std::string& output, std::size_t value) {
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

bool read_line(std::string_view input, std::size_t& cursor, std::string_view& line) {
    if (cursor > input.size()) return false;
    const std::size_t end = input.find('\n', cursor);
    if (end == std::string_view::npos) return false;
    line = input.substr(cursor, end - cursor);
    cursor = end + 1;
    return true;
}

bool read_bytes(std::string_view input, std::size_t& cursor, std::size_t length,
                std::string_view& bytes) {
    if (length > input.size() - cursor) return false;
    bytes = input.substr(cursor, length);
    cursor += length;
    return cursor < input.size() && input[cursor++] == '\n';
}

struct LoadedParameter {
    std::string name;
    std::vector<std::size_t> shape;
    std::vector<float> values;
};

bool parse_checkpoint(std::string_view input, TransformerConfig& config,
                      std::vector<LoadedParameter>& loaded, std::string* error) {
    std::size_t cursor = 0;
    std::string_view line;
    if (!read_line(input, cursor, line) || line != kMagic) {
        set_error(error, "checkpoint magic is invalid");
        return false;
    }
    if (!read_line(input, cursor, line) || line != "config_bytes") {
        set_error(error, "checkpoint config header is invalid");
        return false;
    }
    if (!read_line(input, cursor, line)) {
        set_error(error, "checkpoint config length is missing");
        return false;
    }
    std::size_t config_length = 0;
    if (!parse_size(line, config_length)) {
        set_error(error, "checkpoint config length is invalid");
        return false;
    }
    std::string_view config_text;
    if (!read_bytes(input, cursor, config_length, config_text)) {
        set_error(error, "checkpoint config payload is truncated");
        return false;
    }
    std::string config_error;
    if (!TransformerConfig::deserialize(std::string(config_text), config, &config_error)) {
        if (error != nullptr) *error = "checkpoint configuration is invalid: " + config_error;
        return false;
    }
    if (!read_line(input, cursor, line) || line != "parameter_count") {
        set_error(error, "checkpoint parameter header is invalid");
        return false;
    }
    if (!read_line(input, cursor, line)) {
        set_error(error, "checkpoint parameter count is missing");
        return false;
    }
    std::size_t parameter_count = 0;
    if (!parse_size(line, parameter_count)) {
        set_error(error, "checkpoint parameter count is invalid");
        return false;
    }
    try {
        loaded.reserve(parameter_count);
    } catch (...) {
        set_error(error, "checkpoint parameter allocation failed");
        return false;
    }
    std::string previous_name;
    for (std::size_t parameter_index = 0; parameter_index < parameter_count; ++parameter_index) {
        LoadedParameter parameter;
        if (!read_line(input, cursor, line) || line != "name_bytes") {
            set_error(error, "checkpoint parameter name header is invalid");
            return false;
        }
        if (!read_line(input, cursor, line)) {
            set_error(error, "checkpoint parameter name length is missing");
            return false;
        }
        std::size_t name_length = 0;
        if (!parse_size(line, name_length)) {
            set_error(error, "checkpoint parameter name length is invalid");
            return false;
        }
        std::string_view name_bytes;
        if (!read_bytes(input, cursor, name_length, name_bytes) || name_bytes.empty()) {
            set_error(error, "checkpoint parameter name is invalid");
            return false;
        }
        parameter.name = std::string(name_bytes);
        if (!previous_name.empty() && parameter.name <= previous_name) {
            set_error(error, "checkpoint parameter names are not strictly sorted");
            return false;
        }
        previous_name = parameter.name;
        if (!read_line(input, cursor, line) || line != "rank") {
            set_error(error, "checkpoint parameter rank header is invalid");
            return false;
        }
        if (!read_line(input, cursor, line)) {
            set_error(error, "checkpoint parameter rank is missing");
            return false;
        }
        std::size_t rank = 0;
        if (!parse_size(line, rank) || rank == 0) {
            set_error(error, "checkpoint parameter rank is invalid");
            return false;
        }
        parameter.shape.reserve(rank);
        if (!read_line(input, cursor, line) || line != "shape") {
            set_error(error, "checkpoint parameter shape header is invalid");
            return false;
        }
        if (!read_line(input, cursor, line)) {
            set_error(error, "checkpoint parameter shape is missing");
            return false;
        }
        std::size_t shape_cursor = 0;
        while (shape_cursor < line.size()) {
            const std::size_t separator = line.find(' ', shape_cursor);
            const std::size_t end = separator == std::string_view::npos ? line.size() : separator;
            std::size_t dimension = 0;
            if (!parse_size(line.substr(shape_cursor, end - shape_cursor), dimension) || dimension == 0) {
                set_error(error, "checkpoint parameter dimension is invalid");
                return false;
            }
            parameter.shape.push_back(dimension);
            if (parameter.shape.size() > rank) {
                set_error(error, "checkpoint parameter shape rank is invalid");
                return false;
            }
            shape_cursor = separator == std::string_view::npos ? line.size() : separator + 1;
        }
        if (parameter.shape.size() != rank) {
            set_error(error, "checkpoint parameter shape rank does not match");
            return false;
        }
        if (!read_line(input, cursor, line) || line != "value_count") {
            set_error(error, "checkpoint parameter value header is invalid");
            return false;
        }
        if (!read_line(input, cursor, line)) {
            set_error(error, "checkpoint parameter value count is missing");
            return false;
        }
        std::size_t value_count = 0;
        if (!parse_size(line, value_count)) {
            set_error(error, "checkpoint parameter value count is invalid");
            return false;
        }
        parameter.values.reserve(value_count);
        if (!read_line(input, cursor, line) || line != "values") {
            set_error(error, "checkpoint parameter values header is invalid");
            return false;
        }
        for (std::size_t value_index = 0; value_index < value_count; ++value_index) {
            if (!read_line(input, cursor, line)) {
                set_error(error, "checkpoint parameter value is missing");
                return false;
            }
            float value = 0.0f;
            if (!parse_float(line, value)) {
                set_error(error, "checkpoint parameter value is invalid");
                return false;
            }
            parameter.values.push_back(value);
        }
        loaded.push_back(std::move(parameter));
    }
    if (cursor != input.size()) {
        set_error(error, "checkpoint has trailing data");
        return false;
    }
    return true;
}

} // namespace

bool TransformerCheckpoint::serialize(const TransformerConfig& config,
                                      const ParameterStore& parameters,
                                      std::string& output,
                                      std::string* error) {
    if (!config.validate(error) || !parameters.all_finite()) {
        if (error != nullptr && error->empty()) *error = "checkpoint parameters are not finite";
        return false;
    }
    std::string config_text;
    if (!config.serialize(config_text, error)) return false;
    output.clear();
    output.append(kMagic);
    output.push_back('\n');
    output.append("config_bytes\n");
    append_size(output, config_text.size());
    output.push_back('\n');
    output.append(config_text);
    output.push_back('\n');
    const std::vector<std::string> names = parameters.names();
    output.append("parameter_count\n");
    append_size(output, names.size());
    output.push_back('\n');
    for (const std::string& name : names) {
        const Parameter* parameter = parameters.find(name);
        if (parameter == nullptr || !parameter->value.valid() || parameter->value.size() == 0 ||
            !parameter->value.all_finite()) {
            set_error(error, "checkpoint parameter is missing, empty, or nonfinite");
            return false;
        }
        output.append("name_bytes\n");
        append_size(output, name.size());
        output.push_back('\n');
        output.append(name);
        output.push_back('\n');
        output.append("rank\n");
        append_size(output, parameter->value.shape().size());
        output.push_back('\n');
        output.append("shape\n");
        for (std::size_t dimension = 0; dimension < parameter->value.shape().size(); ++dimension) {
            if (dimension != 0) output.push_back(' ');
            append_size(output, parameter->value.shape()[dimension]);
        }
        output.push_back('\n');
        output.append("value_count\n");
        append_size(output, parameter->value.size());
        output.push_back('\n');
        output.append("values\n");
        for (std::size_t index = 0; index < parameter->value.size(); ++index) {
            append_float(output, parameter->value.data()[index]);
            output.push_back('\n');
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

bool TransformerCheckpoint::load(std::string_view input,
                                 TransformerModel& model,
                                 ParameterStore& parameters,
                                 std::string* error) {
    if (model.initialized() || parameters.size() != 0) {
        set_error(error, "checkpoint load requires a fresh model and parameter store");
        return false;
    }
    TransformerConfig config;
    std::vector<LoadedParameter> loaded;
    if (!parse_checkpoint(input, config, loaded, error)) return false;
    if (!model.register_parameters(config, parameters, error)) return false;
    const std::vector<std::string> names = parameters.names();
    if (loaded.size() != names.size()) {
        set_error(error, "checkpoint parameter count does not match model");
        return false;
    }
    for (const LoadedParameter& loaded_parameter : loaded) {
        Parameter* parameter = parameters.find(loaded_parameter.name);
        if (parameter == nullptr || parameter->value.shape() != loaded_parameter.shape ||
            parameter->value.size() != loaded_parameter.values.size()) {
            set_error(error, "checkpoint parameter name or shape does not match model");
            return false;
        }
        for (std::size_t index = 0; index < parameter->value.size(); ++index) {
            parameter->value.data()[index] = loaded_parameter.values[index];
        }
    }
    if (!parameters.all_finite()) {
        set_error(error, "checkpoint reload produced nonfinite parameters");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

} // namespace attention

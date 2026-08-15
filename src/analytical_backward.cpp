#include "analytical_backward.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace attention {
namespace {

struct Var {
    int id = -1;
};

struct Node {
    double value = 0.0;
    int left = -1;
    int right = -1;
    double d_left = 0.0;
    double d_right = 0.0;
};

class Tape {
public:
    Var constant(double value) {
        nodes_.push_back(Node{value, -1, -1, 0.0, 0.0});
        return Var{static_cast<int>(nodes_.size() - 1)};
    }

    Var variable(double value) {
        return constant(value);
    }

    double value(Var variable) const { return nodes_[static_cast<std::size_t>(variable.id)].value; }

    Var cast_float(Var variable) {
        return unary(variable, static_cast<double>(static_cast<float>(value(variable))), 1.0);
    }

    Var add(Var left, Var right) {
        return binary(left, right, value(left) + value(right), 1.0, 1.0);
    }

    Var sub(Var left, Var right) {
        return binary(left, right, value(left) - value(right), 1.0, -1.0);
    }

    Var mul(Var left, Var right) {
        return binary(left, right, value(left) * value(right), value(right), value(left));
    }

    Var div(Var left, Var right) {
        const double denominator = value(right);
        const double result = value(left) / denominator;
        return binary(left, right, result, 1.0 / denominator,
                      -value(left) / (denominator * denominator));
    }

    Var exp(Var variable) {
        const double result = std::exp(value(variable));
        return unary(variable, result, result);
    }

    Var log(Var variable) {
        return unary(variable, std::log(value(variable)), 1.0 / value(variable));
    }

    Var sqrt(Var variable) {
        const double result = std::sqrt(value(variable));
        return unary(variable, result, 0.5 / result);
    }

    Var tanh(Var variable) {
        const double result = std::tanh(value(variable));
        return unary(variable, result, 1.0 - result * result);
    }

    void reverse(Var loss) {
        gradients_.assign(nodes_.size(), 0.0);
        gradients_[static_cast<std::size_t>(loss.id)] = 1.0;
        for (std::size_t index = nodes_.size(); index-- > 0;) {
            const Node& node = nodes_[index];
            const double gradient = gradients_[index];
            if (node.left >= 0) gradients_[static_cast<std::size_t>(node.left)] += gradient * node.d_left;
            if (node.right >= 0) gradients_[static_cast<std::size_t>(node.right)] += gradient * node.d_right;
        }
    }

    double gradient(Var variable) const {
        return gradients_.empty() ? 0.0 : gradients_[static_cast<std::size_t>(variable.id)];
    }

private:
    Var unary(Var input, double result, double derivative) {
        nodes_.push_back(Node{result, input.id, -1, derivative, 0.0});
        return Var{static_cast<int>(nodes_.size() - 1)};
    }

    Var binary(Var left, Var right, double result, double d_left, double d_right) {
        nodes_.push_back(Node{result, left.id, right.id, d_left, d_right});
        return Var{static_cast<int>(nodes_.size() - 1)};
    }

    std::vector<Node> nodes_;
    std::vector<double> gradients_;
};

class ParameterVars {
public:
    ParameterVars(ParameterStore& parameters, Tape& tape, std::string* error)
        : parameters_(parameters), tape_(tape), error_(error) {}

    bool initialize() {
        for (Parameter& parameter : parameters_.parameters()) {
            std::vector<Var> values;
            values.reserve(parameter.value.size());
            for (std::size_t index = 0; index < parameter.value.size(); ++index) {
                if (!std::isfinite(parameter.value.data()[index])) {
                    set_error("parameter value is nonfinite");
                    return false;
                }
                values.push_back(tape_.variable(static_cast<double>(parameter.value.data()[index])));
            }
            values_.emplace(parameter.name, std::move(values));
        }
        return true;
    }

    Var get(const std::string& name, std::size_t index) {
        auto iterator = values_.find(name);
        if (iterator == values_.end() || index >= iterator->second.size()) {
            set_error("analytical backward parameter is missing or out of range");
            return Var{-1};
        }
        return iterator->second[index];
    }

    bool valid() const noexcept { return error_ == nullptr || error_->empty(); }

    bool write_gradients() {
        for (Parameter& parameter : parameters_.parameters()) {
            const auto iterator = values_.find(parameter.name);
            if (iterator == values_.end() || iterator->second.size() != parameter.gradient.size()) {
                set_error("analytical backward parameter gradient shape mismatch");
                return false;
            }
            for (std::size_t index = 0; index < parameter.gradient.size(); ++index) {
                const double gradient = tape_.gradient(iterator->second[index]);
                if (!std::isfinite(gradient) || std::abs(gradient) > static_cast<double>(std::numeric_limits<float>::max())) {
                    set_error("analytical backward gradient is nonfinite or out of float range");
                    return false;
                }
                parameter.gradient.data()[index] = static_cast<float>(gradient);
            }
        }
        return true;
    }

private:
    void set_error(const char* message) {
        if (error_ != nullptr) *error_ = message;
    }

    ParameterStore& parameters_;
    Tape& tape_;
    std::string* error_;
    std::unordered_map<std::string, std::vector<Var>> values_;
};

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

Var cast(Tape& tape, Var value) { return tape.cast_float(value); }

Var sum(Tape& tape, const std::vector<Var>& values) {
    Var output = tape.constant(0.0);
    for (Var value : values) output = tape.add(output, value);
    return output;
}

Var affine(Tape& tape, ParameterVars& parameters, const std::string& weight_name,
           const std::string& bias_name, const std::vector<Var>& input,
           std::size_t output_index, std::size_t input_size) {
    Var output = parameters.get(bias_name, output_index);
    for (std::size_t input_index = 0; input_index < input_size; ++input_index) {
        output = tape.add(output, tape.mul(input[input_index], parameters.get(weight_name, output_index * input_size + input_index)));
    }
    return cast(tape, output);
}

std::vector<Var> layer_norm(Tape& tape, ParameterVars& parameters, const std::vector<Var>& input,
                            const std::string& prefix) {
    const std::size_t hidden = input.size();
    const Var mean = tape.div(sum(tape, input), tape.constant(static_cast<double>(hidden)));
    std::vector<Var> centered(hidden);
    std::vector<Var> squares(hidden);
    for (std::size_t index = 0; index < hidden; ++index) {
        centered[index] = tape.sub(input[index], mean);
        squares[index] = tape.mul(centered[index], centered[index]);
    }
    const Var variance = tape.div(sum(tape, squares), tape.constant(static_cast<double>(hidden)));
    const Var inverse_std = tape.div(tape.constant(1.0), tape.sqrt(tape.add(variance, tape.constant(1e-5))));
    std::vector<Var> output(hidden);
    for (std::size_t index = 0; index < hidden; ++index) {
        const Var normalized = tape.mul(centered[index], inverse_std);
        output[index] = cast(tape, tape.add(
            tape.mul(normalized, parameters.get(prefix + ".weight", index)),
            parameters.get(prefix + ".bias", index)));
    }
    return output;
}

Var gelu(Tape& tape, Var input) {
    constexpr double sqrt_two_over_pi = 0.7978845608028654;
    constexpr double coefficient = 0.044715;
    const Var cubic = tape.mul(tape.mul(input, input), input);
    const Var inner = tape.mul(tape.constant(sqrt_two_over_pi),
                               tape.add(input, tape.mul(tape.constant(coefficient), cubic)));
    return cast(tape, tape.mul(tape.constant(0.5),
                               tape.mul(input, tape.add(tape.constant(1.0), tape.tanh(inner)))));
}

std::vector<Var> feed_forward(Tape& tape, ParameterVars& parameters, const std::vector<Var>& input,
                              const std::string& prefix, std::size_t feed_forward_size) {
    const std::size_t hidden = input.size();
    std::vector<Var> expanded(feed_forward_size);
    for (std::size_t expansion = 0; expansion < feed_forward_size; ++expansion) {
        Var value = parameters.get(prefix + ".up.bias", expansion);
        for (std::size_t dimension = 0; dimension < hidden; ++dimension) {
            value = tape.add(value, tape.mul(input[dimension], parameters.get(
                prefix + ".up.weight", dimension * feed_forward_size + expansion)));
        }
        expanded[expansion] = gelu(tape, cast(tape, value));
    }
    std::vector<Var> output(hidden);
    for (std::size_t dimension = 0; dimension < hidden; ++dimension) {
        Var value = parameters.get(prefix + ".down.bias", dimension);
        for (std::size_t expansion = 0; expansion < feed_forward_size; ++expansion) {
            value = tape.add(value, tape.mul(expanded[expansion], parameters.get(
                prefix + ".down.weight", expansion * hidden + dimension)));
        }
        output[dimension] = cast(tape, value);
    }
    return output;
}

std::vector<Var> qkv(Tape& tape, ParameterVars& parameters, const std::vector<Var>& input,
                     const std::string& prefix, const char* projection, std::size_t hidden) {
    std::vector<Var> output(hidden);
    const std::string weight = prefix + ".attention." + projection + "_proj.weight";
    const std::string bias = prefix + ".attention." + projection + "_proj.bias";
    for (std::size_t dimension = 0; dimension < hidden; ++dimension) {
        output[dimension] = affine(tape, parameters, weight, bias, input, dimension, hidden);
    }
    return output;
}

std::vector<Var> attention(Tape& tape, const std::vector<std::vector<Var>>& query,
                           const std::vector<std::vector<Var>>& key,
                           const std::vector<std::vector<Var>>& value,
                           std::size_t hidden,
                           std::size_t head_count) {
    const std::size_t sequence = query.size();
    const std::size_t head_size = hidden / head_count;
    std::vector<Var> output(sequence * hidden);
    std::vector<std::vector<Var>> prefix_normalizer(sequence, std::vector<Var>(hidden));
    std::vector<std::vector<Var>> prefix_state(
        sequence, std::vector<Var>(head_count * head_size * head_size));
    std::vector<Var> previous_normalizer(hidden);
    std::vector<Var> previous_state(head_count * head_size * head_size);
    for (Var& item : previous_normalizer) item = tape.constant(0.0);
    for (Var& item : previous_state) item = tape.constant(0.0);

    for (std::size_t position = 0; position < sequence; ++position) {
        std::vector<Var> key_features(hidden);
        for (std::size_t channel = 0; channel < hidden; ++channel) {
            const double raw = tape.value(key[position][channel]);
            const double clipped = std::clamp(raw, -20.0, 20.0);
            key_features[channel] = (raw <= -20.0 || raw >= 20.0)
                ? tape.constant(std::exp(clipped))
                : tape.exp(key[position][channel]);
            prefix_normalizer[position][channel] = tape.add(
                previous_normalizer[channel], key_features[channel]);
        }
        for (std::size_t head = 0; head < head_count; ++head) {
            const std::size_t channel_offset = head * head_size;
            const std::size_t state_offset = head * head_size * head_size;
            for (std::size_t key_channel = 0; key_channel < head_size; ++key_channel) {
                for (std::size_t value_channel = 0; value_channel < head_size; ++value_channel) {
                    const std::size_t index = state_offset + key_channel * head_size + value_channel;
                    prefix_state[position][index] = tape.add(
                        previous_state[index], tape.mul(
                            key_features[channel_offset + key_channel],
                            value[position][channel_offset + value_channel]));
                }
            }
        }
        previous_normalizer = prefix_normalizer[position];
        previous_state = prefix_state[position];

        std::vector<Var> query_features(hidden);
        for (std::size_t channel = 0; channel < hidden; ++channel) {
            const double raw = tape.value(query[position][channel]);
            const double clipped = std::clamp(raw, -20.0, 20.0);
            query_features[channel] = (raw <= -20.0 || raw >= 20.0)
                ? tape.constant(std::exp(clipped))
                : tape.exp(query[position][channel]);
        }
        for (std::size_t head = 0; head < head_count; ++head) {
            const std::size_t channel_offset = head * head_size;
            const std::size_t state_offset = head * head_size * head_size;
            Var denominator = tape.constant(0.0);
            for (std::size_t key_channel = 0; key_channel < head_size; ++key_channel) {
                const std::size_t channel = channel_offset + key_channel;
                denominator = tape.add(denominator, tape.mul(
                    query_features[channel], prefix_normalizer[position][channel]));
            }
            const Var safe_denominator = tape.value(denominator) > 1e-6
                ? denominator : tape.constant(1e-6);
            for (std::size_t value_channel = 0; value_channel < head_size; ++value_channel) {
                Var numerator = tape.constant(0.0);
                for (std::size_t key_channel = 0; key_channel < head_size; ++key_channel) {
                    numerator = tape.add(numerator, tape.mul(
                        query_features[channel_offset + key_channel],
                        prefix_state[position][state_offset + key_channel * head_size + value_channel]));
                }
                output[position * hidden + channel_offset + value_channel] = cast(
                    tape, tape.div(numerator, safe_denominator));
            }
        }
    }
    return output;
}

} // namespace

bool analytical_backward(const TransformerModel& model,
                         const std::vector<std::size_t>& token_ids,
                         std::size_t batch_size,
                         std::size_t sequence_length,
                         ParameterStore& parameters,
                         std::string* error) {
    if (!model.initialized()) {
        set_error(error, "analytical backward model is not initialized");
        return false;
    }
    const TransformerConfig& config = model.config();
    if (batch_size == 0 || sequence_length < 2 || sequence_length > config.context_length ||
        token_ids.size() != batch_size * sequence_length) {
        set_error(error, "analytical backward token shape requires a valid batch and sequence");
        return false;
    }
    for (std::size_t token : token_ids) {
        if (token >= config.vocabulary_size) {
            set_error(error, "analytical backward token is outside vocabulary");
            return false;
        }
    }
    parameters.clear_gradients();
    Tape tape;
    ParameterVars parameter_vars(parameters, tape, error);
    if (!parameter_vars.initialize()) return false;
    std::vector<std::vector<Var>> hidden(batch_size * sequence_length,
                                         std::vector<Var>(config.hidden_size));
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const std::size_t row = batch * sequence_length + position;
            for (std::size_t dimension = 0; dimension < config.hidden_size; ++dimension) {
                Var embedded = parameter_vars.get("embedding.weight",
                    token_ids[row] * config.hidden_size + dimension);
                const std::size_t pair_index = dimension / 2;
                const double exponent = (2.0 * static_cast<double>(pair_index)) /
                    static_cast<double>(config.hidden_size);
                const double angle = static_cast<double>(position) * std::exp(-9.210340371976184 * exponent);
                const double positional = dimension % 2 == 0 ? std::sin(angle) : std::cos(angle);
                hidden[row][dimension] = cast(tape, tape.add(embedded, tape.constant(positional)));
            }
        }
    }

    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        const std::string block = "layers." + std::to_string(layer);
        std::vector<std::vector<Var>> normalized(batch_size * sequence_length);
        std::vector<std::vector<Var>> queries(batch_size * sequence_length);
        std::vector<std::vector<Var>> keys(batch_size * sequence_length);
        std::vector<std::vector<Var>> values(batch_size * sequence_length);
        for (std::size_t row = 0; row < batch_size * sequence_length; ++row) {
            normalized[row] = layer_norm(tape, parameter_vars, hidden[row], block + ".norm1");
            queries[row] = qkv(tape, parameter_vars, normalized[row], block, "q", config.hidden_size);
            keys[row] = qkv(tape, parameter_vars, normalized[row], block, "k", config.hidden_size);
            values[row] = qkv(tape, parameter_vars, normalized[row], block, "v", config.hidden_size);
        }
        std::vector<std::vector<Var>> after_attention(batch_size * sequence_length);
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            std::vector<std::vector<Var>> batch_query(sequence_length), batch_key(sequence_length), batch_value(sequence_length);
            for (std::size_t position = 0; position < sequence_length; ++position) {
                batch_query[position] = queries[batch * sequence_length + position];
                batch_key[position] = keys[batch * sequence_length + position];
                batch_value[position] = values[batch * sequence_length + position];
            }
            const std::vector<Var> attended = attention(tape, batch_query, batch_key, batch_value, config.hidden_size,
                      config.attention_head_count);
            for (std::size_t position = 0; position < sequence_length; ++position) {
                after_attention[batch * sequence_length + position].resize(config.hidden_size);
                for (std::size_t dimension = 0; dimension < config.hidden_size; ++dimension) {
                    after_attention[batch * sequence_length + position][dimension] = cast(tape, tape.add(
                        hidden[batch * sequence_length + position][dimension],
                        attended[position * config.hidden_size + dimension]));
                }
            }
        }
        std::vector<std::vector<Var>> next_hidden(batch_size * sequence_length);
        for (std::size_t row = 0; row < batch_size * sequence_length; ++row) {
            const std::vector<Var> normalized_ffn = layer_norm(tape, parameter_vars,
                after_attention[row], block + ".norm2");
            const std::vector<Var> ffn = feed_forward(tape, parameter_vars, normalized_ffn,
                block + ".ffn", config.feed_forward_size);
            next_hidden[row].resize(config.hidden_size);
            for (std::size_t dimension = 0; dimension < config.hidden_size; ++dimension) {
                next_hidden[row][dimension] = cast(tape, tape.add(after_attention[row][dimension], ffn[dimension]));
            }
        }
        hidden = std::move(next_hidden);
    }

    std::vector<Var> losses;
    losses.reserve(batch_size * (sequence_length - 1));
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position + 1 < sequence_length; ++position) {
            const std::size_t row = batch * sequence_length + position;
            const std::vector<Var> normalized = layer_norm(tape, parameter_vars, hidden[row], "final_norm");
            std::vector<Var> logits(config.vocabulary_size);
            for (std::size_t vocabulary = 0; vocabulary < config.vocabulary_size; ++vocabulary) {
                Var value = parameter_vars.get("lm_head.bias", vocabulary);
                for (std::size_t dimension = 0; dimension < config.hidden_size; ++dimension) {
                    value = tape.add(value, tape.mul(normalized[dimension], parameter_vars.get(
                        config.tie_embeddings ? "embedding.weight" : "lm_head.weight",
                        vocabulary * config.hidden_size + dimension)));
                }
                logits[vocabulary] = value;
            }
            const std::size_t target = token_ids[row + 1];
            const double maximum = [&]() {
                double result = -std::numeric_limits<double>::infinity();
                for (Var logit : logits) result = std::max(result, tape.value(logit));
                return result;
            }();
            Var sum_exp = tape.constant(0.0);
            for (Var logit : logits) sum_exp = tape.add(sum_exp, tape.exp(tape.sub(logit, tape.constant(maximum))));
            losses.push_back(tape.sub(tape.log(sum_exp), tape.sub(logits[target], tape.constant(maximum))));
        }
    }
    if (losses.empty()) {
        set_error(error, "analytical backward produced no prediction losses");
        return false;
    }
    const Var total = tape.div(sum(tape, losses), tape.constant(static_cast<double>(losses.size())));
    if (!std::isfinite(tape.value(total))) {
        set_error(error, "analytical backward loss is nonfinite");
        return false;
    }
    tape.reverse(total);
    if (!parameter_vars.write_gradients()) return false;
    if (!parameters.all_finite()) {
        set_error(error, "analytical backward produced nonfinite parameter state");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

} // namespace attention

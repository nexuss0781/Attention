#ifndef ATTENTION_OPTIMIZER_H
#define ATTENTION_OPTIMIZER_H

#include "attention/parameter_store.h"

#include <cstdint>
#include <string>
#include <vector>

namespace attention {

enum class OptimizerKind {
    SGD,
    AdamW,
};

struct OptimizerState {
    OptimizerKind kind = OptimizerKind::SGD;
    float learning_rate = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float epsilon = 0.0f;
    float weight_decay = 0.0f;
    float gradient_clip_norm = 0.0f;
    std::uint64_t step_count = 0;
    std::vector<std::string> parameter_names;
    std::vector<std::vector<float>> first_moments;
    std::vector<std::vector<float>> second_moments;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual bool step(ParameterStore& parameters, std::string* error = nullptr) const = 0;
    virtual bool export_state(OptimizerState& state, std::string* error = nullptr) const = 0;
    virtual bool import_state(const OptimizerState& state, std::string* error = nullptr) = 0;
};

class SgdOptimizer final : public Optimizer {
public:
    explicit SgdOptimizer(float learning_rate, float gradient_clip_norm = 0.0f) noexcept
        : learning_rate_(learning_rate), gradient_clip_norm_(gradient_clip_norm) {}

    [[nodiscard]] float learning_rate() const noexcept { return learning_rate_; }
    [[nodiscard]] float gradient_clip_norm() const noexcept { return gradient_clip_norm_; }

    bool step(ParameterStore& parameters, std::string* error = nullptr) const override;
    bool export_state(OptimizerState& state, std::string* error = nullptr) const override;
    bool import_state(const OptimizerState& state, std::string* error = nullptr) override;

private:
    float learning_rate_ = 0.0f;
    float gradient_clip_norm_ = 0.0f;
};

class AdamWOptimizer final : public Optimizer {
public:
    explicit AdamWOptimizer(float learning_rate,
                            float beta1 = 0.9f,
                            float beta2 = 0.999f,
                            float epsilon = 1e-8f,
                            float weight_decay = 0.0f,
                            float gradient_clip_norm = 0.0f) noexcept
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
          weight_decay_(weight_decay), gradient_clip_norm_(gradient_clip_norm) {}

    [[nodiscard]] float learning_rate() const noexcept { return learning_rate_; }
    [[nodiscard]] float beta1() const noexcept { return beta1_; }
    [[nodiscard]] float beta2() const noexcept { return beta2_; }
    [[nodiscard]] float epsilon() const noexcept { return epsilon_; }
    [[nodiscard]] float weight_decay() const noexcept { return weight_decay_; }
    [[nodiscard]] float gradient_clip_norm() const noexcept { return gradient_clip_norm_; }
    [[nodiscard]] std::uint64_t step_count() const noexcept { return step_count_; }

    bool step(ParameterStore& parameters, std::string* error = nullptr) const override;
    bool export_state(OptimizerState& state, std::string* error = nullptr) const override;
    bool import_state(const OptimizerState& state, std::string* error = nullptr) override;

private:
    bool initialize_moments(const ParameterStore& parameters, std::string* error) const;

    float learning_rate_ = 0.0f;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float epsilon_ = 1e-8f;
    float weight_decay_ = 0.0f;
    float gradient_clip_norm_ = 0.0f;
    mutable std::uint64_t step_count_ = 0;
    mutable float beta1_power_ = 1.0f;
    mutable float beta2_power_ = 1.0f;
    mutable std::vector<std::vector<float>> first_moments_;
    mutable std::vector<std::vector<float>> second_moments_;
};

} // namespace attention

#endif // ATTENTION_OPTIMIZER_H

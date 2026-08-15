#ifndef ATTENTION_OPTIMIZER_H
#define ATTENTION_OPTIMIZER_H

#include "attention/parameter_store.h"

#include <string>

namespace attention {

class SgdOptimizer {
public:
    explicit SgdOptimizer(float learning_rate, float gradient_clip_norm = 0.0f) noexcept
        : learning_rate_(learning_rate), gradient_clip_norm_(gradient_clip_norm) {}

    [[nodiscard]] float learning_rate() const noexcept { return learning_rate_; }
    [[nodiscard]] float gradient_clip_norm() const noexcept { return gradient_clip_norm_; }

    bool step(ParameterStore& parameters, std::string* error = nullptr) const;

private:
    float learning_rate_ = 0.0f;
    float gradient_clip_norm_ = 0.0f;
};

} // namespace attention

#endif // ATTENTION_OPTIMIZER_H

#include "attention/trainer.h"

#include <cmath>

namespace attention {

bool Trainer::step(const TransformerModel& model,
                   const std::vector<std::size_t>& token_ids,
                   std::size_t batch_size,
                   std::size_t sequence_length,
                   ParameterStore& parameters,
                   const SgdOptimizer& optimizer,
                   TrainingStepResult& result,
                   std::string* error) {
    if (batch_size == 0 || sequence_length < 2 || token_ids.size() != batch_size * sequence_length) {
        if (error != nullptr) *error = "training batch shape is invalid";
        return false;
    }
    if (!model.causal_loss(token_ids, batch_size, sequence_length, parameters, result.loss_before, error)) {
        return false;
    }
    if (!std::isfinite(result.loss_before)) {
        if (error != nullptr) *error = "pre-update loss is nonfinite";
        return false;
    }
    if (!model.backward(token_ids, batch_size, sequence_length, parameters, 1e-3f, error)) {
        return false;
    }
    if (!optimizer.step(parameters, error)) {
        return false;
    }
    if (!model.causal_loss(token_ids, batch_size, sequence_length, parameters, result.loss_after, error)) {
        return false;
    }
    if (!std::isfinite(result.loss_after)) {
        if (error != nullptr) *error = "post-update loss is nonfinite";
        return false;
    }
    return true;
}

} // namespace attention

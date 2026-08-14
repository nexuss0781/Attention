#ifndef ATTENTION_TRAINER_H
#define ATTENTION_TRAINER_H

#include "attention/optimizer.h"
#include "attention/transformer_model.h"

#include <cstddef>
#include <string>
#include <vector>

namespace attention {

struct TrainingStepResult {
    float loss_before = 0.0f;
    float loss_after = 0.0f;
};

class Trainer {
public:
    static bool step(const TransformerModel& model,
                     const std::vector<std::size_t>& token_ids,
                     std::size_t batch_size,
                     std::size_t sequence_length,
                     ParameterStore& parameters,
                     const SgdOptimizer& optimizer,
                     TrainingStepResult& result,
                     std::string* error = nullptr);
};

} // namespace attention

#endif // ATTENTION_TRAINER_H

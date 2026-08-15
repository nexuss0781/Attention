#ifndef ATTENTION_VALIDATION_H
#define ATTENTION_VALIDATION_H

#include "attention/parameter_store.h"
#include "attention/training_data.h"
#include "attention/transformer_model.h"

#include <cstddef>
#include <string>

namespace attention {

struct ValidationResult {
    float mean_loss = 0.0f;
    std::size_t batches = 0;
    std::size_t prediction_tokens = 0;
};

class ValidationEvaluator {
public:
    static bool evaluate(const TransformerModel& model,
                         TrainingBatchLoader& loader,
                         const ParameterStore& parameters,
                         ValidationResult& result,
                         std::string* error = nullptr);
};

} // namespace attention

#endif // ATTENTION_VALIDATION_H

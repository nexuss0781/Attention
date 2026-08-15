#include "attention/validation.h"

#include <cmath>
#include <limits>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool checked_prediction_count(const TrainingBatch& batch, std::size_t& count) {
    if (batch.batch_size == 0 || batch.sequence_length < 2 ||
        batch.sequence_length > std::numeric_limits<std::size_t>::max() / batch.batch_size) {
        return false;
    }
    count = batch.batch_size * (batch.sequence_length - 1);
    return count != 0;
}

} // namespace

bool ValidationEvaluator::evaluate(const TransformerModel& model,
                                   TrainingBatchLoader& loader,
                                   const ParameterStore& parameters,
                                   ValidationResult& result,
                                   std::string* error) {
    if (loader.batch_count() == 0 || !parameters.all_finite()) {
        set_error(error, "validation inputs are empty or nonfinite");
        return false;
    }
    loader.reset();
    double total_loss = 0.0;
    std::size_t prediction_tokens = 0;
    std::size_t batches = 0;
    while (!loader.exhausted()) {
        TrainingBatch batch;
        if (!loader.next(batch, error)) return false;
        std::size_t batch_predictions = 0;
        if (!checked_prediction_count(batch, batch_predictions)) {
            set_error(error, "validation batch shape is invalid");
            loader.reset();
            return false;
        }
        float batch_loss = 0.0f;
        if (!model.causal_loss(batch.token_ids, batch.batch_size, batch.sequence_length,
                               parameters, batch_loss, error)) {
            loader.reset();
            return false;
        }
        if (!std::isfinite(batch_loss) ||
            batch_predictions > std::numeric_limits<std::size_t>::max() - prediction_tokens) {
            set_error(error, "validation loss or prediction count is invalid");
            loader.reset();
            return false;
        }
        total_loss += static_cast<double>(batch_loss) * static_cast<double>(batch_predictions);
        if (!std::isfinite(total_loss)) {
            set_error(error, "validation loss accumulation is nonfinite");
            loader.reset();
            return false;
        }
        prediction_tokens += batch_predictions;
        ++batches;
    }
    loader.reset();
    if (prediction_tokens == 0) {
        set_error(error, "validation produced no prediction tokens");
        return false;
    }
    const double mean_loss = total_loss / static_cast<double>(prediction_tokens);
    if (!std::isfinite(mean_loss) || mean_loss > std::numeric_limits<float>::max()) {
        set_error(error, "validation mean loss is invalid");
        return false;
    }
    result.mean_loss = static_cast<float>(mean_loss);
    result.batches = batches;
    result.prediction_tokens = prediction_tokens;
    if (error != nullptr) error->clear();
    return true;
}

} // namespace attention

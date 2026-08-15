#ifndef ATTENTION_TRAINING_CHECKPOINT_H
#define ATTENTION_TRAINING_CHECKPOINT_H

#include "attention/checkpoint.h"
#include "attention/optimizer.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace attention {

struct TrainingProgress {
    std::string run_id;
    std::string dataset_id;
    std::string dataset_revision;
    std::uint64_t global_step = 0;
    std::uint64_t tokens_processed = 0;
    std::uint64_t next_batch_index = 0;
    float learning_rate = 0.0f;
};

class TrainingCheckpoint {
public:
    static bool serialize(const TransformerConfig& config,
                          const ParameterStore& parameters,
                          const TrainingProgress& progress,
                          std::string& output,
                          std::string* error = nullptr,
                          const TokenizerMetadata& tokenizer = TokenizerMetadata::byte_level_v1(),
                          const OptimizerState* optimizer_state = nullptr);

    static bool load(std::string_view input,
                     TransformerModel& model,
                     ParameterStore& parameters,
                     TrainingProgress& progress,
                     std::string* error = nullptr,
                     const TokenizerMetadata& expected_tokenizer = TokenizerMetadata::byte_level_v1(),
                     OptimizerState* optimizer_state = nullptr);
};

} // namespace attention

#endif // ATTENTION_TRAINING_CHECKPOINT_H

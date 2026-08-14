#ifndef ATTENTION_TRAINING_RUN_H
#define ATTENTION_TRAINING_RUN_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace attention {

struct TrainingRunMetadata {
    std::string run_id;
    std::string stage;
    std::string dataset_id;
    std::string dataset_revision;
    std::string source_checksums;
    std::string tokenizer_version;
    std::size_t tokenizer_vocabulary_size = 0;
    std::string architecture_serialization;
    std::string code_commit;
    std::uint64_t seed = 0;
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;
    float learning_rate = 0.0f;
};

struct TrainingLogRecord {
    std::uint64_t step = 0;
    std::uint64_t batch_index = 0;
    std::uint64_t token_offset = 0;
    std::uint64_t tokens_processed = 0;
    float loss_before = 0.0f;
    float loss_after = 0.0f;
    float learning_rate = 0.0f;
    float gradient_l2_norm = 0.0f;
};

class TrainingRunLogger {
public:
    bool initialize(const TrainingRunMetadata& metadata, std::string* error = nullptr);
    bool append(const TrainingLogRecord& record, std::string* error = nullptr);
    bool serialize(std::string& output, std::string* error = nullptr) const;

    [[nodiscard]] const TrainingRunMetadata& metadata() const noexcept;
    [[nodiscard]] const std::vector<TrainingLogRecord>& records() const noexcept;

private:
    TrainingRunMetadata metadata_;
    std::vector<TrainingLogRecord> records_;
    bool initialized_ = false;
};

} // namespace attention

#endif // ATTENTION_TRAINING_RUN_H

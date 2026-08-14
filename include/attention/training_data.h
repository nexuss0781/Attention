#ifndef ATTENTION_TRAINING_DATA_H
#define ATTENTION_TRAINING_DATA_H

#include <cstddef>
#include <string>
#include <vector>

namespace attention {

struct TrainingBatch {
    std::vector<std::size_t> token_ids;
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;
    std::size_t token_offset = 0;
};

class TrainingBatchLoader {
public:
    bool initialize(std::vector<std::size_t> token_stream,
                    std::size_t batch_size,
                    std::size_t sequence_length,
                    bool drop_remainder = true,
                    std::string* error = nullptr);

    bool next(TrainingBatch& batch, std::string* error = nullptr);
    void reset() noexcept;

    [[nodiscard]] bool exhausted() const noexcept;
    [[nodiscard]] std::size_t batch_count() const noexcept;
    [[nodiscard]] std::size_t batches_emitted() const noexcept;
    [[nodiscard]] std::size_t tokens_processed() const noexcept;
    [[nodiscard]] std::size_t batch_size() const noexcept;
    [[nodiscard]] std::size_t sequence_length() const noexcept;

private:
    std::vector<std::size_t> token_stream_;
    std::size_t batch_size_ = 0;
    std::size_t sequence_length_ = 0;
    std::size_t batch_token_count_ = 0;
    std::size_t batch_count_ = 0;
    std::size_t next_batch_ = 0;
    bool initialized_ = false;
};

} // namespace attention

#endif // ATTENTION_TRAINING_DATA_H

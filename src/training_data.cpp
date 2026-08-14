#include "attention/training_data.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool checked_product(std::size_t left, std::size_t right, std::size_t& result) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) return false;
    result = left * right;
    return true;
}

} // namespace

bool TrainingBatchLoader::initialize(std::vector<std::size_t> token_stream,
                                     std::size_t batch_size,
                                     std::size_t sequence_length,
                                     bool drop_remainder,
                                     std::string* error) {
    if (batch_size == 0 || sequence_length < 2) {
        set_error(error, "batch size must be positive and sequence length must be at least two");
        return false;
    }
    std::size_t batch_token_count = 0;
    if (!checked_product(batch_size, sequence_length, batch_token_count) || batch_token_count == 0) {
        set_error(error, "batch shape overflows or is empty");
        return false;
    }
    if (drop_remainder && token_stream.size() < batch_token_count) {
        set_error(error, "token stream is shorter than one complete batch");
        return false;
    }
    const std::size_t batch_count = token_stream.size() / batch_token_count;
    const std::size_t remainder = token_stream.size() % batch_token_count;
    std::size_t total_batch_count = batch_count;
    if (!drop_remainder && remainder != 0) {
        if (remainder < 2) {
            set_error(error, "non-dropping remainder is too short for causal training");
            return false;
        }
        ++total_batch_count;
    }
    token_stream_ = std::move(token_stream);
    batch_size_ = batch_size;
    sequence_length_ = sequence_length;
    batch_token_count_ = batch_token_count;
    batch_count_ = total_batch_count;
    next_batch_ = 0;
    initialized_ = true;
    return true;
}

bool TrainingBatchLoader::next(TrainingBatch& batch, std::string* error) {
    if (!initialized_) {
        set_error(error, "batch loader is not initialized");
        return false;
    }
    if (next_batch_ >= batch_count_) {
        set_error(error, "batch loader is exhausted");
        return false;
    }
    const std::size_t offset = next_batch_ * batch_token_count_;
    const std::size_t remaining = token_stream_.size() - offset;
    const std::size_t count = std::min(batch_token_count_, remaining);
    if (count != batch_token_count_) {
        batch.batch_size = 1;
        batch.sequence_length = count;
    } else {
        batch.batch_size = batch_size_;
        batch.sequence_length = sequence_length_;
    }
    batch.token_offset = offset;
    batch.token_ids.assign(token_stream_.begin() + static_cast<std::ptrdiff_t>(offset),
                           token_stream_.begin() + static_cast<std::ptrdiff_t>(offset + count));
    ++next_batch_;
    return true;
}

void TrainingBatchLoader::reset() noexcept {
    next_batch_ = 0;
}

bool TrainingBatchLoader::exhausted() const noexcept {
    return !initialized_ || next_batch_ >= batch_count_;
}

std::size_t TrainingBatchLoader::batch_count() const noexcept {
    return batch_count_;
}

std::size_t TrainingBatchLoader::batches_emitted() const noexcept {
    return next_batch_;
}

std::size_t TrainingBatchLoader::tokens_processed() const noexcept {
    return std::min(next_batch_ * batch_token_count_, token_stream_.size());
}

std::size_t TrainingBatchLoader::batch_size() const noexcept {
    return batch_size_;
}

std::size_t TrainingBatchLoader::sequence_length() const noexcept {
    return sequence_length_;
}

} // namespace attention

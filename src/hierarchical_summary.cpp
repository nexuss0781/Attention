#include "attention/hierarchical_summary.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_input(const Tensor& input) {
    return input.valid() && input.rank() == 3 &&
           input.data_type() == TensorDataType::F32 &&
           input.device() == TensorDevice::CPU &&
           input.layout() == TensorLayout::RowMajor &&
           input.all_finite();
}

bool multiply_fits(std::size_t left, std::size_t right) noexcept {
    return left == 0 || right <= std::numeric_limits<std::size_t>::max() / left;
}

} // namespace

bool HierarchicalSummaryState::reset(std::size_t logical_context_length,
                                     std::size_t level_count,
                                     std::size_t fanout,
                                     std::size_t batch_size,
                                     std::size_t hidden_size,
                                     std::string* error) noexcept {
    if (logical_context_length == 0 || level_count == 0 || fanout < 2 ||
        batch_size == 0 || hidden_size == 0) {
        set_error(error, "summary context, levels, fanout, batch size, and hidden size are invalid");
        return false;
    }
    if (!multiply_fits(batch_size, level_count) ||
        !multiply_fits(batch_size * level_count, hidden_size)) {
        set_error(error, "summary state size overflows size_t");
        return false;
    }
    const std::size_t summary_count = batch_size * level_count * hidden_size;
    if (!multiply_fits(batch_size, level_count)) {
        set_error(error, "summary count size overflows size_t");
        return false;
    }
    try {
        sums_.assign(summary_count, 0.0);
        counts_.assign(batch_size * level_count, 0);
        published_.assign(summary_count, 0.0f);
        carry_.assign(hidden_size, 0.0f);
    } catch (const std::bad_alloc&) {
        clear();
        set_error(error, "summary state allocation failed");
        return false;
    }
    logical_context_length_ = logical_context_length;
    level_count_ = level_count;
    fanout_ = fanout;
    batch_size_ = batch_size;
    hidden_size_ = hidden_size;
    tokens_processed_ = 0;
    if (error != nullptr) error->clear();
    return true;
}

bool HierarchicalSummaryState::append(const Tensor& input, std::string* error) {
    if (logical_context_length_ == 0 || level_count_ == 0 || fanout_ < 2 ||
        batch_size_ == 0 || hidden_size_ == 0) {
        set_error(error, "hierarchical summary state is not initialized");
        return false;
    }
    if (!valid_input(input)) {
        set_error(error, "summary input must be a finite F32 CPU row-major rank-3 tensor");
        return false;
    }
    const auto& shape = input.shape();
    const std::size_t chunk_batch = shape[0];
    const std::size_t chunk_length = shape[1];
    if (chunk_batch != batch_size_ || chunk_length == 0 || shape[2] != hidden_size_ ||
        tokens_processed_ > logical_context_length_ ||
        chunk_length > logical_context_length_ - tokens_processed_) {
        set_error(error, "summary chunk exceeds configured batch, hidden size, or logical context");
        return false;
    }
    for (std::size_t batch = 0; batch < batch_size_; ++batch) {
        for (std::size_t position = 0; position < chunk_length; ++position) {
            const float* row = input.data() + (batch * chunk_length + position) * hidden_size_;
            for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
                carry_[channel] = row[channel];
            }
            if (!accumulate(batch, 0, carry_.data(), error)) return false;
        }
    }
    tokens_processed_ += chunk_length;
    if (error != nullptr) error->clear();
    return true;
}

bool HierarchicalSummaryState::accumulate(std::size_t batch,
                                           std::size_t level,
                                           const float* values,
                                           std::string* error) {
    const std::size_t summary_offset = (batch * level_count_ + level) * hidden_size_;
    const std::size_t count_offset = batch * level_count_ + level;
    const std::size_t previous_count = counts_[count_offset];
    for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
        sums_[summary_offset + channel] += static_cast<double>(values[channel]);
    }
    if (previous_count >= fanout_ - 1) {
        const double denominator = static_cast<double>(previous_count + 1);
        for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
            carry_[channel] = static_cast<float>(sums_[summary_offset + channel] / denominator);
            if (!std::isfinite(carry_[channel])) {
                set_error(error, "summary aggregate is not finite");
                return false;
            }
            published_[summary_offset + channel] = carry_[channel];
            sums_[summary_offset + channel] = 0.0;
        }
        counts_[count_offset] = 0;
        if (level + 1 < level_count_) return accumulate(batch, level + 1, carry_.data(), error);
    } else {
        ++counts_[count_offset];
    }
    return true;
}

bool HierarchicalSummaryState::snapshot(Tensor& output, std::string* error) const {
    if (logical_context_length_ == 0 || level_count_ == 0 || batch_size_ == 0 || hidden_size_ == 0) {
        set_error(error, "hierarchical summary state is not initialized");
        return false;
    }
    if (!output.reset({batch_size_, level_count_, hidden_size_},
                      TensorDataType::F32, TensorDevice::CPU, error)) return false;
    for (std::size_t batch = 0; batch < batch_size_; ++batch) {
        for (std::size_t level = 0; level < level_count_; ++level) {
            const std::size_t count = counts_[batch * level_count_ + level];
            const std::size_t offset = (batch * level_count_ + level) * hidden_size_;
            if (count == 0) {
                for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
                    output.data()[offset + channel] = published_[offset + channel];
                }
            } else {
                const double denominator = static_cast<double>(count);
                for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
                    output.data()[offset + channel] = static_cast<float>(sums_[offset + channel] / denominator);
                }
            }
        }
    }
    if (!output.all_finite()) {
        set_error(error, "summary snapshot is not finite");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

void HierarchicalSummaryState::clear() noexcept {
    logical_context_length_ = 0;
    level_count_ = 0;
    fanout_ = 0;
    batch_size_ = 0;
    hidden_size_ = 0;
    tokens_processed_ = 0;
    sums_.clear();
    counts_.clear();
    published_.clear();
    carry_.clear();
}

std::size_t HierarchicalSummaryState::logical_context_length() const noexcept { return logical_context_length_; }
std::size_t HierarchicalSummaryState::level_count() const noexcept { return level_count_; }
std::size_t HierarchicalSummaryState::fanout() const noexcept { return fanout_; }
std::size_t HierarchicalSummaryState::batch_size() const noexcept { return batch_size_; }
std::size_t HierarchicalSummaryState::hidden_size() const noexcept { return hidden_size_; }
std::size_t HierarchicalSummaryState::tokens_processed() const noexcept { return tokens_processed_; }

std::size_t HierarchicalSummaryState::state_bytes() const noexcept {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (sums_.size() > maximum / sizeof(double) || counts_.size() > maximum / sizeof(std::size_t) ||
        published_.size() > maximum / sizeof(float) || carry_.size() > maximum / sizeof(float)) return 0;
    const std::size_t sums_bytes = sums_.size() * sizeof(double);
    const std::size_t counts_bytes = counts_.size() * sizeof(std::size_t);
    const std::size_t published_bytes = published_.size() * sizeof(float);
    const std::size_t carry_bytes = carry_.size() * sizeof(float);
    if (counts_bytes > maximum - sums_bytes || published_bytes > maximum - sums_bytes - counts_bytes ||
        carry_bytes > maximum - sums_bytes - counts_bytes - published_bytes) return 0;
    return sums_bytes + counts_bytes + published_bytes + carry_bytes;
}

} // namespace attention

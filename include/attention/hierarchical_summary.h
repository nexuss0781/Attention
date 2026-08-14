#ifndef ATTENTION_HIERARCHICAL_SUMMARY_H
#define ATTENTION_HIERARCHICAL_SUMMARY_H

#include "attention/tensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace attention {

class HierarchicalSummaryState {
public:
    bool reset(std::size_t logical_context_length,
               std::size_t level_count,
               std::size_t fanout,
               std::size_t batch_size,
               std::size_t hidden_size,
               std::string* error = nullptr) noexcept;

    bool append(const Tensor& input, std::string* error = nullptr);

    bool snapshot(Tensor& output, std::string* error = nullptr) const;

    void clear() noexcept;
    [[nodiscard]] std::size_t logical_context_length() const noexcept;
    [[nodiscard]] std::size_t level_count() const noexcept;
    [[nodiscard]] std::size_t fanout() const noexcept;
    [[nodiscard]] std::size_t batch_size() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] std::size_t tokens_processed() const noexcept;
    [[nodiscard]] std::size_t state_bytes() const noexcept;

private:
    bool accumulate(std::size_t batch,
                    std::size_t level,
                    const float* values,
                    std::string* error);

    std::size_t logical_context_length_ = 0;
    std::size_t level_count_ = 0;
    std::size_t fanout_ = 0;
    std::size_t batch_size_ = 0;
    std::size_t hidden_size_ = 0;
    std::size_t tokens_processed_ = 0;
    std::vector<double> sums_;
    std::vector<std::size_t> counts_;
    std::vector<float> published_;
    std::vector<float> carry_;
};

} // namespace attention

#endif // ATTENTION_HIERARCHICAL_SUMMARY_H

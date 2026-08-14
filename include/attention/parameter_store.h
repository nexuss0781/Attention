#ifndef ATTENTION_PARAMETER_STORE_H
#define ATTENTION_PARAMETER_STORE_H

#include "attention/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace attention {

struct Parameter {
    std::string name;
    Tensor value;
    Tensor gradient;
};

class ParameterStore {
public:
    bool add(std::string name,
             const std::vector<std::size_t>& shape,
             std::string* error = nullptr);

    bool initialize(std::uint64_t seed, std::string* error = nullptr);
    void clear_gradients() noexcept;

    [[nodiscard]] const Parameter* find(const std::string& name) const noexcept;
    [[nodiscard]] Parameter* find(const std::string& name) noexcept;
    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool all_finite() const noexcept;
    [[nodiscard]] float gradient_l2_norm() const noexcept;

    [[nodiscard]] std::vector<Parameter>& parameters() noexcept;
    [[nodiscard]] const std::vector<Parameter>& parameters() const noexcept;

private:
    std::vector<Parameter> parameters_;
};

} // namespace attention

#endif // ATTENTION_PARAMETER_STORE_H

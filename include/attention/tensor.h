#ifndef ATTENTION_TENSOR_H
#define ATTENTION_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace attention {

enum class TensorDataType : std::uint8_t {
    F32 = 0,
};

enum class TensorDevice : std::uint8_t {
    CPU = 0,
};

enum class TensorLayout : std::uint8_t {
    RowMajor = 0,
};

class Tensor {
public:
    Tensor() = default;

    bool reset(std::vector<std::size_t> shape,
               TensorDataType data_type = TensorDataType::F32,
               TensorDevice device = TensorDevice::CPU,
               std::string* error = nullptr);

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept;
    [[nodiscard]] TensorDataType data_type() const noexcept;
    [[nodiscard]] TensorDevice device() const noexcept;
    [[nodiscard]] TensorLayout layout() const noexcept;
    [[nodiscard]] float* data() noexcept;
    [[nodiscard]] const float* data() const noexcept;
    [[nodiscard]] bool all_finite() const noexcept;

    void fill(float value) noexcept;

private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<float> data_;
    TensorDataType data_type_ = TensorDataType::F32;
    TensorDevice device_ = TensorDevice::CPU;
    TensorLayout layout_ = TensorLayout::RowMajor;
};

} // namespace attention

#endif // ATTENTION_TENSOR_H

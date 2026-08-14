#include "attention/residual_connection.h"

#include <cmath>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_activation(const Tensor& tensor) {
    return tensor.valid() && tensor.rank() == 3 &&
           tensor.data_type() == TensorDataType::F32 &&
           tensor.device() == TensorDevice::CPU &&
           tensor.layout() == TensorLayout::RowMajor &&
           tensor.all_finite();
}

bool compatible(const Tensor& left, const Tensor& right) {
    return valid_activation(left) && valid_activation(right) &&
           left.shape() == right.shape();
}

} // namespace

bool ResidualConnection::add(const Tensor& main,
                             const Tensor& residual,
                             Tensor& output,
                             std::string* error) const {
    if (!compatible(main, residual)) {
        set_error(error, "residual inputs must be finite compatible rank-3 tensors");
        return false;
    }
    if (&output == &main || &output == &residual) {
        set_error(error, "out-of-place residual output must be distinct from inputs");
        return false;
    }
    if (!output.reset(main.shape(), TensorDataType::F32, TensorDevice::CPU, error)) {
        return false;
    }
    for (std::size_t index = 0; index < main.size(); ++index) {
        const float value = main.data()[index] + residual.data()[index];
        if (!std::isfinite(value)) {
            set_error(error, "residual sum contains NaN or infinity");
            return false;
        }
        output.data()[index] = value;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool ResidualConnection::add_in_place(Tensor& target,
                                      const Tensor& residual,
                                      std::string* error) const {
    if (!compatible(target, residual)) {
        set_error(error, "residual inputs must be finite compatible rank-3 tensors");
        return false;
    }
    for (std::size_t index = 0; index < target.size(); ++index) {
        const float value = target.data()[index] + residual.data()[index];
        if (!std::isfinite(value)) {
            set_error(error, "residual sum contains NaN or infinity");
            return false;
        }
        target.data()[index] = value;
    }
    if (error != nullptr) error->clear();
    return true;
}

} // namespace attention

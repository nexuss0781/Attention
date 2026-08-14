#ifndef ATTENTION_RESIDUAL_CONNECTION_H
#define ATTENTION_RESIDUAL_CONNECTION_H

#include "attention/tensor.h"

#include <string>

namespace attention {

class ResidualConnection {
public:
    bool add(const Tensor& main,
             const Tensor& residual,
             Tensor& output,
             std::string* error = nullptr) const;

    bool add_in_place(Tensor& target,
                      const Tensor& residual,
                      std::string* error = nullptr) const;
};

} // namespace attention

#endif // ATTENTION_RESIDUAL_CONNECTION_H

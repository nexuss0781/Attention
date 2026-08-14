#ifndef ATTENTION_SMAO_LINEAR_BOUNDARY_H
#define ATTENTION_SMAO_LINEAR_BOUNDARY_H

#include "attention/tensor.h"
#include "smao_phase1/core/types.h"

#include <cstddef>
#include <string>

namespace attention {

struct SMAOLinearBoundaryReport {
    bool accepted = false;
    bool whitened_coordinates_consumed = false;
    bool scalar_factors_consumed = false;
    std::size_t sequence_length = 0;
    std::size_t hidden_size = 0;
    std::string reason;
};

class SMAOLinearBoundary {
public:
    bool adapt(const smao::Phase1Output& smao_output,
               Tensor& query,
               Tensor& key,
               SMAOLinearBoundaryReport* report = nullptr) const;
};

} // namespace attention

#endif // ATTENTION_SMAO_LINEAR_BOUNDARY_H

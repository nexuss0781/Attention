#ifndef ATTENTION_ANALYTICAL_BACKWARD_H
#define ATTENTION_ANALYTICAL_BACKWARD_H

#include "attention/parameter_store.h"
#include "attention/transformer_model.h"

#include <cstddef>
#include <string>
#include <vector>

namespace attention {

bool analytical_backward(const TransformerModel& model,
                         const std::vector<std::size_t>& token_ids,
                         std::size_t batch_size,
                         std::size_t sequence_length,
                         ParameterStore& parameters,
                         std::string* error = nullptr);

} // namespace attention

#endif // ATTENTION_ANALYTICAL_BACKWARD_H

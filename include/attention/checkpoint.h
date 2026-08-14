#ifndef ATTENTION_CHECKPOINT_H
#define ATTENTION_CHECKPOINT_H

#include "attention/parameter_store.h"
#include "attention/transformer_model.h"

#include <string>
#include <string_view>

namespace attention {

class TransformerCheckpoint {
public:
    static bool serialize(const TransformerConfig& config,
                          const ParameterStore& parameters,
                          std::string& output,
                          std::string* error = nullptr);

    static bool load(std::string_view input,
                     TransformerModel& model,
                     ParameterStore& parameters,
                     std::string* error = nullptr);
};

} // namespace attention

#endif // ATTENTION_CHECKPOINT_H

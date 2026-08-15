#include "attention/checkpoint.h"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: attention_initialize_checkpoint OUTPUT_CHECKPOINT\n";
        return 2;
    }
    attention::TransformerConfig config;
    config.vocabulary_size = 260;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    attention::TransformerModel model;
    attention::ParameterStore parameters;
    std::string error;
    if (!model.register_parameters(config, parameters, &error) ||
        !parameters.initialize(17, &error)) {
        std::cerr << "bootstrap initialization failed: " << error << '\n';
        return 1;
    }
    std::string checkpoint;
    if (!attention::TransformerCheckpoint::serialize(config, parameters, checkpoint, &error)) {
        std::cerr << "bootstrap checkpoint serialization failed: " << error << '\n';
        return 1;
    }
    std::ofstream output(argv[1], std::ios::binary);
    if (!output) {
        std::cerr << "cannot write bootstrap checkpoint: " << argv[1] << '\n';
        return 1;
    }
    output << checkpoint;
    std::cout << "wrote bootstrap checkpoint: " << argv[1] << '\n';
    return 0;
}

#include "attention/checkpoint.h"
#include "attention/trainer.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
    attention::TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;

    attention::TransformerModel model;
    attention::ParameterStore parameters;
    std::string error;
    if (!model.register_parameters(config, parameters, &error) || !parameters.initialize(17, &error)) {
        std::cerr << "initialization failed: " << error << '\n';
        return 1;
    }
    const std::vector<std::size_t> tokens{0, 1, 2, 0};
    attention::SgdOptimizer optimizer(0.05f);
    std::cout << std::setprecision(9) << "step,loss_before,loss_after\n";
    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int step = 0; step < 4; ++step) {
        attention::TrainingStepResult result;
        if (!attention::Trainer::step(model, tokens, 1, 4, parameters, optimizer, result, &error)) {
            std::cerr << "training step failed: " << error << '\n';
            return 1;
        }
        if (step == 0) first_loss = result.loss_before;
        last_loss = result.loss_after;
        std::cout << step << ',' << result.loss_before << ',' << result.loss_after << '\n';
    }
    if (!(last_loss < first_loss)) {
        std::cerr << "loss did not decrease" << '\n';
        return 1;
    }

    std::string checkpoint;
    if (!attention::TransformerCheckpoint::serialize(config, parameters, checkpoint, &error)) {
        std::cerr << "checkpoint serialization failed: " << error << '\n';
        return 1;
    }
    attention::TransformerModel reloaded;
    attention::ParameterStore reloaded_parameters;
    if (!attention::TransformerCheckpoint::load(checkpoint, reloaded, reloaded_parameters, &error)) {
        std::cerr << "checkpoint reload failed: " << error << '\n';
        return 1;
    }
    float reloaded_loss = 0.0f;
    if (!reloaded.causal_loss(tokens, 1, 4, reloaded_parameters, reloaded_loss, &error)) {
        std::cerr << "reloaded loss failed: " << error << '\n';
        return 1;
    }
    std::cout << "reloaded_loss," << reloaded_loss << '\n';
    return 0;
}

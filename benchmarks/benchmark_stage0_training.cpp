#include "attention/checkpoint.h"
#include "attention/trainer.h"
#include "attention/training_data.h"
#include "attention/training_run.h"
#include "attention/training_checkpoint.h"
#include "attention/validation.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool load_tokens(const std::string& path, std::vector<std::size_t>& tokens, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open token stream: " + path;
        return false;
    }
    std::size_t token = 0;
    while (input >> token) tokens.push_back(token);
    if (!input.eof()) {
        error = "token stream contains a non-integer value";
        return false;
    }
    if (tokens.empty()) {
        error = "token stream is empty";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    attention::TransformerConfig config;
    const bool file_backed = argc >= 2;
    config.vocabulary_size = file_backed ? 260 : 3;
    config.context_length = file_backed ? 32 : 4;
    config.layer_count = file_backed ? 2 : 1;
    config.hidden_size = file_backed ? 32 : 2;
    config.attention_head_count = file_backed ? 4 : 1;
    config.feed_forward_size = file_backed ? 128 : 4;

    std::vector<std::size_t> token_stream;
    std::string error;
    std::string dataset_id = "stage0.synthetic_debug";
    if (argc >= 2) {
        if (!load_tokens(argv[1], token_stream, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        dataset_id = "stage0.token_stream_file";
    } else {
        const std::vector<std::size_t> pattern{0, 1, 2, 0};
        for (int repeat = 0; repeat < 4; ++repeat) {
            token_stream.insert(token_stream.end(), pattern.begin(), pattern.end());
        }
    }

    attention::TrainingBatchLoader loader;
    const std::size_t training_sequence_length = file_backed ? 32 : 4;
    if (!loader.initialize(std::move(token_stream), 1, training_sequence_length, true, &error)) {
        std::cerr << "batch loader initialization failed: " << error << '\n';
        return 1;
    }

    attention::TransformerModel model;
    attention::ParameterStore parameters;
    if (argc >= 7) {
        std::ifstream parent_input(argv[6], std::ios::binary);
        if (!parent_input) {
            std::cerr << "cannot open parent checkpoint: " << argv[6] << '\n';
            return 1;
        }
        const std::string parent_payload((std::istreambuf_iterator<char>(parent_input)),
                                         std::istreambuf_iterator<char>());
        if (!attention::TransformerCheckpoint::load(parent_payload, model, parameters, &error)) {
            std::cerr << "parent checkpoint load failed: " << error << '\n';
            return 1;
        }
        config = model.config();
    } else if (!model.register_parameters(config, parameters, &error) || !parameters.initialize(17, &error)) {
        std::cerr << "initialization failed: " << error << '\n';
        return 1;
    }
    std::string architecture;
    if (!config.serialize(architecture, &error)) {
        std::cerr << "architecture serialization failed: " << error << '\n';
        return 1;
    }
    attention::TrainingRunMetadata metadata;
    const char* session_id = std::getenv("ATTENTION_SESSION_ID");
    const char* stage_id = std::getenv("ATTENTION_STAGE_ID");
    const char* dataset_id_override = std::getenv("ATTENTION_DATASET_ID");
    const char* dataset_revision = std::getenv("ATTENTION_DATASET_REVISION");
    metadata.run_id = session_id != nullptr ? session_id : "stage0-debug-training";
    metadata.stage = stage_id != nullptr ? stage_id : "stage0_debug";
    metadata.dataset_id = dataset_id_override != nullptr ? dataset_id_override : dataset_id;
    metadata.dataset_revision = dataset_revision != nullptr ? dataset_revision : "fixed-local-v1";
    const char* source_checksum = std::getenv("ATTENTION_SOURCE_SHA256");
    metadata.source_checksums = source_checksum != nullptr ? source_checksum : "not-provided";
    metadata.tokenizer_version = "attention.byte_utf8.v1";
    metadata.tokenizer_vocabulary_size = 260;
    metadata.architecture_serialization = architecture;
    const char* commit = std::getenv("ATTENTION_CODE_COMMIT");
    metadata.code_commit = commit != nullptr ? commit : "unknown";
    metadata.seed = 17;
    metadata.batch_size = loader.batch_size();
    metadata.sequence_length = loader.sequence_length();
    metadata.learning_rate = file_backed ? 0.005f : 0.01f;
    const char* learning_rate_text = std::getenv("ATTENTION_LEARNING_RATE");
    if (learning_rate_text != nullptr) {
        try {
            metadata.learning_rate = std::stof(learning_rate_text);
        } catch (...) {
            std::cerr << "ATTENTION_LEARNING_RATE is invalid\n";
            return 1;
        }
    }
    if (!std::isfinite(metadata.learning_rate) || !(metadata.learning_rate > 0.0f)) {
        std::cerr << "ATTENTION_LEARNING_RATE must be finite and positive\n";
        return 1;
    }

    attention::TrainingRunLogger logger;
    if (!logger.initialize(metadata, &error)) {
        std::cerr << "run metadata initialization failed: " << error << '\n';
        return 1;
    }
    std::vector<std::size_t> validation_tokens;
    if (argc >= 5) {
        if (!load_tokens(argv[4], validation_tokens, error)) {
            std::cerr << error << '\n';
            return 1;
        }
    } else {
        validation_tokens = {1, 2, 0, 1, 2, 0, 1, 2};
    }
    attention::TrainingBatchLoader validation_loader;
    if (!validation_loader.initialize(std::move(validation_tokens), 1, training_sequence_length, true, &error)) {
        std::cerr << "validation loader initialization failed: " << error << '\n';
        return 1;
    }
    attention::SgdOptimizer optimizer(metadata.learning_rate);
    std::uint64_t max_steps = 0;
    const char* max_steps_text = std::getenv("ATTENTION_MAX_STEPS");
    if (max_steps_text != nullptr) {
        try {
            max_steps = std::stoull(max_steps_text);
        } catch (...) {
            std::cerr << "ATTENTION_MAX_STEPS is invalid\n";
            return 1;
        }
        if (max_steps == 0) {
            std::cerr << "ATTENTION_MAX_STEPS must be positive\n";
            return 1;
        }
    }
    const std::size_t tokens_per_epoch = loader.batch_count() * loader.batch_size() * loader.sequence_length();
    std::cout << std::setprecision(9) << "step,loss_before,loss_after,gradient_l2_norm,validation_loss\n";
    float first_loss = 0.0f;
    float last_loss = 0.0f;
    float final_validation_loss = 0.0f;
    attention::TrainingBatch last_batch;
    std::uint64_t step = 0;
    while (max_steps == 0 || step < max_steps) {
        loader.reset();
        while (!loader.exhausted() && (max_steps == 0 || step < max_steps)) {
            attention::TrainingBatch batch;
            if (!loader.next(batch, &error)) {
                std::cerr << "batch loading failed: " << error << '\n';
                return 1;
            }
            attention::TrainingStepResult result;
            if (!attention::Trainer::step(model, batch.token_ids, batch.batch_size,
                                          batch.sequence_length, parameters, optimizer, result, &error)) {
                std::cerr << "training step failed: " << error << '\n';
                return 1;
            }
            const float gradient_norm = parameters.gradient_l2_norm();
            attention::ValidationResult validation_result;
            if (!attention::ValidationEvaluator::evaluate(model, validation_loader, parameters,
                                                          validation_result, &error)) {
                std::cerr << "validation failed: " << error << '\n';
                return 1;
            }
            final_validation_loss = validation_result.mean_loss;
            attention::TrainingLogRecord record;
            const std::uint64_t epoch = tokens_per_epoch == 0 ? 0 : step / loader.batch_count();
            record.step = step;
            record.batch_index = epoch * loader.batch_count() + loader.batches_emitted() - 1;
            record.token_offset = epoch * tokens_per_epoch + batch.token_offset;
            record.tokens_processed = epoch * tokens_per_epoch + loader.tokens_processed();
            record.loss_before = result.loss_before;
            record.loss_after = result.loss_after;
            record.learning_rate = metadata.learning_rate;
            record.gradient_l2_norm = gradient_norm;
            record.validation_loss = validation_result.mean_loss;
            if (!logger.append(record, &error)) {
                std::cerr << "run logging failed: " << error << '\n';
                return 1;
            }
            if (step == 0) first_loss = result.loss_before;
            last_loss = result.loss_after;
            last_batch = batch;
            std::cout << step << ',' << result.loss_before << ',' << result.loss_after << ','
                      << gradient_norm << ',' << validation_result.mean_loss << '\n';
            ++step;
        }
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
    if (!reloaded.causal_loss(last_batch.token_ids, last_batch.batch_size,
                              last_batch.sequence_length, reloaded_parameters,
                              reloaded_loss, &error)) {
        std::cerr << "reloaded loss failed: " << error << '\n';
        return 1;
    }
    attention::TrainingProgress progress;
    progress.run_id = metadata.run_id;
    progress.dataset_id = metadata.dataset_id;
    progress.dataset_revision = metadata.dataset_revision;
    progress.global_step = step;
    progress.tokens_processed = step * loader.batch_size() * loader.sequence_length();
    progress.next_batch_index = loader.batches_emitted();
    progress.learning_rate = metadata.learning_rate;
    std::string training_checkpoint;
    if (!attention::TrainingCheckpoint::serialize(config, parameters, progress,
                                                 training_checkpoint, &error)) {
        std::cerr << "training checkpoint serialization failed: " << error << '\n';
        return 1;
    }
    const std::string training_checkpoint_path = argc >= 4
                                                   ? argv[3]
                                                   : "/tmp/attention_stage0_training_state.chk";
    const std::string model_checkpoint_path = argc >= 6
                                               ? argv[5]
                                               : "/tmp/attention_stage0_model.checkpoint";
    {
        std::ofstream state_output(training_checkpoint_path, std::ios::binary);
        if (!state_output) {
            std::cerr << "cannot write training checkpoint: " << training_checkpoint_path << '\n';
            return 1;
        }
        state_output << training_checkpoint;
    }
    {
        std::ofstream model_output(model_checkpoint_path, std::ios::binary);
        if (!model_output) {
            std::cerr << "cannot write model checkpoint: " << model_checkpoint_path << '\n';
            return 1;
        }
        model_output << checkpoint;
    }
    std::ifstream state_input(training_checkpoint_path, std::ios::binary);
    if (!state_input) {
        std::cerr << "cannot read training checkpoint: " << training_checkpoint_path << '\n';
        return 1;
    }
    const std::string persisted_training_checkpoint(
        (std::istreambuf_iterator<char>(state_input)), std::istreambuf_iterator<char>());
    attention::TransformerModel resumed_model;
    attention::ParameterStore resumed_parameters;
    attention::TrainingProgress resumed_progress;
    if (!attention::TrainingCheckpoint::load(persisted_training_checkpoint, resumed_model,
                                             resumed_parameters, resumed_progress, &error)) {
        std::cerr << "training checkpoint reload failed: " << error << '\n';
        return 1;
    }
    float resumed_loss = 0.0f;
    if (!resumed_model.causal_loss(last_batch.token_ids, last_batch.batch_size,
                                   last_batch.sequence_length, resumed_parameters,
                                   resumed_loss, &error)) {
        std::cerr << "resumed loss failed: " << error << '\n';
        return 1;
    }
    if (resumed_loss != reloaded_loss || resumed_progress.global_step != progress.global_step) {
        std::cerr << "training checkpoint continuity check failed\n";
        return 1;
    }
    std::string run_json;
    if (!logger.serialize(run_json, &error)) {
        std::cerr << "run serialization failed: " << error << '\n';
        return 1;
    }
    const std::string output_path = argc >= 3 ? argv[2] : "/tmp/attention_stage0_training_run.json";
    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "cannot write run log: " << output_path << '\n';
        return 1;
    }
    output << run_json;
    std::cout << "reloaded_loss," << reloaded_loss << '\n';
    std::cout << "resumed_loss," << resumed_loss << '\n';
    std::cout << "validation_loss," << final_validation_loss << '\n';
    std::cout << "run_log," << output_path << '\n';
    std::cout << "training_checkpoint," << training_checkpoint_path << '\n';
    std::cout << "model_checkpoint," << model_checkpoint_path << '\n';
    return 0;
}

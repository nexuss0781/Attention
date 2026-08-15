#include "attention/checkpoint.h"
#include "attention/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

bool read_file(const std::string& path, std::string& output, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open checkpoint: " + path;
        return false;
    }
    output.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if (output.empty()) {
        error = "checkpoint is empty";
        return false;
    }
    return true;
}

std::string escape(const std::string& value) {
    std::string output;
    for (const unsigned char character : value) {
        if (character == '\\') output += "\\\\";
        else if (character == '\"') output += "\\\"";
        else if (character == '\n') output += "\\n";
        else if (character >= 0x20 && character <= 0x7e) output.push_back(static_cast<char>(character));
        else {
            char buffer[8]{};
            std::snprintf(buffer, sizeof(buffer), "\\x%02x", character);
            output += buffer;
        }
    }
    return output;
}

std::string token_list(const std::vector<std::size_t>& tokens) {
    std::string output;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) output.push_back(' ');
        output += std::to_string(tokens[index]);
    }
    return output;
}

bool greedy_continuation(const attention::TransformerModel& model,
                         const attention::ParameterStore& parameters,
                         std::vector<std::size_t>& tokens,
                         std::size_t new_tokens,
                         std::string& error) {
    const std::size_t context = model.context_length();
    const std::size_t vocabulary = model.vocabulary_size();
    for (std::size_t generated = 0; generated < new_tokens; ++generated) {
        const std::size_t begin = tokens.size() > context ? tokens.size() - context : 0;
        const std::vector<std::size_t> window(tokens.begin() + static_cast<std::ptrdiff_t>(begin), tokens.end());
        attention::Tensor logits;
        if (!model.forward(window, 1, window.size(), parameters, logits, &error)) return false;
        const std::size_t position = window.size() - 1;
        const float* row = logits.data() + position * vocabulary;
        std::size_t best = 0;
        for (std::size_t candidate = 1; candidate < vocabulary; ++candidate) {
            if (row[candidate] > row[best]) best = candidate;
        }
        tokens.push_back(best);
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: attention_stage0_evaluation CHECKPOINT [OUTPUT_REPORT] [NEW_TOKENS]\n";
        return 2;
    }
    const std::string checkpoint_path = argv[1];
    const std::string output_path = argc >= 3 ? argv[2] : "/tmp/attention_stage0_evaluation.tsv";
    const std::size_t new_tokens = argc >= 4 ? static_cast<std::size_t>(std::stoul(argv[3])) : 8;
    if (new_tokens == 0 || new_tokens > 64) {
        std::cerr << "new token count must be in [1,64]\n";
        return 2;
    }

    std::string payload;
    std::string error;
    if (!read_file(checkpoint_path, payload, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    attention::TransformerModel model;
    attention::ParameterStore parameters;
    if (!attention::TransformerCheckpoint::load(payload, model, parameters, &error)) {
        std::cerr << "checkpoint load failed: " << error << '\n';
        return 1;
    }
    attention::ByteLevelTokenizer tokenizer;
    const std::vector<std::string> prompts{
        "The sky is",
        "Once upon a time",
        "A cat sat on the",
        "The capital of France is",
    };
    std::ofstream report(output_path);
    if (!report) {
        std::cerr << "cannot write evaluation report: " << output_path << '\n';
        return 1;
    }
    report << "format\tattention.stage0_evaluation.v1\n";
    report << "checkpoint\t" << checkpoint_path << "\n";
    report << "vocabulary_size\t" << model.vocabulary_size() << "\n";
    report << "context_length\t" << model.context_length() << "\n";
    report << "new_tokens\t" << new_tokens << "\n";
    report << "prompt\twindow_loss\tgenerated_token_ids\n";
    std::cout << "prompt\twindow_loss\tgenerated_token_ids\n";
    for (const std::string& prompt : prompts) {
        std::vector<std::size_t> tokens;
        if (!tokenizer.encode(prompt, tokens, true, false, &error)) {
            std::cerr << "tokenization failed: " << error << '\n';
            return 1;
        }
        const std::size_t context = model.context_length();
        const std::size_t begin = tokens.size() > context ? tokens.size() - context : 0;
        const std::vector<std::size_t> window(tokens.begin() + static_cast<std::ptrdiff_t>(begin), tokens.end());
        float loss = 0.0f;
        if (window.size() < 2 || !model.causal_loss(window, 1, window.size(), parameters, loss, &error)) {
            std::cerr << "prompt loss failed: " << error << '\n';
            return 1;
        }
        if (!greedy_continuation(model, parameters, tokens, new_tokens, error)) {
            std::cerr << "generation failed: " << error << '\n';
            return 1;
        }
        const std::string escaped_prompt = escape(prompt);
        const std::string generated = token_list(std::vector<std::size_t>(tokens.begin() + static_cast<std::ptrdiff_t>(tokens.size() - new_tokens), tokens.end()));
        report << escaped_prompt << '\t' << std::setprecision(9) << loss << '\t' << generated << '\n';
        std::cout << escaped_prompt << '\t' << std::setprecision(9) << loss << '\t' << generated << '\n';
    }
    return 0;
}

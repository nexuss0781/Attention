#include "attention/checkpoint.h"
#include "attention/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
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
        else if (character == '\t') output += "\\t";
        else if (character >= 0x20 && character <= 0x7e) output.push_back(static_cast<char>(character));
        else {
            char buffer[8]{};
            std::snprintf(buffer, sizeof(buffer), "\\x%02x", character);
            output += buffer;
        }
    }
    return output;
}

struct EvaluationCase {
    std::string case_id;
    std::string document_id;
    std::string prompt;
    std::string expected_continuation;
};

bool split_case_line(const std::string& line, EvaluationCase& output) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (fields.size() != 4) return false;
    output.case_id = std::move(fields[0]);
    output.document_id = std::move(fields[1]);
    output.prompt = std::move(fields[2]);
    output.expected_continuation = std::move(fields[3]);
    return !output.case_id.empty() && !output.prompt.empty();
}

bool read_cases(const std::string& path, std::vector<EvaluationCase>& cases, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open evaluation cases: " + path;
        return false;
    }
    std::string line;
    if (!std::getline(input, line) || line != "case_id\tdocument_id\tprompt\texpected_continuation") {
        error = "evaluation cases header is invalid";
        return false;
    }
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        EvaluationCase item;
        if (!split_case_line(line, item)) {
            error = "evaluation case row is invalid";
            return false;
        }
        cases.push_back(std::move(item));
    }
    if (cases.empty()) {
        error = "evaluation cases are empty";
        return false;
    }
    return true;
}

std::size_t unique_count(const std::vector<std::size_t>& tokens) {
    std::vector<std::size_t> sorted = tokens;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    return sorted.size();
}

std::size_t max_repeated_run(const std::vector<std::size_t>& tokens) {
    if (tokens.empty()) return 0;
    std::size_t best = 1;
    std::size_t current = 1;
    for (std::size_t index = 1; index < tokens.size(); ++index) {
        current = tokens[index] == tokens[index - 1] ? current + 1 : 1;
        best = std::max(best, current);
    }
    return best;
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
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: attention_stage0_evaluation CHECKPOINT [OUTPUT_REPORT] [NEW_TOKENS] [CASES_TSV]\n";
        return 2;
    }
    const std::string checkpoint_path = argv[1];
    const std::string output_path = argc >= 3 ? argv[2] : "/tmp/attention_stage0_evaluation.tsv";
    const std::size_t new_tokens = argc >= 4 ? static_cast<std::size_t>(std::stoul(argv[3])) : 8;
    const std::string cases_path = argc >= 5 ? argv[4] : "";
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
    std::vector<EvaluationCase> cases;
    if (!cases_path.empty()) {
        if (!read_cases(cases_path, cases, error)) {
            std::cerr << error << '\n';
            return 1;
        }
    } else {
        const std::vector<std::string> prompts{
            "The sky is",
            "Once upon a time",
            "A cat sat on the",
            "The capital of France is",
        };
        for (std::size_t index = 0; index < prompts.size(); ++index) {
            cases.push_back(EvaluationCase{"fixed-" + std::to_string(index), "", prompts[index], ""});
        }
    }
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
    report << "cases\t" << cases.size() << "\n";
    report << "case_id\tdocument_id\tprompt\texpected_continuation\twindow_loss\tgenerated_text\tgenerated_token_ids\tunique_generated_tokens\tmax_repeated_run\tvalid_utf8\n";
    std::cout << "case_id\tprompt\texpected_continuation\twindow_loss\tgenerated_text\tgenerated_token_ids\tunique_generated_tokens\tmax_repeated_run\tvalid_utf8\n";
    for (const EvaluationCase& item : cases) {
        const std::string& prompt = item.prompt;
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
        const std::vector<std::size_t> generated_tokens(tokens.begin() + static_cast<std::ptrdiff_t>(tokens.size() - new_tokens), tokens.end());
        std::string generated_text;
        const bool decoded = tokenizer.decode(generated_tokens, generated_text, true, &error);
        const bool valid_utf8 = decoded && attention::ByteLevelTokenizer::valid_utf8(generated_text, &error);
        const std::string escaped_case = escape(item.case_id);
        const std::string escaped_document = escape(item.document_id);
        const std::string escaped_prompt = escape(prompt);
        const std::string escaped_expected = escape(item.expected_continuation);
        const std::string escaped_generated = escape(generated_text);
        const std::string generated = token_list(generated_tokens);
        const std::size_t distinct = unique_count(generated_tokens);
        const std::size_t repeated = max_repeated_run(generated_tokens);
        report << escaped_case << '\t' << escaped_document << '\t' << escaped_prompt << '\t'
               << escaped_expected << '\t' << std::setprecision(9) << loss << '\t'
               << escaped_generated << '\t' << generated << '\t' << distinct << '\t'
               << repeated << '\t' << (valid_utf8 ? "true" : "false") << '\n';
        std::cout << escaped_case << '\t' << escaped_prompt << '\t' << escaped_expected << '\t'
                  << std::setprecision(9) << loss << '\t' << escaped_generated << '\t'
                  << generated << '\t' << distinct << '\t' << repeated << '\t'
                  << (valid_utf8 ? "true" : "false") << '\n';
    }
    return 0;
}

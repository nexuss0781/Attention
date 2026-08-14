#include "attention/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Sample {
    const char* category;
    const char* filename;
};

bool read_file(const std::filesystem::path& path, std::string& text, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open " + path.string();
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error = "cannot read " + path.string();
        return false;
    }
    text = buffer.str();
    return true;
}

std::size_t percentile(std::vector<std::size_t> values, double fraction) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const std::size_t rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(values.size())));
    return values[std::min(values.size() - 1, rank == 0 ? 0 : rank - 1)];
}

std::size_t count_code_points(std::string_view text) {
    std::size_t count = 0;
    for (const unsigned char byte : text) {
        if ((byte & 0xc0u) != 0x80u) ++count;
    }
    return count;
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path corpus = argc > 1 ? argv[1] : "data/tokenizer_measurement/corpus";
    const std::filesystem::path output = argc > 2 ? argv[2] : "";
    const std::filesystem::path distribution_output = argc > 3 ? argv[3] : "";
    const std::vector<Sample> samples{
        {"english", "english_udhr.txt"},
        {"amharic", "amharic_wikipedia_ethiopia.txt"},
        {"bilingual", "bilingual_udhr.txt"},
        {"code", "code_cpp.txt"},
        {"structured", "structured_checkpoint_test.txt"},
    };
    attention::ByteLevelTokenizer tokenizer;
    std::ostringstream report;
    std::ostringstream distribution;
    report << "category,bytes,code_points,tokens,bytes_per_token,unknown_tokens,unknown_rate,round_trip\n";
    distribution << "category,nonempty_lines,empty_lines,mean_tokens,p50_tokens,p95_tokens,max_tokens\n";
    for (const Sample sample : samples) {
        std::string text;
        std::string error;
        if (!read_file(corpus / sample.filename, text, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        std::vector<std::size_t> tokens;
        if (!tokenizer.encode(text, tokens, false, false, &error)) {
            std::cerr << sample.category << ": " << error << '\n';
            return 1;
        }
        std::string decoded;
        if (!tokenizer.decode(tokens, decoded, true, &error)) {
            std::cerr << sample.category << ": decode failed: " << error << '\n';
            return 1;
        }
        std::size_t unknown = 0;
        for (const std::size_t token : tokens) {
            if (token == attention::ByteLevelTokenizer::kUnknown) ++unknown;
        }
        const double bytes_per_token = tokens.empty() ? 0.0 :
            static_cast<double>(text.size()) / static_cast<double>(tokens.size());
        const double unknown_rate = tokens.empty() ? 0.0 :
            static_cast<double>(unknown) / static_cast<double>(tokens.size());
        report << sample.category << ',' << text.size() << ',' << count_code_points(text) << ','
               << tokens.size() << ',' << std::fixed << std::setprecision(6) << bytes_per_token << ','
               << unknown << ',' << unknown_rate << ',' << (decoded == text ? "true" : "false") << '\n';

        std::vector<std::size_t> line_lengths;
        std::size_t empty_lines = 0;
        std::size_t line_start = 0;
        while (line_start <= text.size()) {
            const std::size_t line_end = text.find('\n', line_start);
            const std::size_t end = line_end == std::string::npos ? text.size() : line_end;
            std::size_t content_end = end;
            if (content_end > line_start && text[content_end - 1] == '\r') --content_end;
            const std::string_view line(text.data() + line_start, content_end - line_start);
            if (line.empty()) {
                ++empty_lines;
            } else {
                std::vector<std::size_t> line_tokens;
                if (!tokenizer.encode(line, line_tokens, false, false, &error)) {
                    std::cerr << sample.category << ": line encoding failed: " << error << '\n';
                    return 1;
                }
                line_lengths.push_back(line_tokens.size());
            }
            if (line_end == std::string::npos) break;
            line_start = line_end + 1;
        }
        std::size_t line_total = 0;
        for (const std::size_t length : line_lengths) line_total += length;
        const double mean = line_lengths.empty() ? 0.0 :
            static_cast<double>(line_total) / static_cast<double>(line_lengths.size());
        distribution << sample.category << ',' << line_lengths.size() << ',' << empty_lines << ','
                     << std::fixed << std::setprecision(6) << mean << ','
                     << percentile(line_lengths, 0.50) << ',' << percentile(line_lengths, 0.95) << ','
                     << percentile(line_lengths, 1.00) << '\n';
        if (decoded != text || !std::isfinite(bytes_per_token) || !std::isfinite(unknown_rate)) {
            std::cerr << sample.category << ": invalid measurement result\n";
            return 1;
        }
    }
    if (!output.empty()) {
        std::ofstream file(output);
        if (!file) {
            std::cerr << "cannot write " << output.string() << '\n';
            return 1;
        }
        file << report.str();
    }
    if (!distribution_output.empty()) {
        std::ofstream file(distribution_output);
        if (!file) {
            std::cerr << "cannot write " << distribution_output.string() << '\n';
            return 1;
        }
        file << distribution.str();
    }
    std::cout << report.str() << distribution.str();
    return 0;
}

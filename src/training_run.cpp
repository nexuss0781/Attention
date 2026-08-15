#include "attention/training_run.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(character);
                }
                break;
        }
    }
    return output.str();
}

bool finite_record(const TrainingLogRecord& record) {
    return std::isfinite(record.loss_before) && std::isfinite(record.loss_after) &&
           std::isfinite(record.learning_rate) && std::isfinite(record.gradient_l2_norm) &&
           std::isfinite(record.validation_loss);
}

} // namespace

bool TrainingRunLogger::initialize(const TrainingRunMetadata& metadata, std::string* error) {
    if (metadata.run_id.empty() || metadata.stage.empty() || metadata.dataset_id.empty() ||
        metadata.tokenizer_version.empty() || metadata.architecture_serialization.empty() ||
        metadata.code_commit.empty() || metadata.batch_size == 0 || metadata.sequence_length < 2 ||
        !std::isfinite(metadata.learning_rate) || metadata.learning_rate <= 0.0f) {
        set_error(error, "training run metadata is incomplete or invalid");
        return false;
    }
    metadata_ = metadata;
    records_.clear();
    initialized_ = true;
    return true;
}

bool TrainingRunLogger::append(const TrainingLogRecord& record, std::string* error) {
    if (!initialized_) {
        set_error(error, "training run logger is not initialized");
        return false;
    }
    if (!finite_record(record)) {
        set_error(error, "training log record contains a nonfinite value");
        return false;
    }
    if (!records_.empty() && record.step <= records_.back().step) {
        set_error(error, "training log steps must increase strictly");
        return false;
    }
    records_.push_back(record);
    return true;
}

bool TrainingRunLogger::serialize(std::string& output, std::string* error) const {
    if (!initialized_) {
        set_error(error, "training run logger is not initialized");
        return false;
    }
    std::ostringstream json;
    json << std::setprecision(9);
    json << "{\n"
         << "  \"format\": \"attention.training_run.v1\",\n"
         << "  \"run_id\": \"" << json_escape(metadata_.run_id) << "\",\n"
         << "  \"stage\": \"" << json_escape(metadata_.stage) << "\",\n"
         << "  \"dataset\": {\n"
         << "    \"id\": \"" << json_escape(metadata_.dataset_id) << "\",\n"
         << "    \"revision\": \"" << json_escape(metadata_.dataset_revision) << "\",\n"
         << "    \"source_checksums\": \"" << json_escape(metadata_.source_checksums) << "\"\n"
         << "  },\n"
         << "  \"tokenizer\": {\n"
         << "    \"version\": \"" << json_escape(metadata_.tokenizer_version) << "\",\n"
         << "    \"vocabulary_size\": " << metadata_.tokenizer_vocabulary_size << "\n"
         << "  },\n"
         << "  \"architecture_serialization\": \""
         << json_escape(metadata_.architecture_serialization) << "\",\n"
         << "  \"code_commit\": \"" << json_escape(metadata_.code_commit) << "\",\n"
         << "  \"seed\": " << metadata_.seed << ",\n"
         << "  \"batch_size\": " << metadata_.batch_size << ",\n"
         << "  \"sequence_length\": " << metadata_.sequence_length << ",\n"
         << "  \"learning_rate\": " << metadata_.learning_rate << ",\n"
         << "  \"records\": [";
    for (std::size_t index = 0; index < records_.size(); ++index) {
        const TrainingLogRecord& record = records_[index];
        if (index != 0) json << ',';
        json << "\n    {\"step\": " << record.step
             << ", \"batch_index\": " << record.batch_index
             << ", \"token_offset\": " << record.token_offset
             << ", \"tokens_processed\": " << record.tokens_processed
             << ", \"loss_before\": " << record.loss_before
             << ", \"loss_after\": " << record.loss_after
             << ", \"learning_rate\": " << record.learning_rate
             << ", \"gradient_l2_norm\": " << record.gradient_l2_norm
             << ", \"validation_loss\": " << record.validation_loss << "}";
    }
    if (!records_.empty()) json << '\n';
    json << "  ]\n}\n";
    output = json.str();
    return true;
}

const TrainingRunMetadata& TrainingRunLogger::metadata() const noexcept {
    return metadata_;
}

const std::vector<TrainingLogRecord>& TrainingRunLogger::records() const noexcept {
    return records_;
}

} // namespace attention

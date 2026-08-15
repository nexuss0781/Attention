#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
BUILD_DIR="${ATTENTION_BUILD_DIR:-/tmp/attention_stage0_build}"
DRIVE_ROOT="${ATTENTION_DRIVE_ROOT:-/content/drive/MyDrive/Attention}"
CHAPTER1_ROOT="${DRIVE_ROOT}/chapter1"
CHAPTER1_DATA="${CHAPTER1_ROOT}/data"
CHAPTER1_SESSIONS="${CHAPTER1_ROOT}/sessions"
CHAPTER1_CHECKPOINTS="${CHAPTER1_ROOT}/checkpoints"
CURRICULUM="${ROOT_DIR}/data/english_competency_curriculum_v1.json"

bootstrap_colab_repository() {
    local repo_root="${1:-/content/drive/MyDrive/Attention}"
    local repo_url="${ATTENTION_REPO_URL:-https://github.com/nexuss0781/Attention.git}"
    local backup_root=""

    if [[ ! -d "/content/drive/MyDrive" && "${repo_root}" == /content/drive/* ]]; then
        echo "Google Drive is not mounted at /content/drive/MyDrive." >&2
        echo "Run the Drive mount cell in Colab, then run: bash run.sh colab" >&2
        return 1
    fi

    mkdir -p "$(dirname "${repo_root}")"
    if [[ -d "${repo_root}/.git" ]] && git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Updating existing Attention repository in Drive..."
        git -C "${repo_root}" remote set-url origin "${repo_url}" 2>/dev/null || git -C "${repo_root}" remote add origin "${repo_url}"
        git -C "${repo_root}" fetch origin master
        git -C "${repo_root}" checkout -B master origin/master
        git -C "${repo_root}" reset --hard origin/master
    else
        if [[ -e "${repo_root}" ]]; then
            backup_root="${repo_root}.non_git_backup.$(date -u +%Y%m%dT%H%M%SZ)"
            echo "Existing non-Git directory found; preserving it at ${backup_root}"
            mv "${repo_root}" "${backup_root}"
        fi
        echo "Cloning Attention into Drive..."
        git clone --branch master "${repo_url}" "${repo_root}"
        if [[ -n "${backup_root}" && -d "${backup_root}/chapter1" ]]; then
            echo "Restoring persistent Chapter 1 artifacts from the preserved directory..."
            cp -a "${backup_root}/chapter1" "${repo_root}/"
        fi
    fi
    chmod +x "${repo_root}/run.sh"
    BOOTSTRAPPED_REPO="${repo_root}"
}

run_colab_workflow() {
    local repo_root="${1:-/content/drive/MyDrive/Attention}"
    bootstrap_colab_repository "${repo_root}"
    repo_root="${BOOTSTRAPPED_REPO}"
    if [[ "${ATTENTION_COLAB_BOOTSTRAP_ONLY:-0}" == "1" ]]; then
        echo "Colab repository bootstrap complete: ${repo_root}"
        return 0
    fi
    echo "Running Attention from: ${repo_root}"
    bash "${repo_root}/run.sh" setup-drive "${repo_root}"
    bash "${repo_root}/run.sh" validate-curriculum
    bash "${repo_root}/run.sh" prepare-module1 "${repo_root}"
    bash "${repo_root}/run.sh" bootstrap "${repo_root}"
    echo "Starting Module 1.1; the command stops at AWAITING_REVIEW for manual verdict."
    bash "${repo_root}/run.sh" train-module 1.1 module_1_1_session_001 "${repo_root}"
}

set_drive_root() {
    DRIVE_ROOT="$1"
    CHAPTER1_ROOT="${DRIVE_ROOT}/chapter1"
    CHAPTER1_DATA="${CHAPTER1_ROOT}/data"
    CHAPTER1_SESSIONS="${CHAPTER1_ROOT}/sessions"
    CHAPTER1_CHECKPOINTS="${CHAPTER1_ROOT}/checkpoints"
}

usage() {
    cat <<'EOF'
Usage:
  ./run.sh colab [DRIVE_REPO]
  ./run.sh setup-drive DRIVE_ROOT
  ./run.sh prepare-module1 DRIVE_ROOT
  ./run.sh bootstrap DRIVE_ROOT
  ./run.sh train-module MODULE_ID SESSION_ID DRIVE_ROOT
  ./run.sh evaluate-module MODULE_ID SESSION_ID DRIVE_ROOT
  ./run.sh decide SESSION_ID PROMOTE|RETRY|ABORT "RATIONALE" DRIVE_ROOT
  ./run.sh status SESSION_ID DRIVE_ROOT
  ./run.sh validate-curriculum
  ./run.sh chapter1-prepare
  ./run.sh chapter1 MODULE
  ./run.sh chapter1-status MODULE
  ./run.sh chapter1-decision MODULE PROMOTE|RETRY|ABORT "RATIONALE"
  bash run.sh retry-module MODULE START_DOCUMENT DRIVE_ROOT
  bash run.sh chapter1-retry MODULE START_DOCUMENT [DRIVE_ROOT]

MODULE values:
  1.1  symbol_and_byte_regularity
  1.2  word_boundary_continuation
  1.3  phrase_continuation
  1.4  sentence_continuation

The chapter1 command performs, in order:
  prepare data -> load parent -> baseline test -> train -> validate
  -> competency test -> seal artifacts -> stop at AWAITING_REVIEW

No module is promoted automatically. A failed module cannot advance.
All Chapter 1 data, checkpoints, reports, and traces persist below:
  /content/drive/MyDrive/Attention/chapter1/
EOF
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "required file does not exist: $1" >&2
        exit 1
    fi
}

build_all() {
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DATTENTION_ENABLE_NATIVE=OFF
    cmake --build "${BUILD_DIR}" -j2
}

module_config() {
    local module="$1"
    case "${module}" in
        1.1)
            MODULE_NAME="symbol_and_byte_regularity"
            MODULE_COMPETENCY="module_1_1_symbol_byte_regularity"
            MODULE_DIR="${CHAPTER1_DATA}/module_1_1"
            MODULE_STEPS=2048
            MODULE_TOKENS=64000
            MODULE_MAX_VALIDATION=8000
            MODULE_DOC_START=0
            MODULE_TRAIN_DOCS=64
            MODULE_VALIDATION_DOCS=16
            MODULE_PROMPT_CHARS=16
            MODULE_CONTINUATION_CHARS=16
            MODULE_FINETUNE_STEPS=256
            MODULE_FINETUNE_TOKENS=6400
            ;;
        1.2)
            MODULE_NAME="word_boundary_continuation"
            MODULE_COMPETENCY="module_1_2_word_boundary_continuation"
            MODULE_DIR="${CHAPTER1_DATA}/module_1_2"
            MODULE_STEPS=3072
            MODULE_TOKENS=256000
            MODULE_MAX_VALIDATION=32000
            MODULE_DOC_START=80
            MODULE_TRAIN_DOCS=256
            MODULE_VALIDATION_DOCS=32
            MODULE_PROMPT_CHARS=32
            MODULE_CONTINUATION_CHARS=24
            MODULE_FINETUNE_STEPS=384
            MODULE_FINETUNE_TOKENS=25600
            ;;
        1.3)
            MODULE_NAME="phrase_continuation"
            MODULE_COMPETENCY="module_1_3_phrase_continuation"
            MODULE_DIR="${CHAPTER1_DATA}/module_1_3"
            MODULE_STEPS=4096
            MODULE_TOKENS=512000
            MODULE_MAX_VALIDATION=64000
            MODULE_DOC_START=368
            MODULE_TRAIN_DOCS=512
            MODULE_VALIDATION_DOCS=64
            MODULE_PROMPT_CHARS=48
            MODULE_CONTINUATION_CHARS=32
            MODULE_FINETUNE_STEPS=512
            MODULE_FINETUNE_TOKENS=51200
            ;;
        1.4)
            MODULE_NAME="sentence_continuation"
            MODULE_COMPETENCY="module_1_4_sentence_continuation"
            MODULE_DIR="${CHAPTER1_DATA}/module_1_4"
            MODULE_STEPS=6144
            MODULE_TOKENS=1024000
            MODULE_MAX_VALIDATION=128000
            MODULE_DOC_START=944
            MODULE_TRAIN_DOCS=1024
            MODULE_VALIDATION_DOCS=128
            MODULE_PROMPT_CHARS=64
            MODULE_CONTINUATION_CHARS=48
            MODULE_FINETUNE_STEPS=768
            MODULE_FINETUNE_TOKENS=102400
            ;;
        *)
            echo "unknown Chapter 1 module: ${module}" >&2
            usage
            exit 2
            ;;
    esac
    if [[ -n "${MODULE_DOC_START_OVERRIDE:-}" ]]; then MODULE_DOC_START="${MODULE_DOC_START_OVERRIDE}"; fi
    if [[ -n "${CHAPTER1_DATA_OVERRIDE:-}" ]]; then CHAPTER1_DATA="${CHAPTER1_DATA_OVERRIDE}"; fi
    if [[ -n "${MODULE_DIR_OVERRIDE:-}" ]]; then MODULE_DIR="${MODULE_DIR_OVERRIDE}"; fi
    if [[ -n "${MODULE_STEPS_OVERRIDE:-}" ]]; then MODULE_STEPS="${MODULE_STEPS_OVERRIDE}"; fi
    if [[ -n "${MODULE_FINETUNE_STEPS_OVERRIDE:-}" ]]; then MODULE_FINETUNE_STEPS="${MODULE_FINETUNE_STEPS_OVERRIDE}"; fi
    MODULE_VALIDATION_INTERVAL="${MODULE_VALIDATION_INTERVAL_OVERRIDE:-128}"
}

module_session_id() {
    if [[ -n "${SESSION_ID_OVERRIDE:-}" ]]; then
        printf '%s\n' "${SESSION_ID_OVERRIDE}"
    else
        printf 'chapter1_module_%s_session_001\n' "${1//./_}"
    fi
}

module_session_id_fixed() {
    printf 'chapter1_module_%s_session_001\n' "${1//./_}"
}

session_dir_fixed() {
    printf '%s/%s\n' "${CHAPTER1_SESSIONS}" "$(module_session_id_fixed "$1")"
}

session_dir_for() {
    printf '%s/%s\n' "${CHAPTER1_SESSIONS}" "$(module_session_id "$1")"
}

current_parent() {
    local pointer="${CHAPTER1_CHECKPOINTS}/current_checkpoint.json"
    if [[ -f "${pointer}" ]]; then
        "${PYTHON}" - "${pointer}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get("checkpoint")
if not isinstance(value, str) or not value:
    raise SystemExit("current checkpoint pointer is invalid")
print(value)
PY
    else
        printf '%s/initial.model.checkpoint\n' "${CHAPTER1_ROOT}"
    fi
}

prepare_module_files() {
    local module="$1"
    module_config "${module}"
    mkdir -p "${MODULE_DIR}"
    require_file "${CHAPTER1_DATA}/selected_documents.jsonl"
    if [[ -f "${MODULE_DIR}/lesson_manifest.json" && -f "${MODULE_DIR}/cases.tsv" ]]; then
        if [[ ! -f "${MODULE_DIR}/finetune.tokens" ]]; then
            "${PYTHON}" - "${MODULE_DIR}/train.tokens" "${MODULE_DIR}/finetune.tokens" "${MODULE_FINETUNE_TOKENS}" <<'PY'
from pathlib import Path
import sys
values = Path(sys.argv[1]).read_text(encoding="utf-8").split()
limit = int(sys.argv[3])
if len(values) < limit:
    raise SystemExit(f"training token stream has {len(values)} tokens; fine-tune requires {limit}")
Path(sys.argv[2]).write_text(" ".join(values[-limit:]) + "\n", encoding="utf-8")
PY
        fi
        return 0
    fi
    "${PYTHON}" "${ROOT_DIR}/scripts/prepare_module_1_1.py" \
        --module-id "${module}" \
        --module-name "${MODULE_NAME}" \
        --input-jsonl "${CHAPTER1_DATA}/selected_documents.jsonl" \
        --output-train-tokens "${MODULE_DIR}/train.tokens" \
        --output-validation-tokens "${MODULE_DIR}/validation.tokens" \
        --output-cases "${MODULE_DIR}/cases.tsv" \
        --output-manifest "${MODULE_DIR}/lesson_manifest.json" \
        --dataset-id "fineweb_english" \
        --dataset-revision "$("${PYTHON}" - "${CHAPTER1_DATA}/download_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["resolved_revision"])
PY
)" \
        --train-start "${MODULE_DOC_START}" --train-count "${MODULE_TRAIN_DOCS}" \
        --validation-start "$((MODULE_DOC_START + MODULE_TRAIN_DOCS))" --validation-count "${MODULE_VALIDATION_DOCS}" \
        --max-train-tokens "${MODULE_TOKENS}" \
        --max-validation-tokens "${MODULE_MAX_VALIDATION}" \
        --case-count 16 \
        --prompt-chars "${MODULE_PROMPT_CHARS}" \
        --continuation-chars "${MODULE_CONTINUATION_CHARS}"
    "${PYTHON}" - "${MODULE_DIR}/train.tokens" "${MODULE_DIR}/finetune.tokens" "${MODULE_FINETUNE_TOKENS}" <<'PY'
from pathlib import Path
import sys
values = Path(sys.argv[1]).read_text(encoding="utf-8").split()
limit = int(sys.argv[3])
if len(values) < limit:
    raise SystemExit(f"training token stream has {len(values)} tokens; fine-tune requires {limit}")
Path(sys.argv[2]).write_text(" ".join(values[-limit:]) + "\n", encoding="utf-8")
PY
}

chapter1_prepare() {
    if [[ "${DRIVE_ROOT}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
        echo "Google Drive is not mounted at /content/drive/MyDrive." >&2
        echo "Mount Drive in Colab, then rerun run.sh." >&2
        exit 1
    fi
    mkdir -p "${CHAPTER1_DATA}" "${CHAPTER1_SESSIONS}" "${CHAPTER1_CHECKPOINTS}"
    cd "${ROOT_DIR}"
    "${PYTHON}" -m pip install -q datasets huggingface_hub
    if [[ ! -f "${CHAPTER1_DATA}/download_manifest.json" || ! -f "${CHAPTER1_DATA}/selected_documents.jsonl" ]]; then
        "${PYTHON}" scripts/prepare_colab_english_stage1.py \
            --output-dir "${CHAPTER1_DATA}" \
            --start-document 0 \
            --train-documents 1880 \
            --validation-documents 216 \
            --max-train-tokens 3840000 \
            --max-validation-tokens 512000
    fi
    if [[ ! -f "${CHAPTER1_DATA}/chunk_manifest.json" ]]; then
        "${PYTHON}" scripts/prepare_english_chunk.py \
            --input-jsonl "${CHAPTER1_DATA}/selected_documents.jsonl" \
            --output-train-tokens "${CHAPTER1_DATA}/train.tokens" \
            --output-validation-tokens "${CHAPTER1_DATA}/validation.tokens" \
            --output-manifest "${CHAPTER1_DATA}/chunk_manifest.json" \
            --dataset-id "fineweb_english" \
            --dataset-revision "$(${PYTHON} - "${CHAPTER1_DATA}/download_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["resolved_revision"])
PY
)" \
            --train-start 0 --train-count 1880 \
            --validation-start 1880 --validation-count 216 \
            --language-mode dedicated_english \
            --max-train-tokens 3840000 \
            --max-validation-tokens 512000
    fi
    local prepare_module
    for prepare_module in 1.1 1.2 1.3 1.4; do
        prepare_module_files "${prepare_module}"
    done
    cat > "${CHAPTER1_ROOT}/chapter1_plan.json" <<'EOF'
{
  "format": "attention.chapter1_learning_plan.v1",
  "chapter": "1",
  "name": "English Sequence Learning",
  "modules": {
    "1.1": {"goal": "symbol_and_byte_regularity", "train_tokens": 64000, "validation_tokens": 8000, "max_steps": 2048},
    "1.2": {"goal": "word_boundary_continuation", "train_tokens": 256000, "validation_tokens": 32000, "max_steps": 3072},
    "1.3": {"goal": "phrase_continuation", "train_tokens": 512000, "validation_tokens": 64000, "max_steps": 4096},
    "1.4": {"goal": "sentence_continuation", "train_tokens": 1024000, "validation_tokens": 128000, "max_steps": 6144}
  },
  "promotion_rule": "module_n_plus_1_requires_explicit_PROMOTE_for_module_n",
  "failure_rule": "failure_stops_progression_and_requires_new_chunk_retry",
  "persistent_root": "/content/drive/MyDrive/Attention/chapter1"
}
EOF
    if [[ ! -f "${CHAPTER1_ROOT}/initial.model.checkpoint" ]]; then
        build_all
        "${BUILD_DIR}/attention_initialize_checkpoint" "${CHAPTER1_ROOT}/initial.model.checkpoint"
    fi
    echo "Chapter 1 preparation complete: ${CHAPTER1_ROOT}"
}

previous_module_promoted() {
    local module="$1"
    case "${module}" in
        1.1) return 0 ;;
        1.2) [[ "$(cat "$(session_dir_fixed 1.1)/STATE" 2>/dev/null || true)" == "PROMOTED" ]] ;;
        1.3) [[ "$(cat "$(session_dir_fixed 1.2)/STATE" 2>/dev/null || true)" == "PROMOTED" ]] ;;
        1.4) [[ "$(cat "$(session_dir_fixed 1.3)/STATE" 2>/dev/null || true)" == "PROMOTED" ]] ;;
    esac
}

write_competency_report() {
    local session_dir="$1"
    local module="$2"
    local evaluation="$3"
    "${PYTHON}" - "${session_dir}" "${module}" "${evaluation}" <<'PY'
import csv, json, math, sys
from pathlib import Path
session = Path(sys.argv[1])
module = sys.argv[2]
evaluation = Path(sys.argv[3])
lines = evaluation.read_text(encoding="utf-8").splitlines()
header = "case_id\tdocument_id\tprompt\texpected_continuation\twindow_loss\tgenerated_text\tgenerated_token_ids\tunique_generated_tokens\tmax_repeated_run\tvalid_utf8"
rows = list(csv.DictReader(lines[lines.index(header):], delimiter="\t"))
valid = sum(row["valid_utf8"] == "true" for row in rows)
diverse = sum(int(row["unique_generated_tokens"]) > 1 for row in rows)
not_collapsed = sum(int(row["max_repeated_run"]) < 16 for row in rows)
loss_finite = all(math.isfinite(float(row["window_loss"])) for row in rows)
threshold = math.ceil(len(rows) * 0.80)
case_passes = sum(row["valid_utf8"] == "true" and int(row["unique_generated_tokens"]) > 1 and int(row["max_repeated_run"]) < 16 for row in rows)
passed = len(rows) > 0 and case_passes >= threshold and loss_finite
validation = json.loads((session / "validation_report.json").read_text(encoding="utf-8"))
report = {
    "format": "attention.module_competency_report.v1",
    "module": module,
    "competency": "English sequence learning",
    "lesson": "symbol, byte, word-boundary, phrase, or sentence continuation",
    "cases": len(rows),
    "valid_utf8_cases": valid,
    "diverse_cases": diverse,
    "not_collapsed_cases": not_collapsed,
    "case_passes": case_passes,
    "required_cases": threshold,
    "window_loss_finite": loss_finite,
    "training_loss_before": validation["training_loss_before"],
    "training_loss_after": validation["training_loss_after"],
    "validation_loss_first": validation["validation_loss_first"],
    "validation_loss_final": validation["validation_loss_final"],
    "model_reload_loss": validation["model_reload_loss"],
    "training_state_reload_loss": validation["training_state_reload_loss"],
    "automated_competency_passed": passed,
    "manual_review_required": True,
    "decision": "AWAITING_REVIEW",
    "interpretation": "This report evaluates the selected lesson only; it does not claim general conversational ability.",
}
(session / "competency_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with (session / "competency_results.txt").open("w", encoding="utf-8") as out:
    out.write(f"Module {module} Competency Test Results\n")
    out.write("=" * 72 + "\n")
    out.write("CASE | PROMPT | GENERATED | UNIQUE | MAX_RUN | UTF8 | PASS\n")
    out.write("-" * 72 + "\n")
    for row in rows:
        prompt = row["prompt"].replace("\\t", " ")[:40]
        generated = row["generated_text"].replace("\\t", " ")[:40]
        case_pass = row["valid_utf8"] == "true" and int(row["unique_generated_tokens"]) > 1 and int(row["max_repeated_run"]) < 16
        out.write(f"{row['case_id']} | {prompt} | {generated} | {row['unique_generated_tokens']} | {row['max_repeated_run']} | {row['valid_utf8']} | {'PASS' if case_pass else 'FAIL'}\n")
    out.write("\n")
    out.write(f"SUMMARY: {case_passes}/{len(rows)} cases passed (diverse, non-collapsed, valid UTF-8)\n")
    out.write(f"AUTOMATED LESSON VERDICT: {'PASS' if passed else 'FAIL'}\n")
    out.write("MANUAL VERDICT REQUIRED: PROMOTE | RETRY | ABORT\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
}

chapter1_run() {
    local module="$1"
    module_config "${module}"
    if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then chapter1_prepare; fi
    module_config "${module}"
    if ! previous_module_promoted "${module}"; then
        echo "${module} is blocked: the preceding module has not been PROMOTED." >&2
        exit 1
    fi
    local session_id
    session_id="$(module_session_id "${module}")"
    local session_dir
    session_dir="$(session_dir_for "${module}")"
    if [[ -f "${session_dir}/STATE" ]]; then
        local state
        state="$(tr -d '\n' < "${session_dir}/STATE")"
        if [[ "${state}" == "AWAITING_REVIEW" ]]; then
            echo "${module} is already awaiting review: ${session_dir}"
            exit 0
        fi
        if [[ "${state}" == "PROMOTED" ]]; then
            echo "${module} is already promoted; do not rerun the immutable session." >&2
            exit 1
        fi
        if [[ "${state}" != "PLANNED" ]]; then
            echo "${module} has state ${state}; create a new retry session after a manual RETRY decision." >&2
            exit 1
        fi
    else
        build_all
        local parent
        parent="$(current_parent)"
        require_file "${parent}"
        cd "${ROOT_DIR}"
        "${PYTHON}" scripts/session_artifacts.py init \
            --root "${CHAPTER1_SESSIONS}" \
            --session-id "${session_id}" \
            --stage-id "chapter1_english_sequence_learning" \
            --competency-id "${MODULE_COMPETENCY}" \
            --parent-checkpoint "${parent}" \
            --dataset-manifest "${MODULE_DIR}/lesson_manifest.json" \
            --dataset-chunk "${CHAPTER1_DATA}/chunk_manifest.json" \
            --seed 17 \
            --max-steps "${MODULE_STEPS}" \
            --max-tokens "${MODULE_TOKENS}"
    fi

    build_all
    local baseline="${session_dir}/baseline_evaluation.tsv"
    local evaluation="${session_dir}/competency_evaluation.tsv"
    local parent
    parent="$("${PYTHON}" - "${session_dir}/session_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["parent_checkpoint"])
PY
)"
    if [[ ! -f "${baseline}" ]]; then
        "${BUILD_DIR}/attention_stage0_evaluation" "${parent}" "${baseline}" 16 "${MODULE_DIR}/cases.tsv"
    fi
    cd "${ROOT_DIR}"
    ATTENTION_CODE_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || printf unknown)" \
        ATTENTION_LEARNING_RATE="${ATTENTION_LEARNING_RATE:-0.005}" \
        ATTENTION_VALIDATION_INTERVAL="${MODULE_VALIDATION_INTERVAL}" \
        "${PYTHON}" scripts/run_session.py \
            --session-dir "${session_dir}" \
            --executable "${BUILD_DIR}/attention_stage0_training" \
            --train-tokens "${MODULE_DIR}/train.tokens" \
            --validation-tokens "${MODULE_DIR}/validation.tokens"
    ATTENTION_SESSION_ID="${session_id}_finetune" \
        ATTENTION_STAGE_ID="chapter1_english_sequence_learning_finetune" \
        ATTENTION_DATASET_ID="fineweb_english" \
        ATTENTION_CODE_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || printf unknown)" \
        ATTENTION_LEARNING_RATE="${ATTENTION_FINETUNE_LEARNING_RATE:-${ATTENTION_LEARNING_RATE:-0.005}}" \
        ATTENTION_VALIDATION_INTERVAL="${MODULE_VALIDATION_INTERVAL}" \
        ATTENTION_MAX_STEPS="${MODULE_FINETUNE_STEPS}" \
        "${BUILD_DIR}/attention_stage0_training" \
            "${MODULE_DIR}/finetune.tokens" \
            "${session_dir}/finetuning_log.json" \
            "${session_dir}/finetuning_state.checkpoint" \
            "${MODULE_DIR}/validation.tokens" \
            "${session_dir}/finetuned.model.checkpoint" \
            "${session_dir}/model.checkpoint" \
            > "${session_dir}/finetuning_trace.csv" 2>&1
    mv -f "${session_dir}/finetuned.model.checkpoint" "${session_dir}/model.checkpoint"
    mv -f "${session_dir}/finetuning_state.checkpoint" "${session_dir}/training_state.checkpoint"
    "${PYTHON}" - "${session_dir}" <<'PY'
import json, math, sys
from pathlib import Path
session = Path(sys.argv[1])
manifest = json.loads((session / "session_manifest.json").read_text(encoding="utf-8"))
run = json.loads((session / "finetuning_log.json").read_text(encoding="utf-8"))
records = run.get("records", [])
if not records:
    raise SystemExit("fine-tuning log contains no records")
values = {}
for line in (session / "finetuning_trace.csv").read_text(encoding="utf-8").splitlines():
    if "," in line:
        key, value = line.split(",", 1)
        if key in {"reloaded_loss", "resumed_loss"}:
            values[key] = float(value)
if not all(math.isfinite(values.get(k, float("nan"))) for k in ("reloaded_loss", "resumed_loss")):
    raise SystemExit("fine-tuning checkpoint reload metrics are missing or nonfinite")
report = {
    "format": "attention.competency_validation_report.v1",
    "session_id": manifest["session_id"],
    "competency_id": manifest["competency_id"],
    "automated_pass": True,
    "training_loss_before": float(records[0]["loss_before"]),
    "training_loss_after": float(records[-1]["loss_after"]),
    "validation_loss_first": float(records[0]["validation_loss"]),
    "validation_loss_final": float(records[-1]["validation_loss"]),
    "validation_loss_finite": all(math.isfinite(float(r["validation_loss"])) for r in records),
    "gradient_norms_finite": all(math.isfinite(float(r["gradient_l2_norm"])) for r in records),
    "model_reload_loss": values["reloaded_loss"],
    "training_state_reload_loss": values["resumed_loss"],
    "manual_review_required": True,
    "training_phase": "pretrain_then_finetune",
}
(session / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
    "${BUILD_DIR}/attention_stage0_evaluation" "${session_dir}/model.checkpoint" "${evaluation}" 16 "${MODULE_DIR}/cases.tsv"
    write_competency_report "${session_dir}" "${module}" "${evaluation}"
    "${PYTHON}" scripts/session_artifacts.py seal --session-dir "${session_dir}"
    echo
    cat "${session_dir}/competency_results.txt"
    echo "STOP: ${module} is sealed at AWAITING_REVIEW."
    echo "Paste these files for the joint verdict:"
    echo "  ${session_dir}/competency_report.json"
    echo "  ${session_dir}/baseline_evaluation.tsv"
    echo "  ${session_dir}/competency_evaluation.tsv"
    echo "  ${session_dir}/validation_report.json"
}

chapter1_status() {
    local module="$1"
    module_config "${module}"
    local directory
    directory="$(session_dir_for "${module}")"
    require_file "${directory}/session_manifest.json"
    cat "${directory}/session_manifest.json"
    printf '\nSTATE: '
    cat "${directory}/STATE"
    if [[ -f "${directory}/competency_report.json" ]]; then
        cat "${directory}/competency_report.json"
    fi
}

setup_drive() {
    local root="$1"
    set_drive_root "${root}"
    if [[ "${root}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
        echo "Google Drive is not mounted at /content/drive/MyDrive." >&2
        echo "Mount Drive in Colab, then rerun this command." >&2
        exit 1
    fi
    mkdir -p "${CHAPTER1_DATA}" "${CHAPTER1_SESSIONS}" "${CHAPTER1_CHECKPOINTS}"
    printf 'Drive-persistent root ready: %s\\n' "${DRIVE_ROOT}"
}

bootstrap_drive() {
    local root="$1"
    set_drive_root "${root}"
    if [[ "${DRIVE_ROOT}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
        echo "Google Drive is not mounted at /content/drive/MyDrive." >&2
        echo "Mount Drive in Colab, then rerun run.sh." >&2
        exit 1
    fi
    mkdir -p "${CHAPTER1_DATA}" "${CHAPTER1_SESSIONS}" "${CHAPTER1_CHECKPOINTS}"
    build_all
    if [[ ! -f "${CHAPTER1_ROOT}/initial.model.checkpoint" ]]; then
        "${BUILD_DIR}/attention_initialize_checkpoint" "${CHAPTER1_ROOT}/initial.model.checkpoint"
    fi
    printf 'Bootstrap checkpoint: %s\\n' "${CHAPTER1_ROOT}/initial.model.checkpoint"
}

train_module_alias() {
    local module="$1" session="$2" root="$3"
    set_drive_root "${root}"
    SESSION_ID_OVERRIDE="${session}" chapter1_run "${module}"
}

evaluate_module_alias() {
    local module="$1" session="$2" root="$3"
    set_drive_root "${root}"
    module_config "${module}"
    local directory="${CHAPTER1_SESSIONS}/${session}"
    require_file "${directory}/competency_results.txt"
    cat "${directory}/competency_results.txt"
    printf '\\nPersistent evaluation directory: %s\\n' "${directory}"
}

status_session_alias() {
    local session="$1" root="$2"
    set_drive_root "${root}"
    local directory="${CHAPTER1_SESSIONS}/${session}"
    require_file "${directory}/session_manifest.json"
    cat "${directory}/session_manifest.json"
    printf '\\nSTATE: '
    cat "${directory}/STATE"
    if [[ -f "${directory}/competency_results.txt" ]]; then cat "${directory}/competency_results.txt"; fi
}

decide_session_alias() {
    local session="$1" decision="$2" rationale="$3" root="$4"
    case "${decision}" in PROMOTE|RETRY|ABORT) ;; *) echo "decision must be PROMOTE, RETRY, or ABORT" >&2; exit 2 ;; esac
    set_drive_root "${root}"
    local directory="${CHAPTER1_SESSIONS}/${session}" decision_file
    require_file "${directory}/session_manifest.json"
    decision_file="${directory}/manual_decision.input.txt"
    local competency
    competency="$("${PYTHON}" - "${directory}/session_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["competency_id"])
PY
)"
    cat > "${decision_file}" <<EOF
DECISION: ${decision}
COMPETENCY: ${competency}
RATIONALE: ${rationale}
NEXT_ACTION: ${decision}
EOF
    cd "${ROOT_DIR}"
    "${PYTHON}" scripts/session_artifacts.py decision --session-dir "${directory}" --decision-file "${decision_file}"
    if [[ "${decision}" == "PROMOTE" ]]; then
        local promoted="${CHAPTER1_CHECKPOINTS}/${session}.model.checkpoint"
        cp --reflink=auto "${directory}/model.checkpoint" "${promoted}"
        local digest="$(sha256sum "${promoted}" | cut -d ' ' -f1)"
        cat > "${CHAPTER1_CHECKPOINTS}/current_checkpoint.json.tmp" <<EOF
{
  "format": "attention.promoted_checkpoint_pointer.v1",
  "session_id": "${session}",
  "checkpoint": "${promoted}",
  "sha256": "${digest}"
}
EOF
        mv -f "${CHAPTER1_CHECKPOINTS}/current_checkpoint.json.tmp" "${CHAPTER1_CHECKPOINTS}/current_checkpoint.json"
        echo "PROMOTED: ${promoted}"
    fi
}

chapter1_retry() {
    local module="$1" retry_start="$2" root="${3:-${DRIVE_ROOT}}"
    [[ "${retry_start}" =~ ^[0-9]+$ ]] || { echo "START_DOCUMENT must be a nonnegative integer" >&2; exit 2; }
    set_drive_root "${root}"
    module_config "${module}"
    local retry_data="${CHAPTER1_ROOT}/data/retry_${module//./_}_${retry_start}"
    mkdir -p "${retry_data}"
    cd "${ROOT_DIR}"
    "${PYTHON}" -m pip install -q datasets huggingface_hub
    if [[ ! -f "${retry_data}/download_manifest.json" ]]; then
        "${PYTHON}" scripts/prepare_colab_english_stage1.py \
            --output-dir "${retry_data}" \
            --start-document "${retry_start}" \
            --train-documents "${MODULE_TRAIN_DOCS}" \
            --validation-documents "${MODULE_VALIDATION_DOCS}" \
            --max-train-tokens "${MODULE_TOKENS}" \
            --max-validation-tokens "${MODULE_MAX_VALIDATION}"
    fi
    local retry_session="chapter1_module_${module//./_}_retry_${retry_start}"
    CHAPTER1_DATA_OVERRIDE="${retry_data}" \
        MODULE_DIR_OVERRIDE="${retry_data}/module_${module//./_}" \
        MODULE_DOC_START_OVERRIDE=0 \
        SESSION_ID_OVERRIDE="${retry_session}" \
        SKIP_PREPARE=1 chapter1_run "${module}"
}

retry_module() {
    local module="$1" retry_start="$2" root="$3"
    set_drive_root "${root}"
    module_config "${module}"
    local session_dir="$(session_dir_fixed "${module}")"
    require_file "${session_dir}/session_manifest.json"
    local state="$(cat "${session_dir}/STATE")"
    if [[ "${state}" == "AWAITING_REVIEW" ]]; then
        decide_session_alias "$(module_session_id_fixed "${module}")" RETRY "Competency test failed; automatically retrying with a new FineWeb document chunk." "${root}"
    elif [[ "${state}" != "RETRY" ]]; then
        echo "Cannot retry ${module}: session state is ${state}; expected AWAITING_REVIEW or RETRY." >&2
        exit 1
    fi
    chapter1_retry "${module}" "${retry_start}" "${root}"
}

chapter1_decision() {
    local module="$1"
    local decision="$2"
    local rationale="$3"
    module_config "${module}"
    case "${decision}" in PROMOTE|RETRY|ABORT) ;; *) echo "decision must be PROMOTE, RETRY, or ABORT" >&2; exit 2 ;; esac
    local session_id session_dir decision_file
    session_id="$(module_session_id "${module}")"
    session_dir="$(session_dir_for "${module}")"
    require_file "${session_dir}/session_manifest.json"
    decision_file="${session_dir}/manual_decision.input.txt"
    cat > "${decision_file}" <<EOF
DECISION: ${decision}
COMPETENCY: ${MODULE_COMPETENCY}
RATIONALE: ${rationale}
NEXT_ACTION: ${decision}
EOF
    cd "${ROOT_DIR}"
    "${PYTHON}" scripts/session_artifacts.py decision --session-dir "${session_dir}" --decision-file "${decision_file}"
    if [[ "${decision}" == "PROMOTE" ]]; then
        local promoted="${CHAPTER1_CHECKPOINTS}/${session_id}.model.checkpoint"
        mkdir -p "${CHAPTER1_CHECKPOINTS}"
        cp --reflink=auto "${session_dir}/model.checkpoint" "${promoted}"
        local digest
        digest="$(sha256sum "${promoted}" | cut -d ' ' -f1)"
        cat > "${CHAPTER1_CHECKPOINTS}/current_checkpoint.json.tmp" <<EOF
{
  "format": "attention.promoted_checkpoint_pointer.v1",
  "session_id": "${session_id}",
  "checkpoint": "${promoted}",
  "sha256": "${digest}"
}
EOF
        mv -f "${CHAPTER1_CHECKPOINTS}/current_checkpoint.json.tmp" "${CHAPTER1_CHECKPOINTS}/current_checkpoint.json"
        echo "PROMOTED: ${promoted}"
    fi
}

if [[ $# -eq 0 ]]; then
    if [[ -d "/content/drive/MyDrive" ]]; then
        run_colab_workflow "/content/drive/MyDrive/Attention"
        exit $?
    fi
    usage
    exit 2
fi
command="$1"
shift
case "${command}" in
    colab)
        [[ $# -eq 0 || $# -eq 1 ]] || { usage; exit 2; }
        run_colab_workflow "${1:-/content/drive/MyDrive/Attention}"
        ;;
    setup-drive)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        setup_drive "$1"
        ;;
    prepare-module1)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        set_drive_root "$1"
        chapter1_prepare
        ;;
    bootstrap)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        bootstrap_drive "$1"
        ;;
    train-module)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        train_module_alias "$1" "$2" "$3"
        ;;
    evaluate-module)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        evaluate_module_alias "$1" "$2" "$3"
        ;;
    decide)
        [[ $# -eq 4 ]] || { usage; exit 2; }
        decide_session_alias "$1" "$2" "$3" "$4"
        ;;
    status)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        status_session_alias "$1" "$2"
        ;;
    validate-curriculum)
        [[ $# -eq 0 ]] || { usage; exit 2; }
        cd "${ROOT_DIR}"
        "${PYTHON}" scripts/validate_english_curriculum.py "${CURRICULUM}"
        ;;
    chapter1-prepare)
        [[ $# -eq 0 ]] || { usage; exit 2; }
        chapter1_prepare
        ;;
    chapter1)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        chapter1_run "$1"
        ;;
    chapter1-status)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        chapter1_status "$1"
        ;;
    chapter1-decision)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        chapter1_decision "$1" "$2" "$3"
        ;;
    retry-module)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        retry_module "$1" "$2" "$3"
        ;;
    chapter1-retry)
        [[ $# -eq 2 || $# -eq 3 ]] || { usage; exit 2; }
        chapter1_retry "$1" "$2" "${3:-${DRIVE_ROOT}}"
        ;;
    *)
        usage
        exit 2
        ;;
esac

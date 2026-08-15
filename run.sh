#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SESSION_ROOT="${ATTENTION_SESSION_ROOT:-${ROOT_DIR}/artifacts/sessions}"
CHECKPOINT_ROOT="${ATTENTION_CHECKPOINT_ROOT:-${ROOT_DIR}/artifacts/checkpoints}"
BUILD_DIR="${ATTENTION_BUILD_DIR:-/tmp/attention_stage0_build}"
CURRICULUM="${ROOT_DIR}/data/english_competency_curriculum_v1.json"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh validate-curriculum
  ./run.sh init SESSION_ID STAGE_ID COMPETENCY_ID PARENT_CHECKPOINT DATASET_MANIFEST DATASET_CHUNK SEED MAX_STEPS MAX_TOKENS
  ./run.sh train SESSION_ID TRAIN_TOKENS VALIDATION_TOKENS
  ./run.sh decide SESSION_ID DECISION_FILE
  ./run.sh status SESSION_ID

The train command always stops after automated validation and manual-review sealing.
A checkpoint is promoted only when decide is called with a valid PROMOTE decision file.
EOF
}

session_dir() {
    printf '%s/%s\n' "${SESSION_ROOT}" "$1"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "required file does not exist: $1" >&2
        exit 1
    fi
}

ensure_build() {
    if [[ ! -x "${BUILD_DIR}/attention_stage0_training" ]]; then
        cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
        cmake --build "${BUILD_DIR}" -j2
    fi
}

promote_if_requested() {
    local session_id="$1"
    local directory
    directory="$(session_dir "${session_id}")"
    local state
    state="$(tr -d '\n' < "${directory}/STATE")"
    if [[ "${state}" != "PROMOTED" ]]; then
        return 0
    fi
    require_file "${directory}/model.checkpoint"
    mkdir -p "${CHECKPOINT_ROOT}"
    local promoted="${CHECKPOINT_ROOT}/${session_id}.model.checkpoint"
    if [[ -e "${promoted}" ]]; then
        echo "refusing to overwrite existing promoted checkpoint: ${promoted}" >&2
        exit 1
    fi
    cp --reflink=auto "${directory}/model.checkpoint" "${promoted}"
    local digest
    digest="$(sha256sum "${promoted}" | cut -d ' ' -f1)"
    local pointer_tmp="${CHECKPOINT_ROOT}/current_checkpoint.json.tmp"
    cat > "${pointer_tmp}" <<EOF
{
  "format": "attention.promoted_checkpoint_pointer.v1",
  "session_id": "${session_id}",
  "checkpoint": "${promoted}",
  "sha256": "${digest}"
}
EOF
    mv -f "${pointer_tmp}" "${CHECKPOINT_ROOT}/current_checkpoint.json"
    echo "promoted checkpoint: ${promoted}"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

command="$1"
shift
case "${command}" in
    validate-curriculum)
        [[ $# -eq 0 ]] || { usage; exit 2; }
        cd "${ROOT_DIR}"
        "${PYTHON}" scripts/validate_english_curriculum.py "${CURRICULUM}"
        ;;
    init)
        [[ $# -eq 9 ]] || { usage; exit 2; }
        cd "${ROOT_DIR}"
        "${PYTHON}" scripts/session_artifacts.py init \
            --root "${SESSION_ROOT}" \
            --session-id "$1" \
            --stage-id "$2" \
            --competency-id "$3" \
            --parent-checkpoint "$4" \
            --dataset-manifest "$5" \
            --dataset-chunk "$6" \
            --seed "$7" \
            --max-steps "$8" \
            --max-tokens "$9"
        ;;
    train)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        session_id="$1"
        directory="$(session_dir "${session_id}")"
        require_file "${directory}/session_manifest.json"
        require_file "$2"
        require_file "$3"
        ensure_build
        cd "${ROOT_DIR}"
        ATTENTION_CODE_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || printf unknown)" \
            "${PYTHON}" scripts/run_session.py \
            --session-dir "${directory}" \
            --executable "${BUILD_DIR}/attention_stage0_training" \
            --train-tokens "$2" \
            --validation-tokens "$3"
        "${PYTHON}" scripts/session_artifacts.py seal --session-dir "${directory}"
        echo "STOP: session is sealed and awaiting manual competency review."
        ;;
    decide)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        directory="$(session_dir "$1")"
        require_file "${directory}/session_manifest.json"
        require_file "$2"
        cd "${ROOT_DIR}"
        "${PYTHON}" scripts/session_artifacts.py decision \
            --session-dir "${directory}" \
            --decision-file "$2"
        promote_if_requested "$1"
        ;;
    status)
        [[ $# -eq 1 ]] || { usage; exit 2; }
        directory="$(session_dir "$1")"
        require_file "${directory}/session_manifest.json"
        cat "${directory}/session_manifest.json"
        printf '\nSTATE: '
        cat "${directory}/STATE"
        ;;
    *)
        usage
        exit 2
        ;;
esac

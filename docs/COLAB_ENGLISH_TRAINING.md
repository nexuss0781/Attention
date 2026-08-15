# Colab English Training Quick Start

The repository does not download the entire FineWeb corpus. The Colab workflow streams a bounded range from the official English FineWeb `sample-10BT` configuration, resolves the dataset revision, writes a selected-document manifest, tokenizes only the requested train and validation budgets, bootstraps an initial compatible parent checkpoint, and stops after automated validation for manual review.

Run this cell after cloning or refreshing the repository:

```bash
%cd /content/Attention
!pip install -q datasets huggingface_hub
!chmod +x run.sh
!./run.sh validate-curriculum
!./run.sh prepare-fineweb /content/attention_fineweb_stage1 0 32 8 4096 1024
!./run.sh bootstrap /content/attention_fineweb_stage1/initial.model.checkpoint
!ATTENTION_SESSION_ROOT=/content/attention_fineweb_stage1/sessions \
  ATTENTION_CHECKPOINT_ROOT=/content/attention_fineweb_stage1/checkpoints \
  ./run.sh init english_session_001 stage1_english_foundation causal_basics \
  /content/attention_fineweb_stage1/initial.model.checkpoint \
  data/english_competency_curriculum_v1.json \
  /content/attention_fineweb_stage1/chunk_manifest.json \
  17 512 4096
!ATTENTION_SESSION_ROOT=/content/attention_fineweb_stage1/sessions \
  ATTENTION_CHECKPOINT_ROOT=/content/attention_fineweb_stage1/checkpoints \
  ./run.sh train english_session_001 \
  /content/attention_fineweb_stage1/train.tokens \
  /content/attention_fineweb_stage1/validation.tokens
```

The command intentionally stops with `AWAITING_REVIEW`. It does not promote the checkpoint automatically. Review `sessions/english_session_001/validation_report.json`, `training_trace.csv`, and the saved checkpoint artifacts. To promote manually, create a decision file containing a valid `PROMOTE`, `RETRY`, or `ABORT` decision and run `./run.sh decide english_session_001 DECISION_FILE`.

The literal string `SESSION_ID` must not be passed to `run.sh train`; it is only a placeholder in documentation. Use the exact initialized ID, such as `english_session_001`. A retry must use a new session ID so the previous session remains immutable.

The preparation command downloads only the selected streaming range, not all FineWeb data. The current Stage 0 configuration is the 2-layer, hidden-32, four-head, feed-forward-128 diagnostic model. Its training path now uses analytical reverse-mode gradients, so the initial `4,096/1,024` token budgets and `512` steps are meaningful but still bounded for Colab CPU execution.

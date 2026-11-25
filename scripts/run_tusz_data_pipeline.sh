#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_tusz_data_pipeline.sh TUSZ_EDF_ROOT WORK_ROOT [PREPROCESS_WORKERS] [CEREBRA_DEVICE]

Example:
  EEGCAP_BIN=/path/to/eegcap/bin/eeg-captioner \
  CEREBRA_PYTHON=/path/to/cerebra/bin/python \
  bash scripts/run_tusz_data_pipeline.sh \
    /data/TUSZ/v2.0.6/edf \
    /data/derived/tusz_train \
    16 \
    cuda:0

The EDF root must contain train/subject/session/montage/*.edf. This script
explicitly processes only the train partition. The WORK_ROOT must be new or
empty because the preprocessor intentionally refuses to overwrite subject H5
files.
EOF
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

EDF_ROOT=$1
WORK_ROOT=$2
PREPROCESS_WORKERS=${3:-8}
CEREBRA_DEVICE=${4:-cuda:0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GRA_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
EEGCAP_ROOT="$GRA_ROOT/EEG-Captioner"
CEREBRA_ENGINE="$GRA_ROOT/CerebraGloss/DataEngine"
CEREBRA_ADAPTER="$CEREBRA_ENGINE/cerebragloss_pipeline.py"
CEREBRA_CHECKPOINT="$CEREBRA_ENGINE/checkpoints/model.pkl"

EEGCAP_BIN=${EEGCAP_BIN:-eeg-captioner}
CEREBRA_PYTHON=${CEREBRA_PYTHON:-python}

PROCESSED_ROOT="$WORK_ROOT/processed"
FEATURES_ROOT="$WORK_ROOT/features"
SUMMARIES_ROOT="$WORK_ROOT/summaries"
CEREBRA_ROOT="$WORK_ROOT/cerebragloss"
CALIBRATION_PATH="$WORK_ROOT/config/qa_calibration.tusz_train.json"
RULE_QA_PATH="$WORK_ROOT/annotations/signal_rule_qa.jsonl"
MANIFEST_ROOT="$WORK_ROOT/manifests"

if [[ ! -d "$EDF_ROOT" ]]; then
  echo "TUSZ EDF root does not exist: $EDF_ROOT" >&2
  exit 1
fi
if ! command -v "$EEGCAP_BIN" >/dev/null 2>&1; then
  echo "EEG-Captioner command is unavailable: $EEGCAP_BIN" >&2
  exit 1
fi
if ! command -v "$CEREBRA_PYTHON" >/dev/null 2>&1; then
  echo "CerebraGloss Python is unavailable: $CEREBRA_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CEREBRA_ADAPTER" ]]; then
  echo "Missing shared-H5 CerebraGloss adapter: $CEREBRA_ADAPTER" >&2
  exit 1
fi
if [[ ! -f "$CEREBRA_CHECKPOINT" ]]; then
  echo "Missing CerebraGloss YOLO checkpoint: $CEREBRA_CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$CALIBRATION_PATH" || -e "$RULE_QA_PATH" \
      || -e "$MANIFEST_ROOT/stage2_eeg_projector.train.jsonl" \
      || -e "$MANIFEST_ROOT/stage3_multimodal_sft.train.jsonl" \
      || -e "$MANIFEST_ROOT/stage_manifests.train.metadata.json" ]]; then
  echo "Refusing to overwrite an existing calibration, QA, or manifest output under: $WORK_ROOT" >&2
  exit 1
fi
if find "$PROCESSED_ROOT" -maxdepth 1 -name '*.h5' -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to overwrite existing preprocessed H5 files: $PROCESSED_ROOT" >&2
  exit 1
fi

mkdir -p \
  "$PROCESSED_ROOT" \
  "$FEATURES_ROOT" \
  "$SUMMARIES_ROOT" \
  "$CEREBRA_ROOT" \
  "$(dirname "$CALIBRATION_PATH")" \
  "$(dirname "$RULE_QA_PATH")" \
  "$MANIFEST_ROOT"

echo '[1/7] Preprocessing TUSZ train to shared 19-channel, 10-second H5 windows'
cd "$EEGCAP_ROOT"
"$EEGCAP_BIN" preprocess \
  --edf-root "$EDF_ROOT" \
  --partition train \
  --output "$PROCESSED_ROOT" \
  --workers "$PREPROCESS_WORKERS"

echo '[2/7] Extracting EEG-Captioner numerical signal features'
"$EEGCAP_BIN" features \
  --config "$EEGCAP_ROOT/configs/features.yaml" \
  --input-root "$PROCESSED_ROOT" \
  --output-root "$FEATURES_ROOT" \
  --disable-source-network

echo '[3/7] Rendering numerical feature summaries'
"$EEGCAP_BIN" summaries \
  --input-glob "$FEATURES_ROOT/*.h5" \
  --output-dir "$SUMMARIES_ROOT"

echo '[4/7] Fitting the four rule-QA thresholds on TUSZ train only'
"$EEGCAP_BIN" fit-qa-calibration \
  --features-root "$FEATURES_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --output "$CALIBRATION_PATH"

echo '[5/7] Exporting signal-property and seizure-presence rule QA'
"$EEGCAP_BIN" rule-qa \
  --processed "$PROCESSED_ROOT" \
  --summaries "$SUMMARIES_ROOT" \
  --calibration "$CALIBRATION_PATH" \
  --output "$RULE_QA_PATH"

echo '[6/7] Running CerebraGloss waveform, artifact, background, plot, and rule-caption pipeline'
cd "$CEREBRA_ENGINE"
while IFS= read -r -d '' h5_path; do
  subject_id=$(basename "$h5_path" .h5)
  "$CEREBRA_PYTHON" "$CEREBRA_ADAPTER" \
    --h5 "$h5_path" \
    --all \
    --output-dir "$CEREBRA_ROOT/$subject_id" \
    --device "$CEREBRA_DEVICE"
done < <(find "$PROCESSED_ROOT" -maxdepth 1 -type f -name '*.h5' -print0 | sort -z)

echo '[7/7] Building separate Stage-2 EEG-QA and Stage-3 EEG+image training manifests'
cd "$EEGCAP_ROOT"
"$EEGCAP_BIN" stage-manifests \
  --split train \
  --rule-qa "$RULE_QA_PATH" \
  --cerebragloss-root "$CEREBRA_ROOT" \
  --output-dir "$MANIFEST_ROOT"

echo 'Pipeline completed.'
echo "Processed H5:       $PROCESSED_ROOT"
echo "Signal features:    $FEATURES_ROOT"
echo "Signal summaries:   $SUMMARIES_ROOT"
echo "Frozen calibration: $CALIBRATION_PATH"
echo "Signal rule QA:      $RULE_QA_PATH"
echo "CerebraGloss data:   $CEREBRA_ROOT"
echo "Training manifests:  $MANIFEST_ROOT"

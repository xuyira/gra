"""Export EEG-Captioner's deterministic signal-property QA without captions."""

from collections import Counter
from pathlib import Path
import os

import h5py

from eeg_captioner.io import read_jsonl, sha256, write_json, write_jsonl
from .calibration import load_rule_calibration, validate_processed_contract
from .rules import MODEL_CHANNELS, parse_summary_features, script_answer
from .summaries import SEGMENT_STEM, segment_summaries


SEIZURE_QUESTION = {
    "id": "tusz_seizure_presence",
    "type": "seizure_presence",
    "options": ["Yes", "No"],
    "question": (
        "Is seizure activity present in this EEG segment? Options: "
        "A. Yes B. No. "
        "Answer with only the option letter and label."
    ),
}


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _read_store_metadata(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        trials = [_decode(value) for value in f["trial_ids"][:]]
        presegmented = bool(f.attrs.get("presegmented_10s", False))
        segment_rows = (
            {_decode(value): i for i, value in enumerate(f["segment_ids"][:])}
            if presegmented and "segment_ids" in f
            else None
        )
        n_rows = len(trials)
        source_paths = (
            [_decode(value) for value in f["source_paths"][:]]
            if "source_paths" in f
            else [""] * n_rows
        )
        if "seizure_label" not in f or "seizure_fraction" not in f:
            raise ValueError(
                f"Missing required TUSZ seizure label metadata in {path}"
            )
        seizure_labels = [int(value) for value in f["seizure_label"][:]]
        seizure_fractions = [float(value) for value in f["seizure_fraction"][:]]
        if len(seizure_labels) != n_rows or len(seizure_fractions) != n_rows:
            raise ValueError(f"TUSZ seizure metadata does not match data in {path}")
        if any(label not in (0, 1) for label in seizure_labels):
            raise ValueError(f"TUSZ seizure labels must be binary in {path}")
        return {
            "trials": trials,
            "channels": {_decode(value).upper() for value in f["channels"][:]},
            "segment_rows": segment_rows,
            "source_paths": source_paths,
            "seizure_labels": seizure_labels,
            "seizure_fractions": seizure_fractions,
        }


def _locate_segment(meta: dict, trial_id: str, segment: str) -> tuple[int, int, str]:
    if meta["segment_rows"] is not None:
        segment_key = f"{trial_id}__seg{int(segment):04d}"
        if segment_key not in meta["segment_rows"]:
            raise KeyError(f"Missing preprocessed segment: {segment_key}")
        return meta["segment_rows"][segment_key], 0, segment_key
    if trial_id not in meta["trials"]:
        raise KeyError(f"Missing preprocessed trial: {trial_id}")
    return meta["trials"].index(trial_id), int(segment) * 2000, (
        f"{trial_id}__seg{int(segment):04d}"
    )


def export_rule_qa(
    processed: str | Path,
    summaries: str | Path,
    output: str | Path,
    *,
    calibration_path: str | Path | None = None,
) -> dict:
    """Flatten valid deterministic questions into one JSONL record per QA pair."""
    processed, summaries, output = map(Path, (processed, summaries, output))
    questions_path = Path(__file__).with_name("questions_25.jsonl")
    questions = read_jsonl(questions_path)
    calibration = load_rule_calibration(calibration_path)

    rows = []
    stores = {}
    rejections = Counter()
    eligible_by_type = Counter()
    eligible_by_question = Counter()
    segment_count = 0

    for summary_path in segment_summaries(summaries):
        match = SEGMENT_STEM.fullmatch(summary_path.stem)
        if match is None:
            continue
        filename, trial_id, segment = match.groups()
        raw_path = processed / f"{filename}.h5"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing preprocessed H5 for {summary_path}: {raw_path}")
        if raw_path not in stores:
            validate_processed_contract(raw_path, calibration)
            stores[raw_path] = _read_store_metadata(raw_path)
        meta = stores[raw_path]
        required = set(MODEL_CHANNELS) - {"OZ"}
        if not required.issubset(meta["channels"]):
            raise ValueError(f"{raw_path} is missing standard-19 channels")

        row_index, start_sample, raw_segment_id = _locate_segment(
            meta, trial_id, segment
        )
        features = parse_summary_features(summary_path)
        segment_count += 1

        for head_index, question in enumerate(questions):
            label, reason = script_answer(question, features, calibration)
            if label is None:
                rejections[reason or "rule_invalid"] += 1
                continue

            answer_index = question["options"].index(label)
            answer = f"{chr(65 + answer_index)}. {label}"
            rows.append(
                {
                    "id": f"{summary_path.stem}__{question['id']}",
                    "segment_id": summary_path.stem,
                    "raw_segment_id": raw_segment_id,
                    "eeg_path": os.path.relpath(
                        raw_path.resolve(), output.parent.resolve()
                    ),
                    "summary_path": os.path.relpath(
                        summary_path.resolve(), output.parent.resolve()
                    ),
                    "trial_index": row_index,
                    "start_sample": start_sample,
                    "source_path": meta["source_paths"][row_index],
                    "seizure_label": meta["seizure_labels"][row_index],
                    "seizure_fraction": meta["seizure_fractions"][row_index],
                    "question_id": question["id"],
                    "head_index": head_index,
                    "question_type": question["type"],
                    "question": question["question"],
                    "options": question["options"],
                    "answer_index": answer_index,
                    "answer_label": label,
                    "answer": answer,
                    "annotation_source": "eeg_captioner_rule",
                }
            )
            eligible_by_type[question["type"]] += 1
            eligible_by_question[question["id"]] += 1

        seizure_label = meta["seizure_labels"][row_index]
        seizure_question = SEIZURE_QUESTION
        seizure_answer_index = 0 if seizure_label == 1 else 1
        seizure_answer_label = seizure_question["options"][seizure_answer_index]
        rows.append(
            {
                "id": f"{summary_path.stem}__{seizure_question['id']}",
                "segment_id": summary_path.stem,
                "raw_segment_id": raw_segment_id,
                "eeg_path": os.path.relpath(
                    raw_path.resolve(), output.parent.resolve()
                ),
                "summary_path": os.path.relpath(
                    summary_path.resolve(), output.parent.resolve()
                ),
                "trial_index": row_index,
                "start_sample": start_sample,
                "source_path": meta["source_paths"][row_index],
                "seizure_label": seizure_label,
                "seizure_fraction": meta["seizure_fractions"][row_index],
                "question_id": seizure_question["id"],
                "head_index": None,
                "question_type": seizure_question["type"],
                "question": seizure_question["question"],
                "options": seizure_question["options"],
                "answer_index": seizure_answer_index,
                "answer_label": seizure_answer_label,
                "answer": f"{chr(65 + seizure_answer_index)}. {seizure_answer_label}",
                "annotation_source": "tusz_seizure_annotation",
            }
        )
        eligible_by_type[seizure_question["type"]] += 1
        eligible_by_question[seizure_question["id"]] += 1

    if not rows:
        raise ValueError("No eligible EEG rule-QA examples were generated")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate rule-QA IDs generated")

    write_jsonl(output, rows)
    metadata = {
        "schema": 1,
        "annotation_source": "eeg_captioner_rule",
        "segments_seen": segment_count,
        "questions_per_segment": len(questions) + 1,
        "qa_examples": len(rows),
        "calibration_id": calibration.calibration_id,
        "calibration_path": str(calibration.path),
        "calibration_sha256": calibration.file_sha256,
        "min_margin": calibration.min_margin,
        "rule_thresholds": {
            "temporal_pattern": dict(calibration.temporal_pattern),
            "lateral_dominance": dict(calibration.lateral_dominance),
        },
        "eligible_by_type": dict(sorted(eligible_by_type.items())),
        "eligible_by_question": dict(sorted(eligible_by_question.items())),
        "rejections": dict(sorted(rejections.items())),
        "questions_sha256": sha256(questions_path),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata

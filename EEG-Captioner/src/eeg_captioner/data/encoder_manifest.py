"""Build the one-window/25-label manifest used for EEG encoder training."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

from eeg_captioner.io import read_jsonl, sha256, write_json, write_jsonl


NUM_HEADS = 25


def _resolved_from_jsonl(jsonl_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = jsonl_path.parent / path
    return path.resolve()


def _same(existing, incoming, key: str, segment_id: str) -> None:
    if existing != incoming:
        raise ValueError(
            f"Inconsistent {key} for segment {segment_id}: "
            f"{existing!r} != {incoming!r}"
        )


def _read_split(
    qa_path: Path,
    split: str,
    output: Path,
    questions: list[dict],
) -> tuple[list[dict], Counter, dict[int, Counter]]:
    records: dict[str, dict] = {}
    rejection_counts = Counter()
    class_counts: dict[int, Counter] = {
        head: Counter() for head in range(NUM_HEADS)
    }

    for qa in read_jsonl(qa_path):
        segment_id = str(qa["segment_id"])
        eeg_path = _resolved_from_jsonl(qa_path, qa["eeg_path"])
        summary_path = _resolved_from_jsonl(qa_path, qa["summary_path"])
        identity = {
            "eeg_path_abs": str(eeg_path),
            "summary_path_abs": str(summary_path),
            "trial_index": int(qa["trial_index"]),
            "start_sample": int(qa["start_sample"]),
        }
        record = records.get(segment_id)
        if record is None:
            record = {
                "id": segment_id,
                **identity,
                "raw_segment_id": str(qa.get("raw_segment_id", "")),
                "source_path": str(qa.get("source_path", "")),
                "seizure_label": int(qa["seizure_label"]),
                "seizure_fraction": float(qa["seizure_fraction"]),
                "labels": [-100] * NUM_HEADS,
                "split": split,
            }
            records[segment_id] = record
        else:
            for key, value in identity.items():
                _same(record[key], value, key, segment_id)
            _same(
                record["seizure_label"],
                int(qa["seizure_label"]),
                "seizure_label",
                segment_id,
            )
            _same(
                record["seizure_fraction"],
                float(qa["seizure_fraction"]),
                "seizure_fraction",
                segment_id,
            )

        head = qa.get("head_index")
        if head is None:
            if qa.get("question_type") != "seizure_presence":
                raise ValueError(
                    f"Unexpected headless QA for segment {segment_id}: "
                    f"{qa.get('question_id')}"
                )
            continue
        head = int(head)
        if head < 0 or head >= NUM_HEADS:
            raise ValueError(f"Invalid head_index={head} for segment {segment_id}")
        question = questions[head]
        if str(qa["question_id"]) != str(question["id"]):
            raise ValueError(
                f"Question/head mismatch for segment {segment_id}: "
                f"head {head} expects {question['id']}, got {qa['question_id']}"
            )
        answer_index = int(qa["answer_index"])
        if answer_index < 0 or answer_index >= len(question["options"]):
            raise ValueError(
                f"Invalid answer_index={answer_index} for head={head}, "
                f"segment={segment_id}"
            )
        if record["labels"][head] != -100:
            raise ValueError(
                f"Duplicate encoder label for segment={segment_id}, head={head}"
            )
        record["labels"][head] = answer_index

    rows = []
    for segment_id in sorted(records):
        record = records[segment_id]
        valid = sum(label != -100 for label in record["labels"])
        if valid == 0:
            rejection_counts["no_valid_property_label"] += 1
            continue
        for head, label in enumerate(record["labels"]):
            if label != -100:
                class_counts[head][label] += 1
        record["valid_label_count"] = valid
        record["eeg_path"] = os.path.relpath(
            record.pop("eeg_path_abs"), output.parent.resolve()
        )
        record["summary_path"] = os.path.relpath(
            record.pop("summary_path_abs"), output.parent.resolve()
        )
        rows.append(record)
    return rows, rejection_counts, class_counts


def build_encoder_manifest(
    train_qa: str | Path,
    val_qa: str | Path,
    test_qa: str | Path,
    output: str | Path,
    *,
    allow_incomplete_train_coverage: bool = False,
) -> dict:
    """Combine official TUSZ partitions into the encoder training manifest."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite encoder manifest: {output}")
    sources = {
        "train": Path(train_qa),
        "val": Path(val_qa),
        "test": Path(test_qa),
    }
    for split, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} rule-QA JSONL: {path}")

    questions_path = Path(__file__).with_name("questions_25.jsonl")
    questions = read_jsonl(questions_path)
    if len(questions) != NUM_HEADS:
        raise ValueError(f"Expected {NUM_HEADS} encoder questions")

    rows = []
    rejections = Counter()
    support: dict[str, dict[int, Counter]] = {}
    patients_by_split: dict[str, set[str]] = {}
    for split, qa_path in sources.items():
        split_rows, split_rejections, class_counts = _read_split(
            qa_path.resolve(), split, output.resolve(), questions
        )
        if not split_rows:
            raise ValueError(f"No encoder rows generated for split={split}")
        rows.extend(split_rows)
        rejections.update(
            {f"{split}/{key}": value for key, value in split_rejections.items()}
        )
        support[split] = class_counts
        patients_by_split[split] = {Path(row["eeg_path"]).stem for row in split_rows}

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients_by_split[left] & patients_by_split[right]
        if overlap:
            raise ValueError(
                f"Subject/H5 leakage between {left} and {right}: "
                f"{sorted(overlap)[:10]}"
            )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate segment IDs across encoder splits")

    missing_train_classes = []
    for head, question in enumerate(questions):
        for class_index, option in enumerate(question["options"]):
            if support["train"][head][class_index] == 0:
                missing_train_classes.append(
                    {
                        "head_index": head,
                        "question_id": question["id"],
                        "class_index": class_index,
                        "option": option,
                    }
                )
    if missing_train_classes and not allow_incomplete_train_coverage:
        raise ValueError(
            "Train split is missing classes required by the balanced sampler; "
            "inspect the metadata or use --allow-incomplete-train-coverage only "
            "for a smoke test"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    split_counts = Counter(row["split"] for row in rows)
    metadata = {
        "schema": 1,
        "task": "eeg_encoder_25_heads",
        "segments": len(rows),
        "splits": dict(split_counts),
        "split_policy": {
            "train": "TUSZ train",
            "val": "TUSZ dev",
            "test": "TUSZ eval",
        },
        "questions_sha256": sha256(questions_path),
        "source_qa": {
            split: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for split, path in sources.items()
        },
        "valid_labels": {
            split: int(
                sum(sum(counter.values()) for counter in head_counts.values())
            )
            for split, head_counts in support.items()
        },
        "class_support": {
            split: {
                questions[head]["id"]: {
                    str(class_index): int(counter[class_index])
                    for class_index in range(len(questions[head]["options"]))
                }
                for head, counter in head_counts.items()
            }
            for split, head_counts in support.items()
        },
        "missing_train_classes": missing_train_classes,
        "rejections": dict(rejections),
        "note": (
            "Seizure labels are retained as metadata only; encoder supervision "
            "remains the original fixed 25 signal-property heads."
        ),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata

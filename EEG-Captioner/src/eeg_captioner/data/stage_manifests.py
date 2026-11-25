"""Build stage-specific training manifests from rule QA and CerebraGloss.

The source QA file can contain millions of rows, so this module deliberately
streams it.  A temporary SQLite index is used only for window-level joins and
coverage checks; the QA examples themselves are never accumulated in memory.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

import h5py

from eeg_captioner.channels import STANDARD_19_CHANNELS
from eeg_captioner.io import sha256


CAPTION_INSTRUCTION = "Describe the EEG signal characteristics in this segment."
VALID_SPLITS = {"train", "val", "test"}


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _resolve_from(source: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _relative_to_manifest(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), output_dir.resolve())


def _same_float(left, right, *, key: str, segment_id: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Inconsistent {key} for segment {segment_id}: {left!r} != {right!r}"
        )


def _json_line(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, value


def _validate_qa(qa: dict, *, source: Path, line_number: int) -> None:
    location = f"{source}:{line_number}"
    for key in ("id", "segment_id", "raw_segment_id", "question", "answer"):
        if not isinstance(qa[key], str) or not qa[key].strip():
            raise ValueError(f"Empty {key} at {location}")
    if int(qa["seizure_label"]) not in (0, 1):
        raise ValueError(f"seizure_label must be binary at {location}")
    options = qa["options"]
    if not isinstance(options, list) or not options:
        raise ValueError(f"Expected nonempty options at {location}")
    answer_index = int(qa["answer_index"])
    if answer_index < 0 or answer_index >= len(options):
        raise ValueError(f"answer_index is out of range at {location}")
    if str(options[answer_index]) != str(qa["answer_label"]):
        raise ValueError(f"answer_index/answer_label mismatch at {location}")
    if qa["question_type"] == "seizure_presence":
        if options != ["Yes", "No"]:
            raise ValueError(f"Unexpected seizure options at {location}")
        expected = 0 if int(qa["seizure_label"]) == 1 else 1
        if answer_index != expected:
            raise ValueError(f"Seizure QA disagrees with seizure_label at {location}")


class _H5Validator:
    """Validate repeated QA references while keeping only one H5 open."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.handle: h5py.File | None = None
        self.segment_ids: list[str] = []

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
        self.path = None
        self.handle = None
        self.segment_ids = []

    def _open(self, path: Path) -> None:
        self.close()
        if not path.is_file():
            raise FileNotFoundError(f"Missing EEG H5: {path}")
        handle = h5py.File(path, "r")
        try:
            if float(handle.attrs.get("sample_rate", 0.0)) != 200.0:
                raise ValueError(f"Expected 200 Hz EEG H5: {path}")
            if not bool(handle.attrs.get("presegmented_10s", False)):
                raise ValueError(f"Expected presegmented_10s EEG H5: {path}")
            if "data" not in handle or len(handle["data"].shape) != 3:
                raise ValueError(f"Missing three-dimensional EEG data in {path}")
            if tuple(handle["data"].shape[1:]) != (19, 2000):
                raise ValueError(
                    f"Expected EEG rows [19,2000], got {handle['data'].shape} in {path}"
                )
            channels = tuple(_decode(value).upper() for value in handle["channels"][:])
            if channels != STANDARD_19_CHANNELS:
                raise ValueError(f"Unexpected standard-19 channel order in {path}")
            segment_ids = [_decode(value) for value in handle["segment_ids"][:]]
            if len(segment_ids) != handle["data"].shape[0]:
                raise ValueError(f"segment_ids/data length mismatch in {path}")
            for key in ("seizure_label", "seizure_fraction"):
                if key not in handle or len(handle[key]) != len(segment_ids):
                    raise ValueError(f"Missing or misaligned {key} in {path}")
        except Exception:
            handle.close()
            raise
        self.path = path
        self.handle = handle
        self.segment_ids = segment_ids

    def validate(self, path: Path, qa: dict) -> None:
        if self.path != path:
            self._open(path)
        assert self.handle is not None
        trial_index = int(qa["trial_index"])
        if trial_index < 0 or trial_index >= len(self.segment_ids):
            raise IndexError(
                f"trial_index={trial_index} is out of range in {path} for "
                f"{qa['raw_segment_id']}"
            )
        raw_segment_id = str(qa["raw_segment_id"])
        if self.segment_ids[trial_index] != raw_segment_id:
            raise ValueError(
                f"H5 row mismatch in {path}: row {trial_index} is "
                f"{self.segment_ids[trial_index]!r}, not {raw_segment_id!r}"
            )
        if int(qa.get("start_sample", 0)) != 0:
            raise ValueError(
                f"Presegmented EEG must use start_sample=0 for {raw_segment_id}"
            )
        h5_label = int(self.handle["seizure_label"][trial_index])
        if h5_label != int(qa["seizure_label"]):
            raise ValueError(
                f"H5/QA seizure_label mismatch for {raw_segment_id}: "
                f"{h5_label} != {qa['seizure_label']}"
            )
        _same_float(
            self.handle["seizure_fraction"][trial_index],
            qa["seizure_fraction"],
            key="H5/QA seizure_fraction",
            segment_id=raw_segment_id,
        )


def _index_cerebragloss(root: Path, connection: sqlite3.Connection) -> int:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing CerebraGloss output directory: {root}")
    connection.execute(
        "CREATE TABLE cerebra ("
        "segment_id TEXT PRIMARY KEY, json_path TEXT NOT NULL, matched INTEGER NOT NULL DEFAULT 0"
        ") WITHOUT ROWID"
    )
    count = 0
    for path in root.rglob("*.json"):
        segment_id = path.stem
        try:
            connection.execute(
                "INSERT INTO cerebra(segment_id, json_path) VALUES (?, ?)",
                (segment_id, str(path.resolve())),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Duplicate CerebraGloss segment_id={segment_id}: {path}"
            ) from exc
        count += 1
    connection.commit()
    if count == 0:
        raise ValueError(f"No CerebraGloss JSON files found under {root}")
    return count


def _load_cerebragloss(
    connection: sqlite3.Connection,
    raw_segment_id: str,
    *,
    qa: dict,
) -> tuple[dict, Path]:
    indexed = connection.execute(
        "SELECT json_path FROM cerebra WHERE segment_id = ?", (raw_segment_id,)
    ).fetchone()
    if indexed is None:
        raise FileNotFoundError(
            f"Missing CerebraGloss JSON for raw_segment_id={raw_segment_id}"
        )
    json_path = Path(indexed[0])
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid CerebraGloss JSON: {json_path}") from exc
    if data.get("segment_id") != raw_segment_id:
        raise ValueError(
            f"CerebraGloss segment_id mismatch in {json_path}: "
            f"{data.get('segment_id')!r} != {raw_segment_id!r}"
        )
    if int(data.get("row_index", -1)) != int(qa["trial_index"]):
        raise ValueError(f"CerebraGloss row_index mismatch for {raw_segment_id}")
    if float(data.get("sample_rate_hz", 0.0)) != 200.0:
        raise ValueError(f"CerebraGloss sample rate is not 200 Hz for {raw_segment_id}")
    channels = tuple(str(value).upper() for value in data.get("channels", ()))
    if channels != STANDARD_19_CHANNELS:
        raise ValueError(f"CerebraGloss channel order mismatch for {raw_segment_id}")
    if "signal_shape" in data and tuple(data["signal_shape"]) != (19, 2000):
        raise ValueError(f"CerebraGloss signal shape mismatch for {raw_segment_id}")
    if int(data["seizure_label"]) != int(qa["seizure_label"]):
        raise ValueError(f"QA/CerebraGloss seizure_label mismatch for {raw_segment_id}")
    _same_float(
        data["seizure_fraction"],
        qa["seizure_fraction"],
        key="QA/CerebraGloss seizure_fraction",
        segment_id=raw_segment_id,
    )
    caption = data.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"Empty CerebraGloss caption for {raw_segment_id}")
    image_path = json_path.with_suffix(".png")
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing CerebraGloss PNG: {image_path}")
    connection.execute(
        "UPDATE cerebra SET matched = 1 WHERE segment_id = ?", (raw_segment_id,)
    )
    return data, image_path


def _eeg_content(eeg_path: str, qa: dict) -> dict:
    return {
        "type": "eeg",
        "eeg": eeg_path,
        "trial_index": int(qa["trial_index"]),
        "start_sample": int(qa.get("start_sample", 0)),
    }


def _qa_record(
    qa: dict,
    *,
    split: str,
    stage: str,
    eeg_path: str,
    image_path: str | None = None,
) -> dict:
    instruction = str(qa["question"])
    answer = str(qa["answer"])
    user_content = [_eeg_content(eeg_path, qa)]
    modalities = ["eeg"]
    if image_path is not None:
        user_content.append({"type": "image", "image": image_path})
        modalities.append("image")
    user_content.append({"type": "text", "text": instruction})
    record = {
        "id": str(qa["id"]),
        "segment_id": str(qa["segment_id"]),
        "raw_segment_id": str(qa["raw_segment_id"]),
        "split": split,
        "stage": stage,
        "task_type": "qa",
        "modalities": modalities,
        "eeg_path": eeg_path,
        "trial_index": int(qa["trial_index"]),
        "start_sample": int(qa.get("start_sample", 0)),
        "source_path": str(qa.get("source_path", "")),
        "seizure_label": int(qa["seizure_label"]),
        "seizure_fraction": float(qa["seizure_fraction"]),
        "question_id": str(qa["question_id"]),
        "question_type": str(qa["question_type"]),
        "head_index": qa.get("head_index"),
        "instruction": instruction,
        "options": list(qa["options"]),
        "answer_index": int(qa["answer_index"]),
        "answer_label": str(qa["answer_label"]),
        "answer": answer,
        "annotation_source": str(qa["annotation_source"]),
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }
    if image_path is not None:
        record["image_path"] = image_path
    return record


def _caption_record(
    qa: dict,
    cerebra: dict,
    *,
    split: str,
    eeg_path: str,
    image_path: str,
) -> dict:
    answer = cerebra["caption"].strip()
    user_content = [
        _eeg_content(eeg_path, qa),
        {"type": "image", "image": image_path},
        {"type": "text", "text": CAPTION_INSTRUCTION},
    ]
    return {
        "id": f"{qa['segment_id']}__cerebragloss_rule_caption",
        "segment_id": str(qa["segment_id"]),
        "raw_segment_id": str(qa["raw_segment_id"]),
        "split": split,
        "stage": "multimodal_sft",
        "task_type": "caption",
        "modalities": ["eeg", "image"],
        "eeg_path": eeg_path,
        "trial_index": int(qa["trial_index"]),
        "start_sample": int(qa.get("start_sample", 0)),
        "image_path": image_path,
        "source_path": str(qa.get("source_path", "")),
        "seizure_label": int(qa["seizure_label"]),
        "seizure_fraction": float(qa["seizure_fraction"]),
        "instruction": CAPTION_INSTRUCTION,
        "answer": answer,
        "annotation_source": "cerebragloss_rule_caption",
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }


def build_stage_manifests(
    split: str,
    rule_qa: str | Path,
    cerebragloss_root: str | Path,
    output_dir: str | Path,
) -> dict:
    """Build Stage-2 EEG QA and Stage-3 EEG+image QA/caption JSONL files."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
    rule_qa = Path(rule_qa).resolve()
    cerebragloss_root = Path(cerebragloss_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not rule_qa.is_file():
        raise FileNotFoundError(f"Missing rule-QA JSONL: {rule_qa}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2_path = output_dir / f"stage2_eeg_projector.{split}.jsonl"
    stage3_path = output_dir / f"stage3_multimodal_sft.{split}.jsonl"
    metadata_path = output_dir / f"stage_manifests.{split}.metadata.json"
    for path in (stage2_path, stage3_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite training manifest: {path}")

    temporary_paths: list[Path] = []
    validator = _H5Validator()
    connection: sqlite3.Connection | None = None
    stage2_file = None
    stage3_file = None
    try:
        stage2_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir,
            prefix=f".{stage2_path.name}.", suffix=".tmp", delete=False,
        )
        stage3_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir,
            prefix=f".{stage3_path.name}.", suffix=".tmp", delete=False,
        )
        stage2_tmp = Path(stage2_file.name)
        stage3_tmp = Path(stage3_file.name)
        temporary_paths.extend((stage2_tmp, stage3_tmp))
        db_handle = tempfile.NamedTemporaryFile(
            dir=output_dir, prefix=f".stage_join.{split}.", suffix=".sqlite", delete=False
        )
        db_path = Path(db_handle.name)
        db_handle.close()
        temporary_paths.append(db_path)
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        cerebra_count = _index_cerebragloss(cerebragloss_root, connection)
        connection.execute(
            "CREATE TABLE qa_segments ("
            "raw_segment_id TEXT PRIMARY KEY, segment_id TEXT NOT NULL, "
            "eeg_path TEXT NOT NULL, trial_index INTEGER NOT NULL, "
            "seizure_label INTEGER NOT NULL, seizure_fraction REAL NOT NULL, "
            "seizure_questions INTEGER NOT NULL DEFAULT 0"
            ") WITHOUT ROWID"
        )

        qa_count = 0
        caption_count = 0
        by_question_type = Counter()
        by_annotation_source = Counter()
        with stage2_file, stage3_file:
            for line_number, qa in _iter_jsonl(rule_qa):
                required = (
                    "id", "segment_id", "raw_segment_id", "eeg_path",
                    "trial_index", "seizure_label", "seizure_fraction",
                    "question_id", "question_type", "question", "options",
                    "answer_index", "answer_label", "answer", "annotation_source",
                )
                missing = [key for key in required if key not in qa]
                if missing:
                    raise ValueError(
                        f"Missing QA fields {missing} at {rule_qa}:{line_number}"
                    )
                _validate_qa(qa, source=rule_qa, line_number=line_number)
                raw_segment_id = str(qa["raw_segment_id"])
                eeg_abs = _resolve_from(rule_qa, str(qa["eeg_path"]))
                validator.validate(eeg_abs, qa)
                eeg_rel = _relative_to_manifest(eeg_abs, output_dir)

                existing = connection.execute(
                    "SELECT segment_id, eeg_path, trial_index, seizure_label, "
                    "seizure_fraction FROM qa_segments WHERE raw_segment_id = ?",
                    (raw_segment_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO qa_segments(raw_segment_id, segment_id, eeg_path, "
                        "trial_index, seizure_label, seizure_fraction) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            raw_segment_id, str(qa["segment_id"]), str(eeg_abs),
                            int(qa["trial_index"]), int(qa["seizure_label"]),
                            float(qa["seizure_fraction"]),
                        ),
                    )
                    cerebra, image_abs = _load_cerebragloss(
                        connection, raw_segment_id, qa=qa
                    )
                    image_rel = _relative_to_manifest(image_abs, output_dir)
                    _json_line(
                        stage3_file,
                        _caption_record(
                            qa, cerebra, split=split, eeg_path=eeg_rel,
                            image_path=image_rel,
                        ),
                    )
                    caption_count += 1
                else:
                    expected = (
                        str(qa["segment_id"]), str(eeg_abs), int(qa["trial_index"]),
                        int(qa["seizure_label"]),
                    )
                    if existing[:4] != expected:
                        raise ValueError(
                            f"Inconsistent QA window identity for {raw_segment_id}"
                        )
                    _same_float(
                        existing[4], qa["seizure_fraction"],
                        key="QA seizure_fraction", segment_id=raw_segment_id,
                    )
                    indexed = connection.execute(
                        "SELECT json_path FROM cerebra WHERE segment_id = ?",
                        (raw_segment_id,),
                    ).fetchone()
                    assert indexed is not None
                    image_rel = _relative_to_manifest(
                        Path(indexed[0]).with_suffix(".png"), output_dir
                    )

                if qa["question_type"] == "seizure_presence":
                    connection.execute(
                        "UPDATE qa_segments SET seizure_questions = seizure_questions + 1 "
                        "WHERE raw_segment_id = ?", (raw_segment_id,),
                    )

                _json_line(
                    stage2_file,
                    _qa_record(
                        qa, split=split, stage="eeg_projector_alignment",
                        eeg_path=eeg_rel,
                    ),
                )
                _json_line(
                    stage3_file,
                    _qa_record(
                        qa, split=split, stage="multimodal_sft",
                        eeg_path=eeg_rel, image_path=image_rel,
                    ),
                )
                qa_count += 1
                by_question_type[str(qa["question_type"])] += 1
                by_annotation_source[str(qa["annotation_source"])] += 1

        connection.commit()
        if qa_count == 0:
            raise ValueError("No QA examples were found")
        bad_seizure = connection.execute(
            "SELECT raw_segment_id, seizure_questions FROM qa_segments "
            "WHERE seizure_questions != 1 LIMIT 10"
        ).fetchall()
        if bad_seizure:
            raise ValueError(
                "Every window must have exactly one seizure QA; invalid examples: "
                f"{bad_seizure}"
            )
        unmatched = connection.execute(
            "SELECT segment_id FROM cerebra WHERE matched = 0 LIMIT 10"
        ).fetchall()
        if unmatched:
            raise ValueError(
                "CerebraGloss contains windows absent from rule QA; examples: "
                f"{[row[0] for row in unmatched]}"
            )
        segment_count = int(
            connection.execute("SELECT COUNT(*) FROM qa_segments").fetchone()[0]
        )
        if segment_count != cerebra_count or caption_count != segment_count:
            raise ValueError(
                "Window coverage mismatch: "
                f"QA={segment_count}, CerebraGloss={cerebra_count}, captions={caption_count}"
            )

        metadata = {
            "schema": 1,
            "split": split,
            "source_rule_qa": {
                "path": str(rule_qa),
                "sha256": sha256(rule_qa),
            },
            "source_cerebragloss": {
                "root": str(cerebragloss_root),
                "segments": cerebra_count,
            },
            "stage2_eeg_projector": {
                "path": stage2_path.name,
                "examples": qa_count,
                "qa_examples": qa_count,
                "caption_examples": 0,
                "modalities": ["eeg"],
                "sha256": sha256(stage2_tmp),
            },
            "stage3_multimodal_sft": {
                "path": stage3_path.name,
                "examples": qa_count + caption_count,
                "qa_examples": qa_count,
                "caption_examples": caption_count,
                "modalities": ["eeg", "image"],
                "sha256": sha256(stage3_tmp),
            },
            "segments": segment_count,
            "question_types": dict(sorted(by_question_type.items())),
            "annotation_sources": dict(sorted(by_annotation_source.items())),
            "join_key": "rule_qa.raw_segment_id == cerebragloss.segment_id",
            "path_policy": "Paths inside manifests are relative to the manifest directory.",
            "caption_policy": (
                "One unexpanded CerebraGloss rule caption per window; no teacher-LLM text."
            ),
        }
        metadata_tmp_handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir,
            prefix=f".{metadata_path.name}.", suffix=".tmp", delete=False,
        )
        metadata_tmp = Path(metadata_tmp_handle.name)
        temporary_paths.append(metadata_tmp)
        with metadata_tmp_handle:
            json.dump(metadata, metadata_tmp_handle, ensure_ascii=False, indent=2)
            metadata_tmp_handle.write("\n")

        stage2_tmp.replace(stage2_path)
        stage3_tmp.replace(stage3_path)
        metadata_tmp.replace(metadata_path)
        return metadata
    finally:
        validator.close()
        for handle in (stage2_file, stage3_file):
            if handle is not None and not handle.closed:
                handle.close()
        if connection is not None:
            connection.close()
        for path in temporary_paths:
            if path.exists():
                path.unlink()

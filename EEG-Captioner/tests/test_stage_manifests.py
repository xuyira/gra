import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from eeg_captioner.channels import STANDARD_19_CHANNELS
from eeg_captioner.data.stage_manifests import build_stage_manifests
from eeg_captioner.io import read_jsonl, write_jsonl


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    processed = tmp_path / "processed"
    annotations = tmp_path / "annotations"
    cerebra = tmp_path / "cerebragloss" / "subject"
    processed.mkdir()
    annotations.mkdir()
    cerebra.mkdir(parents=True)

    raw_segment_id = "record_s001_t000__seg0000"
    h5_path = processed / "subject.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["sample_rate"] = 200
        handle.attrs["presegmented_10s"] = True
        handle["data"] = np.zeros((1, 19, 2000), dtype=np.float32)
        handle["channels"] = np.asarray(STANDARD_19_CHANNELS, dtype="S")
        handle["segment_ids"] = np.asarray([raw_segment_id], dtype="S")
        handle["seizure_label"] = np.asarray([1], dtype=np.int8)
        handle["seizure_fraction"] = np.asarray([0.25], dtype=np.float32)

    common = {
        "segment_id": f"subject__{raw_segment_id}",
        "raw_segment_id": raw_segment_id,
        "eeg_path": "../processed/subject.h5",
        "summary_path": "unused.md",
        "trial_index": 0,
        "start_sample": 0,
        "source_path": "train/subject/record.edf",
        "seizure_label": 1,
        "seizure_fraction": 0.25,
    }
    qa_rows = [
        {
            **common,
            "id": f"{common['segment_id']}__dominant",
            "question_id": "dominant",
            "head_index": 0,
            "question_type": "dominant_band",
            "question": "Which band is dominant? Options: A. delta B. alpha.",
            "options": ["delta", "alpha"],
            "answer_index": 1,
            "answer_label": "alpha",
            "answer": "B. alpha",
            "annotation_source": "eeg_captioner_rule",
        },
        {
            **common,
            "id": f"{common['segment_id']}__seizure",
            "question_id": "tusz_seizure_presence",
            "head_index": None,
            "question_type": "seizure_presence",
            "question": "Is seizure activity present? Options: A. Yes B. No.",
            "options": ["Yes", "No"],
            "answer_index": 0,
            "answer_label": "Yes",
            "answer": "A. Yes",
            "annotation_source": "tusz_seizure_annotation",
        },
    ]
    qa_path = annotations / "signal_rule_qa.jsonl"
    write_jsonl(qa_path, qa_rows)

    cerebra_json = cerebra / f"{raw_segment_id}.json"
    cerebra_json.write_text(
        json.dumps(
            {
                "segment_id": raw_segment_id,
                "row_index": 0,
                "sample_rate_hz": 200,
                "channels": list(STANDARD_19_CHANNELS),
                "signal_shape": [19, 2000],
                "seizure_label": 1,
                "seizure_fraction": 0.25,
                "caption": "Alpha background activity is observed without artifacts.",
            }
        ),
        encoding="utf-8",
    )
    cerebra_json.with_suffix(".png").write_bytes(b"not-decoded-by-manifest-builder")
    return qa_path, cerebra.parent, cerebra_json


def test_stage_manifests_keep_stage_supervision_separate(tmp_path):
    qa_path, cerebra_root, _ = _write_sources(tmp_path)
    output = tmp_path / "manifests"

    metadata = build_stage_manifests("train", qa_path, cerebra_root, output)
    stage2 = read_jsonl(output / "stage2_eeg_projector.train.jsonl")
    stage3 = read_jsonl(output / "stage3_multimodal_sft.train.jsonl")

    assert len(stage2) == 2
    assert all(row["task_type"] == "qa" for row in stage2)
    assert all(row["modalities"] == ["eeg"] for row in stage2)
    assert all("image_path" not in row for row in stage2)
    assert all("caption" not in row for row in stage2)

    assert len(stage3) == 3
    assert [row["task_type"] for row in stage3].count("qa") == 2
    assert [row["task_type"] for row in stage3].count("caption") == 1
    assert all(row["modalities"] == ["eeg", "image"] for row in stage3)
    assert all("image_path" in row for row in stage3)
    assert all(
        [item["type"] for item in row["messages"][0]["content"]][:2]
        == ["eeg", "image"]
        for row in stage3
    )

    assert metadata["stage2_eeg_projector"]["caption_examples"] == 0
    assert metadata["stage3_multimodal_sft"]["qa_examples"] == 2
    assert metadata["stage3_multimodal_sft"]["caption_examples"] == 1
    assert metadata["segments"] == 1
    assert not Path(stage2[0]["eeg_path"]).is_absolute()
    assert (output / stage2[0]["eeg_path"]).resolve().is_file()
    assert (output / stage3[0]["image_path"]).resolve().is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_stage_manifests("train", qa_path, cerebra_root, output)


@pytest.mark.parametrize("broken", ["png", "caption"])
def test_stage_manifests_reject_incomplete_cerebragloss(tmp_path, broken):
    qa_path, cerebra_root, cerebra_json = _write_sources(tmp_path)
    if broken == "png":
        cerebra_json.with_suffix(".png").unlink()
    else:
        value = json.loads(cerebra_json.read_text(encoding="utf-8"))
        value["caption"] = ""
        cerebra_json.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError)):
        build_stage_manifests("train", qa_path, cerebra_root, tmp_path / "manifests")
    assert not (tmp_path / "manifests/stage2_eeg_projector.train.jsonl").exists()
    assert not (tmp_path / "manifests/stage3_multimodal_sft.train.jsonl").exists()


def test_stage_manifests_reject_h5_row_mismatch(tmp_path):
    qa_path, cerebra_root, _ = _write_sources(tmp_path)
    rows = read_jsonl(qa_path)
    for row in rows:
        row["raw_segment_id"] = "different_record__seg0000"
    qa_path.unlink()
    write_jsonl(qa_path, rows)

    with pytest.raises(ValueError, match="H5 row mismatch"):
        build_stage_manifests("train", qa_path, cerebra_root, tmp_path / "manifests")

"""Random-access datasets for the stage-specific JSONL training manifests."""

from __future__ import annotations

from array import array
import json
from pathlib import Path

from torch.utils.data import Dataset, Sampler

from .dataset import load_eeg


LANGUAGE_STAGES = {"eeg_projector_alignment", "multimodal_sft"}


class StageManifestDataset(Dataset):
    """Index a large JSONL by byte offset without retaining all rows in RAM."""

    def __init__(self, manifest, *, expected_stage: str, expected_split: str):
        if expected_stage not in LANGUAGE_STAGES:
            raise ValueError(f"Unsupported language stage: {expected_stage}")
        if expected_split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {expected_split}")
        self.manifest = Path(manifest).resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Missing stage manifest: {self.manifest}")
        self.expected_stage = expected_stage
        self.expected_split = expected_split
        self.offsets = array("Q")
        self.qa_indices = array("Q")
        self.caption_indices = array("Q")
        self._handle = None
        self._build_index()

    def _build_index(self):
        with self.manifest.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid JSON at {self.manifest}:{line_number}"
                    ) from exc
                self._validate_row(row, line_number)
                index = len(self.offsets)
                self.offsets.append(offset)
                if row["task_type"] == "qa":
                    self.qa_indices.append(index)
                else:
                    self.caption_indices.append(index)
        if not self.offsets:
            raise ValueError(f"Empty stage manifest: {self.manifest}")
        if self.expected_stage == "eeg_projector_alignment" and self.caption_indices:
            raise ValueError("Stage 2 cannot contain caption records")
        if not self.qa_indices:
            raise ValueError("A language-stage manifest must contain QA records")
        if self.expected_stage == "multimodal_sft" and not self.caption_indices:
            raise ValueError("Stage 3 must contain caption records")

    def _validate_row(self, row, line_number):
        required = {
            "id", "raw_segment_id", "split", "stage", "task_type",
            "eeg_path", "trial_index", "start_sample", "instruction", "answer",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"Missing fields {missing} at {self.manifest}:{line_number}"
            )
        if row["stage"] != self.expected_stage:
            raise ValueError(
                f"Expected stage={self.expected_stage}, got {row['stage']} at "
                f"{self.manifest}:{line_number}"
            )
        if row["split"] != self.expected_split:
            raise ValueError(
                f"Expected split={self.expected_split}, got {row['split']} at "
                f"{self.manifest}:{line_number}"
            )
        if row["task_type"] not in {"qa", "caption"}:
            raise ValueError(f"Invalid task_type at {self.manifest}:{line_number}")
        if not str(row["instruction"]).strip() or not str(row["answer"]).strip():
            raise ValueError(f"Empty instruction/answer at {self.manifest}:{line_number}")
        has_image = "image_path" in row
        if self.expected_stage == "eeg_projector_alignment" and has_image:
            raise ValueError(f"Stage 2 row contains an image at line {line_number}")
        if self.expected_stage == "multimodal_sft" and not has_image:
            raise ValueError(f"Stage 3 row has no image at line {line_number}")

    def __len__(self):
        return len(self.offsets)

    def _open(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.manifest.open("rb")
        return self._handle

    def row(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        handle = self._open()
        handle.seek(self.offsets[index])
        return json.loads(handle.readline())

    def __getitem__(self, index):
        row = self.row(index)
        eeg_path = (self.manifest.parent / row["eeg_path"]).resolve()
        if not eeg_path.is_file():
            raise FileNotFoundError(f"Missing EEG H5: {eeg_path}")
        eeg_row = {
            "eeg_path": str(eeg_path),
            "trial_index": int(row["trial_index"]),
            "start_sample": int(row["start_sample"]),
        }
        value = {
            "x": load_eeg(eeg_row),
            "id": str(row["id"]),
            "raw_segment_id": str(row["raw_segment_id"]),
            "task_type": str(row["task_type"]),
            "instruction": str(row["instruction"]),
            "target": str(row["answer"]),
        }
        if self.expected_stage == "multimodal_sft":
            image_path = (self.manifest.parent / row["image_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing EEG image: {image_path}")
            value["image_path"] = str(image_path)
        return value

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    def __del__(self):
        self.close()


class TaskRatioSampler(Sampler):
    """Deterministically sample QA/caption examples with replacement."""

    def __init__(
        self, dataset, num_samples, *, caption_fraction=0.0, seed=42,
        start_index=0,
    ):
        import numpy as np

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not 0.0 <= caption_fraction <= 1.0:
            raise ValueError("caption_fraction must be between zero and one")
        if caption_fraction and not dataset.caption_indices:
            raise ValueError("caption_fraction is nonzero but the dataset has no captions")
        if caption_fraction < 1.0 and not dataset.qa_indices:
            raise ValueError("QA sampling requested but the dataset has no QA examples")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.caption_fraction = float(caption_fraction)
        self.seed = int(seed)
        self.start_index = int(start_index)
        if self.start_index < 0 or self.start_index >= self.num_samples:
            raise ValueError("start_index must be within the sampling schedule")
        self.epoch = 0
        self._np = np

    def __len__(self):
        return self.num_samples - self.start_index

    def __iter__(self):
        rng = self._np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        qa = self.dataset.qa_indices
        captions = self.dataset.caption_indices
        for index in range(self.num_samples):
            use_caption = (
                int((index + 1) * self.caption_fraction)
                > int(index * self.caption_fraction)
            )
            pool = captions if use_caption else qa
            selected = int(pool[int(rng.integers(0, len(pool)))])
            if index >= self.start_index:
                yield selected

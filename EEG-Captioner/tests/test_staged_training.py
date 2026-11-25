from types import SimpleNamespace
import json

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from eeg_captioner.channels import STANDARD_19_CHANNELS
from eeg_captioner.data.stage_dataset import StageManifestDataset, TaskRatioSampler
from eeg_captioner.io import write_jsonl
from eeg_captioner.models.projector import AtomicEEGProjector
from eeg_captioner.models.qwen_vl_eeg import (
    EEG_PLACEHOLDER_TOKEN,
    EEGVisionLanguageModel,
    QwenVLEEGCollator,
    build_chat_messages,
)


def _stage_fixture(tmp_path, stage):
    h5_path = tmp_path / "processed/subject.h5"
    h5_path.parent.mkdir()
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["sample_rate"] = 200
        handle["data"] = np.zeros((1, 19, 2000), dtype=np.float32)
        handle["channels"] = np.asarray(STANDARD_19_CHANNELS, dtype="S")
        handle["trial_lengths_in_samples"] = np.asarray([2000])
    image = tmp_path / "cerebragloss/window.png"
    image.parent.mkdir()
    image.write_bytes(b"image-presence-only")
    common = {
        "segment_id": "subject__window",
        "raw_segment_id": "window",
        "split": "train",
        "stage": stage,
        "eeg_path": "../processed/subject.h5",
        "trial_index": 0,
        "start_sample": 0,
    }
    rows = [
        {
            **common,
            "id": "qa",
            "task_type": "qa",
            "instruction": "Is seizure activity present?",
            "answer": "B. No",
        }
    ]
    if stage == "multimodal_sft":
        rows[0]["image_path"] = "../cerebragloss/window.png"
        rows.append(
            {
                **common,
                "id": "caption",
                "task_type": "caption",
                "image_path": "../cerebragloss/window.png",
                "instruction": "Describe the EEG.",
                "answer": "Alpha background activity is observed.",
            }
        )
    manifest = tmp_path / "manifests/data.jsonl"
    manifest.parent.mkdir()
    write_jsonl(manifest, rows)
    return manifest


def test_stage_dataset_and_task_ratio_sampler(tmp_path):
    manifest = _stage_fixture(tmp_path, "multimodal_sft")
    dataset = StageManifestDataset(
        manifest, expected_stage="multimodal_sft", expected_split="train"
    )
    assert len(dataset) == 2
    assert dataset[0]["x"].shape == (20, 10, 200)
    assert dataset[1]["task_type"] == "caption"
    assert dataset[1]["image_path"].endswith("window.png")
    sampled = list(TaskRatioSampler(dataset, 8, caption_fraction=0.25, seed=7))
    assert sum(dataset.row(index)["task_type"] == "caption" for index in sampled) == 2


def test_stage_dataset_rejects_cross_stage_rows(tmp_path):
    manifest = _stage_fixture(tmp_path, "eeg_projector_alignment")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["image_path"] = "unexpected.png"
    manifest.unlink()
    write_jsonl(manifest, rows)
    with pytest.raises(ValueError, match="Stage 2 row contains an image"):
        StageManifestDataset(
            manifest,
            expected_stage="eeg_projector_alignment",
            expected_split="train",
        )


def test_chat_messages_separate_stage_modalities():
    example = {
        "instruction": "Describe the EEG.",
        "target": "A description.",
        "image_path": "/tmp/window.png",
    }
    stage2 = build_chat_messages(
        example, stage="eeg_projector_alignment", include_target=True
    )
    stage3 = build_chat_messages(
        example, stage="multimodal_sft", include_target=True
    )
    assert [item["type"] for item in stage2[1]["content"]] == ["text"]
    assert [item["type"] for item in stage3[1]["content"]] == ["image", "text"]
    assert stage3[1]["content"][1]["text"].count(EEG_PLACEHOLDER_TOKEN) == 200
    assert stage3[-1]["content"][0]["text"] == "A description."


class _FakeTokenizer:
    padding_side = "left"


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize
        return "full" if any(row["role"] == "assistant" for row in messages) else "prompt"

    def __call__(self, text, padding, return_tensors, **kwargs):
        assert padding and return_tensors == "pt"
        rows = []
        for value in text:
            base = [1] + [9] * 200 + [2]
            rows.append(base + ([5, 6] if value == "full" else []))
        width = max(map(len, rows))
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row)
            mask[index, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_collator_masks_prompt_and_stacks_eeg():
    collator = QwenVLEEGCollator(
        _FakeProcessor(), stage="eeg_projector_alignment"
    )
    batch = collator(
        [
            {
                "id": "one",
                "task_type": "qa",
                "instruction": "Question",
                "target": "B. No",
                "x": torch.zeros(20, 10, 200),
            }
        ]
    )
    assert batch["x"].shape == (1, 20, 10, 200)
    assert batch["labels"][0, :-2].eq(-100).all()
    assert batch["labels"][0, -2:].tolist() == [5, 6]


class _FakeEncoder(nn.Module):
    def forward(self, eeg):
        return torch.ones(len(eeg), 200, 256, device=eeg.device)


class _FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 12)
        self.output = nn.Linear(12, 16, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, labels, attention_mask=None, use_cache=False):
        # A tiny causal context mixer makes the answer depend on earlier EEG
        # placeholders, as attention would in the real language model.
        hidden = self.embedding(input_ids).cumsum(dim=1)
        logits = self.output(hidden)
        loss = F.cross_entropy(
            logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_eeg_embedding_hook_updates_projector_only():
    qwen = _FakeQwen()
    qwen.requires_grad_(False)
    projector = AtomicEEGProjector(12)
    model = EEGVisionLanguageModel(
        _FakeEncoder(), projector, qwen,
        eeg_token_id=9, qwen_trainable=False,
    )
    ids = torch.tensor([[1] + [9] * 200 + [2, 5]], dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[0, -1] = 5
    loss = model(
        {
            "x": torch.zeros(1, 20, 10, 200),
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": labels,
        }
    )
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in projector.parameters()
    )
    assert all(parameter.grad is None for parameter in qwen.parameters())

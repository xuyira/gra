"""Single-GPU training loop for Stage-2 and Stage-3 Qwen2.5-VL adaptation."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_captioner.data.stage_dataset import StageManifestDataset, TaskRatioSampler
from eeg_captioner.io import sha256, write_json
from eeg_captioner.models.language_ops import scheduled_learning_rate
from eeg_captioner.models.qwen_vl_eeg import QwenVLEEGCollator, load_qwen_vl_eeg


STAGES = {"eeg_projector_alignment", "multimodal_sft"}


def _device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    value = torch.device(requested)
    if value.type == "cuda":
        torch.cuda.set_device(value)
    return value


def _save_checkpoint(path, model, processor, optimizer, step, config, provenance):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), path / "eeg_projector.pt")
    if config["stage"] == "multimodal_sft":
        model.qwen.save_pretrained(path / "qwen_lora")
    processor.save_pretrained(path / "processor")
    state = {
        "step": int(step),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "provenance": provenance,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    torch.save(state, path / "training_state.pt")
    write_json(path / "config.json", config)
    write_json(path / "data_provenance.json", provenance)


@torch.inference_mode()
def _validation_loss(model, dataset, collator, config, device):
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    total = 0.0
    examples = 0
    limit = int(config.get("eval_batches", 20))
    for batch_index, batch in enumerate(loader):
        if batch_index >= limit:
            break
        loss = model(batch)
        count = len(batch["ids"])
        total += float(loss) * count
        examples += count
    if examples == 0:
        raise ValueError("Validation loader yielded no examples")
    return total / examples, examples


def train_language_stage(config):
    """Train the EEG projector alone (Stage 2) or projector+LLM LoRA (Stage 3)."""
    config = dict(config)
    stage = config.get("stage")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}")
    if (
        stage == "multimodal_sft"
        and not config.get("initial_checkpoint")
        and not config.get("resume")
    ):
        raise ValueError(
            "Stage 3 requires a Stage-2 initial_checkpoint or a Stage-3 resume"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The initial Qwen2.5-VL adapter supports one GPU only")
    device = _device(config)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output = Path(config["output_dir"])
    resume = Path(config["resume"]) if config.get("resume") else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Use a new output_dir or set resume: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_manifest = Path(config["train_manifest"]).resolve()
    train_data = StageManifestDataset(
        train_manifest, expected_stage=stage, expected_split="train"
    )
    smoke_overfit = bool(config.get("smoke_overfit", False))
    if smoke_overfit:
        validation_data = train_data
        validation_manifest = train_manifest
    else:
        validation_manifest = Path(config["val_manifest"]).resolve()
        validation_data = StageManifestDataset(
            validation_manifest, expected_stage=stage, expected_split="val"
        )
    provenance = {
        "train_manifest": str(train_manifest),
        "train_sha256": sha256(train_manifest),
        "validation_manifest": str(validation_manifest),
        "validation_sha256": sha256(validation_manifest),
        "smoke_overfit": smoke_overfit,
        "evaluation_policy": (
            "training-data overfit check; not a valid model estimate"
            if smoke_overfit else "official TUSZ dev split"
        ),
    }

    model, processor = load_qwen_vl_eeg(config, device)
    collator = QwenVLEEGCollator(processor, stage=stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters for this stage")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    step = 0
    if resume is not None:
        state_path = resume / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing resume state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["provenance"] != provenance:
            raise ValueError("Resume data provenance differs from the original run")
        old = state["config"]
        for key in (
            "stage", "max_steps", "batch_size", "gradient_accumulation_steps",
            "learning_rate", "caption_fraction",
        ):
            if old.get(key) != config.get(key):
                raise ValueError(f"Resume configuration changed: {key}")
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])

    maximum = int(config["max_steps"])
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    batch_size = int(config.get("batch_size", 1))
    if maximum <= 0 or accumulation <= 0 or batch_size <= 0 or step >= maximum:
        raise ValueError("Invalid or already-completed training schedule")
    caption_fraction = (
        float(config.get("caption_fraction", 0.25))
        if stage == "multimodal_sft" else 0.0
    )
    total_examples = maximum * accumulation * batch_size
    consumed_examples = step * accumulation * batch_size
    sampler = TaskRatioSampler(
        train_data,
        total_examples,
        caption_fraction=caption_fraction,
        seed=seed,
        start_index=consumed_examples,
    )
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    write_json(output / "config.json", config)
    write_json(output / "data_provenance.json", provenance)

    use_amp = device.type == "cuda" and config.get("dtype", "bfloat16") in {
        "bfloat16", "float16"
    }
    amp_dtype = getattr(torch, config.get("dtype", "bfloat16"))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader, start=step * accumulation):
        amp = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with amp:
            loss = model(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        (loss / accumulation).backward()
        if (batch_index + 1) % accumulation:
            continue
        torch.nn.utils.clip_grad_norm_(
            parameters, float(config.get("max_grad_norm", 1.0))
        )
        lr = scheduled_learning_rate(
            step,
            maximum,
            int(config.get("warmup_steps", 0)),
            float(config["learning_rate"]),
            bool(config.get("cosine_decay", False)),
            float(config.get("min_lr_ratio", 0.1)),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step == 1 or step % int(config.get("log_steps", 20)) == 0:
            print(
                json.dumps(
                    {"step": step, "loss": float(loss.detach()), "lr": lr},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if step % int(config.get("eval_steps", 500)) == 0 or step == maximum:
            validation_loss, validation_examples = _validation_loss(
                model, validation_data, collator, config, device
            )
            report = {
                "step": step,
                "validation_loss": validation_loss,
                "validation_examples": validation_examples,
                "smoke_overfit": smoke_overfit,
            }
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=False) + "\n")
            _save_checkpoint(
                output / "last", model, processor, optimizer, step, config, provenance
            )
            model.train()
        if step >= maximum:
            break

    _save_checkpoint(
        output / "final", model, processor, optimizer, step, config, provenance
    )
    write_json(output / "COMPLETED.json", {"steps": step, "stage": stage})
    return {"steps": step, "stage": stage, "output_dir": str(output.resolve())}

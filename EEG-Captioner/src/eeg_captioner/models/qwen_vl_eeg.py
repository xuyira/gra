"""Qwen2.5-VL adapter that injects projected EEG tokens alongside images."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from .checkpoint import load_encoder
from .projector import AtomicEEGProjector


EEG_PLACEHOLDER_TOKEN = "<|eeg_pad|>"
EEG_TOKEN_COUNT = 200
STAGE2_SYSTEM = (
    "You are an EEG signal analysis assistant. Answer the multiple-choice "
    "question using the supplied EEG embeddings. Return exactly the option "
    "letter and label, with no explanation."
)
STAGE3_SYSTEM = (
    "You are an EEG signal analysis assistant. Use both the supplied EEG "
    "embeddings and EEG waveform image. For a multiple-choice question, return "
    "exactly the option letter and label. For a description request, return "
    "only the requested EEG description without extra commentary."
)


def eeg_prompt(instruction: str, *, token_count: int = EEG_TOKEN_COUNT) -> str:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return (
        "A 10-second, 19-channel EEG segment sampled at 200 Hz is represented "
        "by learned EEG embeddings below.\n<eeg>\n"
        + EEG_PLACEHOLDER_TOKEN * token_count
        + "\n</eeg>\n"
        + str(instruction).strip()
    )


def build_chat_messages(example: dict, *, stage: str, include_target: bool) -> list:
    if stage not in {"eeg_projector_alignment", "multimodal_sft"}:
        raise ValueError(f"Unsupported stage: {stage}")
    system = STAGE2_SYSTEM if stage == "eeg_projector_alignment" else STAGE3_SYSTEM
    content = []
    if stage == "multimodal_sft":
        if not example.get("image_path"):
            raise ValueError("Stage 3 example is missing image_path")
        content.append({"type": "image", "image": example["image_path"]})
    content.append({"type": "text", "text": eeg_prompt(example["instruction"])})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]
    if include_target:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["target"]}],
            }
        )
    return messages


class QwenVLEEGCollator:
    """Apply the native Qwen-VL processor and mask prompt tokens for SFT."""

    def __init__(self, processor, *, stage: str):
        if stage not in {"eeg_projector_alignment", "multimodal_sft"}:
            raise ValueError(f"Unsupported stage: {stage}")
        self.processor = processor
        self.stage = stage

    def _template(self, messages, *, add_generation_prompt):
        apply = getattr(self.processor, "apply_chat_template", None)
        if apply is None:
            apply = self.processor.tokenizer.apply_chat_template
        return apply(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def __call__(self, examples):
        from PIL import Image

        if not examples:
            raise ValueError("Cannot collate an empty batch")
        full_texts = []
        prompt_texts = []
        images = []
        opened = []
        for example in examples:
            full_texts.append(
                self._template(
                    build_chat_messages(example, stage=self.stage, include_target=True),
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                self._template(
                    build_chat_messages(example, stage=self.stage, include_target=False),
                    add_generation_prompt=True,
                )
            )
            if self.stage == "multimodal_sft":
                with Image.open(example["image_path"]) as image:
                    loaded = image.convert("RGB").copy()
                images.append(loaded)
                opened.append(loaded)

        tokenizer = self.processor.tokenizer
        old_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "right"
            kwargs = {
                "padding": True,
                "return_tensors": "pt",
            }
            if images:
                kwargs["images"] = images
            full = self.processor(text=full_texts, **kwargs)
            prompt = self.processor(text=prompt_texts, **kwargs)
        finally:
            tokenizer.padding_side = old_padding_side
            for image in opened:
                image.close()

        labels = torch.full_like(full["input_ids"], -100)
        for index in range(len(examples)):
            full_ids = full["input_ids"][index][full["attention_mask"][index].bool()]
            prompt_ids = prompt["input_ids"][index][prompt["attention_mask"][index].bool()]
            prompt_length = len(prompt_ids)
            if len(full_ids) <= prompt_length or not torch.equal(
                full_ids[:prompt_length], prompt_ids
            ):
                raise ValueError(
                    "Qwen chat template prompt is not a prefix of the supervised example"
                )
            labels[index, prompt_length:len(full_ids)] = full_ids[prompt_length:]

        result = {key: value for key, value in full.items() if torch.is_tensor(value)}
        result["labels"] = labels
        result["x"] = torch.stack([example["x"] for example in examples])
        result["ids"] = [example["id"] for example in examples]
        result["task_types"] = [example["task_type"] for example in examples]
        return result


class EEGVisionLanguageModel(nn.Module):
    """Keep Qwen's native visual path while replacing EEG placeholder embeds."""

    def __init__(
        self,
        encoder,
        projector,
        qwen,
        *,
        eeg_token_id: int,
        qwen_trainable: bool,
        eeg_token_count: int = EEG_TOKEN_COUNT,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.qwen = qwen
        self.eeg_token_id = int(eeg_token_id)
        self.qwen_trainable = bool(qwen_trainable)
        self.eeg_token_count = int(eeg_token_count)
        self.encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if not self.qwen_trainable:
            self.qwen.eval()
        return self

    @contextmanager
    def _inject_eeg(self, input_ids, eeg_tokens):
        embedding = self.qwen.get_input_embeddings()
        mask = input_ids.eq(self.eeg_token_id)
        counts = mask.sum(dim=1)
        if not torch.all(counts.eq(self.eeg_token_count)):
            raise ValueError(
                f"Every example must contain {self.eeg_token_count} EEG tokens; "
                f"got {counts.tolist()}"
            )
        if tuple(eeg_tokens.shape[:2]) != (len(input_ids), self.eeg_token_count):
            raise ValueError(
                f"Projected EEG shape does not match placeholders: {tuple(eeg_tokens.shape)}"
            )

        def replace(_module, _args, output):
            if not torch.is_tensor(output) or output.ndim != 3:
                raise ValueError("Unexpected Qwen token embedding output")
            if output.shape[-1] != eeg_tokens.shape[-1]:
                raise ValueError("EEG projector/Qwen hidden-size mismatch")
            replaced = output.clone()
            replaced[mask] = eeg_tokens.reshape(-1, eeg_tokens.shape[-1]).to(
                device=output.device, dtype=output.dtype
            )
            return replaced

        handle = embedding.register_forward_hook(replace)
        try:
            yield
        finally:
            handle.remove()

    def forward(self, batch):
        device = next(self.projector.parameters()).device
        eeg = batch["x"].to(device, non_blocking=True)
        with torch.no_grad():
            features = self.encoder(eeg)
        eeg_tokens = self.projector(features)
        model_inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value) and key != "x"
        }
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("Collated batch is missing input_ids")
        with self._inject_eeg(input_ids, eeg_tokens):
            output = self.qwen(**model_inputs, use_cache=False)
        if output.loss is None:
            raise ValueError("Qwen did not return a supervised loss")
        return output.loss


def _load_projector(projector, checkpoint):
    if checkpoint is None:
        return
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "eeg_projector.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing EEG projector checkpoint: {path}")
    projector.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )


def load_qwen_vl_eeg(config, device):
    """Load real Qwen2.5-VL components lazily for Stage 2 or Stage 3."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    stage = config["stage"]
    if stage not in {"eeg_projector_alignment", "multimodal_sft"}:
        raise ValueError(f"Unsupported stage: {stage}")
    dtype_name = config.get("dtype", "bfloat16")
    if dtype_name not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    qwen_path = config["qwen_path"]
    local_only = bool(config.get("local_files_only", True))
    processor = AutoProcessor.from_pretrained(
        qwen_path, local_files_only=local_only
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [EEG_PLACEHOLDER_TOKEN]}
    )
    load_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": local_only,
    }
    if config.get("attn_implementation"):
        load_kwargs["attn_implementation"] = config["attn_implementation"]
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path, **load_kwargs
    )
    if added or len(tokenizer) != base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(tokenizer))
    base.requires_grad_(False)
    base.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    checkpoint = config.get("resume") or config.get("initial_checkpoint")
    qwen_trainable = stage == "multimodal_sft"
    if qwen_trainable:
        from peft import LoraConfig, PeftModel, get_peft_model

        adapter_path = Path(checkpoint) / "qwen_lora" if checkpoint else None
        if adapter_path is not None and adapter_path.is_dir():
            qwen = PeftModel.from_pretrained(
                base, str(adapter_path), is_trainable=True
            )
        else:
            qwen = get_peft_model(
                base,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=int(config.get("lora_rank", 8)),
                    lora_alpha=int(config.get("lora_alpha", 32)),
                    lora_dropout=float(config.get("lora_dropout", 0.05)),
                    target_modules=(
                        r".*language_model\.layers\.\d+\.self_attn\."
                        r"(q_proj|k_proj|v_proj|o_proj)$"
                    ),
                ),
            )
        unexpected = [
            name for name, parameter in qwen.named_parameters()
            if parameter.requires_grad
            and "lora_" not in name
            and "modules_to_save" not in name
        ]
        if unexpected:
            raise ValueError(
                "Stage 3 unexpectedly unfroze non-LoRA Qwen parameters: "
                f"{unexpected[:10]}"
            )
    else:
        qwen = base
        qwen.requires_grad_(False).eval()

    qwen.to(device)
    encoder = load_encoder(config["encoder_checkpoint"], device=device, frozen=True)
    projector = AtomicEEGProjector(base.config.text_config.hidden_size).to(device)
    _load_projector(projector, checkpoint)
    model = EEGVisionLanguageModel(
        encoder,
        projector,
        qwen,
        eeg_token_id=tokenizer.convert_tokens_to_ids(EEG_PLACEHOLDER_TOKEN),
        qwen_trainable=qwen_trainable,
    ).to(device)
    return model, processor

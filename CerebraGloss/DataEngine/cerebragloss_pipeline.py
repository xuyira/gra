"""Headless CerebraGloss inference for shared 19-channel EEG windows.

This command consumes the same stable 10-second H5 rows used by
EEG-Captioner and writes waveform detections, background features, artifact
detections, and the final rule-based CerebraGloss caption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from artifact.detect_artifact import EEGArtifact
from background.background_calculate_FFT import (
    extract_dominant_frequency_and_max_amplitude,
)
from CerebraGlossYOLO.model import CerebraGlossYOLO
from CerebraGlossYOLO.utils import batch_nms, decode_predictions_fpn
from textGenerator.textGenerator_English import text_generator

try:
    from eeg_captioner.data.segment_store import SharedSegment, load_shared_segment
except ImportError as exc:  # Give an actionable error instead of failing later.
    raise ImportError(
        "EEG-Captioner must be installed in the current environment. "
        "From gra/EEG-Captioner run: pip install -e ."
    ) from exc


SAMPLE_RATE = 200
SEQ_LEN = 2000
WINDOW_SECONDS = SEQ_LEN / SAMPLE_RATE
STANDARD_19_CHANNELS = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
)
WAVEFORM_CLASSES = (
    "sharp", "spike", "spsw", "spindle", "Kcomplex", "eyem", "eyer+",
    "eyer-", "hfnoise",
)
ANCHORS_PER_LEVEL = (
    (90 / SEQ_LEN, 300 / SEQ_LEN),
    None,
    None,
    (1900 / SEQ_LEN,),
)
ACTIVE_ANCHORS = tuple(level for level in ANCHORS_PER_LEVEL if level is not None)
NUM_ANCHORS_PER_LEVEL = tuple(
    len(level) if level is not None else None for level in ANCHORS_PER_LEVEL
)

def _artifact_channel_names(channels: Iterable[str]) -> list[str]:
    """Use the capitalization expected by CerebraGloss's artifact module."""
    names = []
    for channel in channels:
        base = channel.split("-")[0].upper()
        if base.startswith("FP"):
            names.append("Fp" + base[2:])
        elif base.endswith("Z"):
            names.append(base[:-1] + "z")
        else:
            names.append(base)
    return names


def _plain_bbox(box: dict) -> dict:
    """Convert NumPy scalar values in a detection to JSON-safe Python values."""
    return {
        "channel_idx": int(box["channel_idx"]),
        "wave_type": str(box["wave_type"]),
        "confidence": float(box.get("confidence", 1.0)),
        "time_range": [float(box["time_range"][0]), float(box["time_range"][1])],
    }


class WaveformDetector:
    """Load the CerebraGloss YOLO checkpoint once and reuse it for all rows."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CerebraGlossYOLO(
            num_classes=len(WAVEFORM_CLASSES),
            num_anchors_per_level=list(NUM_ANCHORS_PER_LEVEL),
        ).to(self.device)
        state = torch.load(
            Path(model_path), map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, signal_uv: np.ndarray, channels: Iterable[str]) -> list[dict]:
        signal_uv = np.asarray(signal_uv, dtype=np.float32)
        channels = tuple(channel.upper().split("-")[0] for channel in channels)
        if signal_uv.shape != (len(STANDARD_19_CHANNELS), SEQ_LEN):
            raise ValueError(f"Expected signal [19,2000], got {signal_uv.shape}")
        if channels != STANDARD_19_CHANNELS:
            raise ValueError(f"Expected fixed standard-19 order, got {channels}")

        signal = torch.from_numpy(signal_uv).unsqueeze(0)
        mean = signal.mean(dim=2, keepdim=True)
        std = signal.std(dim=2, keepdim=True)
        signal = ((signal - mean) / (std + 1e-6)).to(self.device)
        attention_mask = torch.zeros((1, 19), dtype=torch.bool, device=self.device)
        pos_indices = torch.arange(19, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(signal, attention_mask, pos_indices)
            decoded = decode_predictions_fpn(
                predictions,
                list(ACTIVE_ANCHORS),
                conf_threshold=0.5,
            )
            result = batch_nms(
                *decoded,
                nms_iou_threshold=0.5,
                batch_size=1,
            )[0]

        boxes = result["boxes"].detach().cpu()
        scores = result["scores"].detach().cpu()
        labels = result["labels"].detach().cpu()
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            start_norm, channel_idx, end_norm, _ = box.tolist()
            detections.append(
                {
                    "channel_idx": int(channel_idx),
                    "wave_type": WAVEFORM_CLASSES[int(label)],
                    "confidence": float(score),
                    "time_range": [
                        max(0.0, float(start_norm) * WINDOW_SECONDS),
                        min(WINDOW_SECONDS, float(end_norm) * WINDOW_SECONDS),
                    ],
                }
            )
        return detections


def render_cerebragloss_eeg(
    output_path: str | Path,
    *,
    segment: SharedSegment,
) -> None:
    """Render the fixed image style used by CerebraGlossBench training data."""
    signal = segment.signal_uv
    display_channel_width = 200
    offsets = np.arange(signal.shape[0], 0, -1) * display_channel_width
    display_channels = [f"{channel.split('-')[0]}-AV" for channel in segment.channels]

    fig, ax = plt.subplots(figsize=(6, signal.shape[0] * 0.4))
    for channel_idx in range(signal.shape[0]):
        ax.plot(
            np.arange(signal.shape[1]) / SAMPLE_RATE,
            -signal[channel_idx] + offsets[channel_idx],
            label=display_channels[channel_idx],
            color="black",
            linewidth=1,
        )

    bar_y_start = offsets[0] - display_channel_width / 2
    bar_y_end = offsets[0] + display_channel_width / 2
    ax.plot([0, 0], [bar_y_start, bar_y_end], color="red", linewidth=1, zorder=10)
    ax.text(
        -0.1,
        bar_y_start,
        f"{display_channel_width} µV",
        color="red",
        ha="right",
        va="center",
        fontsize=6,
    )

    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_xlim(0, WINDOW_SECONDS)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_yticks(offsets)
    ax.set_yticklabels(display_channels, fontsize=8)
    ax.set_ylim(
        display_channel_width // 2,
        signal.shape[0] * display_channel_width + display_channel_width // 2,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def process_segment(
    segment: SharedSegment,
    detector: WaveformDetector,
    *,
    output_dir: str | Path,
) -> dict:
    signal = segment.signal_uv

    waveform_bboxes = detector.predict(signal, segment.channels)
    dominant_freq_matrix, max_amp_matrix = (
        extract_dominant_frequency_and_max_amplitude(signal, SAMPLE_RATE)
    )
    artifact_model = EEGArtifact(
        data=signal,
        freq=SAMPLE_RATE,
        ch_order=_artifact_channel_names(segment.channels),
        verbose=False,
    )
    artifact_bboxes = [
        _plain_bbox(box)
        for box in artifact_model.get_annotations_dict().get("bboxes", [])
    ]
    waveform_bboxes = [_plain_bbox(box) for box in waveform_bboxes]
    caption = text_generator(
        waveform_bboxes,
        artifact_bboxes,
        dominant_freq_matrix,
        max_amp_matrix,
        list(segment.channels),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "segment_id": segment.segment_id,
        "row_index": segment.row_index,
        "channels": list(segment.channels),
        "sample_rate_hz": SAMPLE_RATE,
        "seizure_fraction": segment.seizure_fraction,
        "seizure_label": segment.seizure_label,
        "waveform_bboxes": waveform_bboxes,
        "artifact_bboxes": artifact_bboxes,
        "dominant_freq_matrix": dominant_freq_matrix.tolist(),
        "max_amp_matrix": max_amp_matrix.tolist(),
        "caption": caption,
    }

    json_path = output_dir / f"{segment.segment_id}.json"
    plot_path = output_dir / f"{segment.segment_id}.png"
    render_cerebragloss_eeg(
        plot_path,
        segment=segment,
    )

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _row_indices(h5_path: Path, args: argparse.Namespace) -> list[int]:
    if args.all:
        with h5py.File(h5_path, "r") as handle:
            return list(range(len(handle["segment_ids"])))
    if args.segment_id is not None:
        segment = load_shared_segment(h5_path, segment_id=args.segment_id)
        return [segment.row_index]
    return [0 if args.row_index is None else args.row_index]


def _build_parser() -> argparse.ArgumentParser:
    data_engine_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run YOLO, background, artifact, and rule-caption generation on shared EEG windows."
    )
    parser.add_argument("--h5", required=True, type=Path, help="Shared presegmented H5 file")
    parser.add_argument("--output-dir", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--row-index", type=int, help="Zero-based H5 row; default: 0")
    selection.add_argument("--segment-id", help="Stable segment ID")
    selection.add_argument("--all", action="store_true", help="Process every row in the H5 file")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.set_defaults(model_path=data_engine_dir / "checkpoints" / "model.pkl")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    detector = WaveformDetector(
        args.model_path,
        device=args.device,
    )
    rows = _row_indices(args.h5, args)
    for row_index in rows:
        segment = load_shared_segment(args.h5, row_index=row_index)
        result = process_segment(
            segment,
            detector,
            output_dir=args.output_dir,
        )
        print(
            f"[{row_index}] {segment.segment_id}: "
            f"YOLO={len(result['waveform_bboxes'])}, "
            f"artifacts={len(result['artifact_bboxes'])}"
        )
    print(f"Wrote {len(rows)} segment(s) to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

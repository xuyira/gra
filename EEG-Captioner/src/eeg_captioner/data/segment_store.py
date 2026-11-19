"""Reader for the shared standard-19, 10-second EEG segment store."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from eeg_captioner.channels import STANDARD_19_CHANNELS


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class SharedSegment:
    """One signal window and the metadata needed to join both feature routes."""

    signal_uv: np.ndarray
    channels: tuple[str, ...]
    segment_id: str
    parent_trial_id: str
    source_path: str
    row_index: int
    segment_index: int
    start_sample: int
    start_sec: float
    seizure_overlap_sec: float
    seizure_fraction: float
    seizure_label: int


def load_shared_segment(
    h5_path: str | Path,
    *,
    row_index: int | None = None,
    segment_id: str | None = None,
) -> SharedSegment:
    """Load exactly one row for CerebraGloss or other raw-signal consumers.

    Select by zero-based ``row_index`` or stable ``segment_id``. When neither
    is supplied, row zero is used. EEG-Captioner's model-facing loader consumes
    the same row and inserts an all-zero OZ channel only at its model boundary.
    """
    if row_index is not None and segment_id is not None:
        raise ValueError("Select a segment by row_index or segment_id, not both")

    with h5py.File(Path(h5_path), "r") as f:
        if not bool(f.attrs.get("presegmented_10s", False)):
            raise ValueError("Expected a presegmented_10s H5 store")
        if float(f.attrs.get("sample_rate", 0.0)) != 200.0:
            raise ValueError("Shared segments must be sampled at 200 Hz")
        channels = tuple(_decode(value).upper() for value in f["channels"][:])
        if channels != STANDARD_19_CHANNELS:
            raise ValueError(
                f"Expected fixed standard-19 order, got {channels}"
            )
        ids = [_decode(value) for value in f["segment_ids"][:]]
        if segment_id is not None:
            try:
                selected = ids.index(segment_id)
            except ValueError as exc:
                raise KeyError(f"Unknown segment_id: {segment_id}") from exc
        else:
            selected = 0 if row_index is None else int(row_index)
        if selected < 0 or selected >= len(ids):
            raise IndexError(f"Segment row out of range: {selected}")

        signal = np.asarray(f["data"][selected], dtype=np.float32)
        if signal.shape != (len(STANDARD_19_CHANNELS), 2000):
            raise ValueError(f"Expected segment [19,2000], got {signal.shape}")
        if not np.isfinite(signal).all():
            raise ValueError("EEG segment contains NaN or infinity")

        return SharedSegment(
            signal_uv=signal,
            channels=channels,
            segment_id=ids[selected],
            parent_trial_id=_decode(f["parent_trial_ids"][selected]),
            source_path=_decode(f["source_paths"][selected]),
            row_index=selected,
            segment_index=int(f["segment_indices"][selected]),
            start_sample=int(f["segment_start_samples"][selected]),
            start_sec=float(f["segment_start_sec"][selected]),
            seizure_overlap_sec=float(f["seizure_overlap_sec"][selected]),
            seizure_fraction=float(f["seizure_fraction"][selected]),
            seizure_label=int(f["seizure_label"][selected]),
        )

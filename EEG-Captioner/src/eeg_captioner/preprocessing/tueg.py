"""TUEG EDF to subject-H5 signal processing, extracted from the research pipeline."""
import csv
import glob
import json
import os
import traceback
from pathlib import Path
import h5py
import mne
import numpy as np

from eeg_captioner.channels import STANDARD_19_CHANNELS

L_FREQ, H_FREQ, NOTCH_FREQ, TARGET_SRATE = 1.6, 70.0, 60.0, 200.0
REFERENCE_SCHEME = "average"
GZIP_LEVEL = 4
SEGMENT_DURATION_SEC = 10.0
SEGMENT_SAMPLES = int(TARGET_SRATE * SEGMENT_DURATION_SEC)
SEIZURE_FRACTION_THRESHOLD = 0.15

STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
]

STANDARD_SET = {ch.upper() for ch in STANDARD_1020}

STANDARD_BY_UPPER = {ch.upper(): ch for ch in STANDARD_1020}

def canonicalize_channel_name(name: str) -> str:
    text = str(name or "").strip().upper().replace(" ", "")
    if text.startswith("EEG"):
        text = text[3:]
    for suffix in ("-REF", "-LE", "-RE", "-AVG", "-AR"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return STANDARD_BY_UPPER.get(text, text)

def get_eeg_channel_names(raw):
    return [ch for ch in raw.ch_names if ch.upper().startswith("EEG ")]

def compute_labram_mapping(channels: list[str]):
    in_labram = np.array([ch.upper() in STANDARD_SET for ch in channels], dtype=bool)
    labram_indices = np.where(in_labram)[0].astype(np.int32)
    labram_names = [channels[i] for i in labram_indices]
    return in_labram, labram_indices, labram_names

def _relative_source_path(path: str) -> str:
    return path.split("/edf/", 1)[1] if "/edf/" in path else os.path.basename(path)

def merge_intervals(intervals: list[tuple[float, float]]) -> np.ndarray:
    """Return sorted, non-overlapping intervals with touching ranges merged."""
    if not intervals:
        return np.empty((0, 2), dtype=np.float32)
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if not np.isfinite(start) or not np.isfinite(stop) or start < 0 or stop <= start:
            raise ValueError(f"Invalid seizure interval [{start}, {stop}]")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return np.asarray(merged, dtype=np.float32)

def read_tusz_seizure_annotation(edf_path: str) -> dict:
    """Read the sibling TUSZ ``.csv_bi`` file for one EDF recording.

    A valid annotation containing only ``bckg`` rows is still available and
    represents a recording with no seizure intervals. Every EDF is required to
    have an annotation because this preprocessing pipeline produces binary
    TUSZ seizure labels only.
    """
    annotation_path = str(Path(edf_path).with_suffix(".csv_bi"))
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(
            f"Missing required TUSZ annotation for {edf_path}: {annotation_path}"
        )

    intervals: list[tuple[float, float]] = []
    with open(annotation_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = (line for line in handle if not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows)
        required = {"start_time", "stop_time", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Malformed TUSZ annotation {annotation_path}: expected columns {sorted(required)}"
            )
        for row_number, row in enumerate(reader, start=2):
            label = str(row.get("label") or "").strip().lower()
            if label != "seiz":
                continue
            try:
                start = float(row["start_time"])
                stop = float(row["stop_time"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed seizure interval at {annotation_path}:{row_number}"
                ) from exc
            intervals.append((start, stop))

    return {
        "intervals_sec": merge_intervals(intervals),
        "annotation_path": _relative_source_path(annotation_path),
    }


def process_single_edf(edf_path: str, channel_order: list[str]) -> dict:
    """处理单个 EDF 文件，返回处理后的数据和元信息。"""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    eeg_ch_names = get_eeg_channel_names(raw)
    if not eeg_ch_names:
        return {"error": "no EEG channels found"}

    raw.pick(eeg_ch_names)

    original_sfreq = raw.info["sfreq"]

    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

    if raw.info["sfreq"] != TARGET_SRATE:
        raw.resample(TARGET_SRATE, verbose=False)

    # Select the fixed standard-19 montage before re-referencing. Averaging all
    # EDF EEG rows would incorrectly include auxiliary/reference channels such
    # as A1/A2, T1/T2, or OZ when they are present.
    data_volts = raw.get_data()
    raw_canonical = {canonicalize_channel_name(ch): i for i, ch in enumerate(raw.ch_names)}
    missing_channels = [name for name in channel_order if name not in raw_canonical]
    if missing_channels:
        raise ValueError(
            f"Missing required standard-19 channels in {edf_path}: {missing_channels}"
        )
    source_indices = [raw_canonical[ch_name] for ch_name in channel_order]
    referenced_volts = data_volts[source_indices]
    referenced_volts -= referenced_volts.mean(axis=0, keepdims=True)

    # MNE stores volts internally; the shared H5 contract is microvolts.
    reordered = (referenced_volts * 1e6).astype(np.float32, copy=False)

    n_samples = reordered.shape[1]
    rel_path = _relative_source_path(edf_path)
    trial_id = Path(edf_path).stem

    return {
        "data": reordered,
        "n_samples": n_samples,
        "original_sfreq": float(original_sfreq),
        "source_path": rel_path,
        "trial_id": trial_id,
        "error": None,
    }


def _segment_annotation(
    intervals_sec: np.ndarray,
    start_sec: float,
    duration_sec: float = SEGMENT_DURATION_SEC,
) -> tuple[np.ndarray, float, float, int]:
    """Clip absolute seizure intervals to one window and make them window-local."""
    stop_sec = start_sec + duration_sec
    local_intervals = []
    for interval_start, interval_stop in np.asarray(intervals_sec).reshape(-1, 2):
        overlap_start = max(start_sec, float(interval_start))
        overlap_stop = min(stop_sec, float(interval_stop))
        if overlap_stop > overlap_start:
            local_intervals.append(
                (overlap_start - start_sec, overlap_stop - start_sec)
            )
    local = merge_intervals(local_intervals)
    overlap_sec = float(np.sum(local[:, 1] - local[:, 0])) if len(local) else 0.0
    fraction = overlap_sec / duration_sec
    label = int(fraction >= SEIZURE_FRACTION_THRESHOLD)
    return local, overlap_sec, fraction, label


def process_group(
    edf_files: list[str],
    output_path: str,
    subject_id: str,
    channel_order: list[str],
) -> dict:
    """Write the shared fixed-standard-19, non-overlapping 10-second store.

    Each H5 row is the authoritative signal window consumed by both
    EEG-Captioner and CerebraGloss. Incomplete tails are recorded and dropped,
    never padded into training samples. Two passes keep peak memory bounded to
    one EDF recording.
    """
    if tuple(channel_order) != STANDARD_19_CHANNELS:
        raise ValueError(
            "Preprocessing channel_order must be the fixed CerebraGloss standard-19 order"
        )

    errors = []
    valid_files = []
    source_records = []
    dropped_tail_samples = []

    for edf_path in sorted(edf_files):
        # Annotation errors intentionally remain fatal: an unlabelled recording
        # must never silently turn into a background recording.
        annotation = read_tusz_seizure_annotation(edf_path)
        try:
            result = process_single_edf(edf_path, channel_order)
            if result["error"]:
                errors.append({"file": edf_path, "error": result["error"]})
                continue
            n_segments = result["n_samples"] // SEGMENT_SAMPLES
            if n_segments == 0:
                errors.append(
                    {
                        "file": edf_path,
                        "error": (
                            f"recording has {result['n_samples']} samples after resampling; "
                            f"at least {SEGMENT_SAMPLES} are required"
                        ),
                    }
                )
                continue
            valid_files.append(edf_path)
            source_records.append(
                {
                    "n_segments": n_segments,
                    "original_sfreq": result["original_sfreq"],
                    "source_path": result["source_path"],
                    "trial_id": result["trial_id"],
                    "seizure_intervals": annotation["intervals_sec"],
                    "seizure_annotation_path": annotation["annotation_path"],
                }
            )
            dropped_tail_samples.append(
                result["n_samples"] - n_segments * SEGMENT_SAMPLES
            )
        except Exception as exc:
            errors.append(
                {
                    "file": edf_path,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    if not valid_files:
        return {"status": "empty", "h5_path": None, "errors": errors}

    n_source_trials = len(valid_files)
    n_segments = sum(record["n_segments"] for record in source_records)
    n_channels = len(channel_order)
    in_labram, labram_indices, labram_names = compute_labram_mapping(channel_order)

    segment_ids = []
    parent_trial_ids = []
    parent_trial_indices = []
    segment_indices = []
    segment_start_samples = []
    segment_start_sec = []
    source_paths = []
    original_sfreqs = []
    seizure_annotation_paths = []
    local_seizure_intervals = []
    seizure_overlap_sec = []
    seizure_fraction = []
    seizure_label = []
    for parent_index, record in enumerate(source_records):
        for segment_index in range(record["n_segments"]):
            start_sample = segment_index * SEGMENT_SAMPLES
            start_sec = start_sample / TARGET_SRATE
            local, overlap, fraction, label = _segment_annotation(
                record["seizure_intervals"], start_sec
            )
            segment_ids.append(f"{record['trial_id']}__seg{segment_index:04d}")
            parent_trial_ids.append(record["trial_id"])
            parent_trial_indices.append(parent_index)
            segment_indices.append(segment_index)
            segment_start_samples.append(start_sample)
            segment_start_sec.append(start_sec)
            source_paths.append(record["source_path"])
            original_sfreqs.append(record["original_sfreq"])
            seizure_annotation_paths.append(record["seizure_annotation_path"])
            local_seizure_intervals.append(local)
            seizure_overlap_sec.append(overlap)
            seizure_fraction.append(fraction)
            seizure_label.append(label)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        data = f.create_dataset(
            "data",
            shape=(n_segments, n_channels, SEGMENT_SAMPLES),
            dtype="float32",
            chunks=(1, n_channels, SEGMENT_SAMPLES),
            compression="gzip",
            compression_opts=GZIP_LEVEL,
        )

        write_index = 0
        for edf_path, record in zip(valid_files, source_records):
            try:
                result = process_single_edf(edf_path, channel_order)
                if result["error"]:
                    raise RuntimeError(result["error"])
                trial_data = result["data"]
                for segment_index in range(record["n_segments"]):
                    start = segment_index * SEGMENT_SAMPLES
                    data[write_index] = trial_data[:, start : start + SEGMENT_SAMPLES]
                    write_index += 1
                del trial_data, result
            except Exception as exc:
                errors.append({"file": edf_path, "error": f"write_pass: {exc}"})

        f.create_dataset(
            "trial_lengths_in_samples",
            data=np.full(n_segments, SEGMENT_SAMPLES, dtype=np.int32),
        )
        f.create_dataset("channels", data=np.asarray(channel_order, dtype="S"))
        f.create_dataset("channels_in_labram", data=in_labram)
        f.create_dataset("labram_channel_indices", data=labram_indices)
        f.create_dataset("labram_channel_names", data=np.asarray(labram_names, dtype="S"))
        f.create_dataset(
            "original_sample_rates_hz", data=np.asarray(original_sfreqs, dtype=np.float32)
        )
        f.create_dataset("source_paths", data=np.asarray(source_paths, dtype="S"))
        # Compatibility: legacy readers call each row a trial.
        f.create_dataset("trial_ids", data=np.asarray(segment_ids, dtype="S"))
        f.create_dataset("segment_ids", data=np.asarray(segment_ids, dtype="S"))
        f.create_dataset("parent_trial_ids", data=np.asarray(parent_trial_ids, dtype="S"))
        f.create_dataset(
            "parent_trial_indices", data=np.asarray(parent_trial_indices, dtype=np.int32)
        )
        f.create_dataset("segment_indices", data=np.asarray(segment_indices, dtype=np.int32))
        f.create_dataset(
            "segment_start_samples", data=np.asarray(segment_start_samples, dtype=np.int64)
        )
        f.create_dataset(
            "segment_start_sec", data=np.asarray(segment_start_sec, dtype=np.float32)
        )
        f.create_dataset(
            "seizure_overlap_sec", data=np.asarray(seizure_overlap_sec, dtype=np.float32)
        )
        f.create_dataset(
            "seizure_fraction", data=np.asarray(seizure_fraction, dtype=np.float32)
        )
        f.create_dataset("seizure_label", data=np.asarray(seizure_label, dtype=np.int8))

        interval_counts = np.asarray(
            [len(intervals) for intervals in local_seizure_intervals], dtype=np.int32
        )
        max_intervals = int(interval_counts.max(initial=0))
        padded_intervals = np.full(
            (n_segments, max_intervals, 2), np.nan, dtype=np.float32
        )
        for row_index, intervals in enumerate(local_seizure_intervals):
            padded_intervals[row_index, : len(intervals)] = intervals
        f.create_dataset("seizure_interval_count", data=interval_counts)
        f.create_dataset("seizure_intervals_sec", data=padded_intervals)
        f.create_dataset(
            "seizure_annotation_paths",
            data=np.asarray(seizure_annotation_paths, dtype="S"),
        )

        channel_signature = "|".join(channel_order)
        preprocess_log = json.dumps(
            {
                "bandpass_hz": [L_FREQ, H_FREQ],
                "notch_hz": [NOTCH_FREQ],
                "resample_to_hz": TARGET_SRATE,
                "source_dataset": "TUEG_v2.0.2",
                "channel_policy": "fixed standard-19",
                "reference_scheme": REFERENCE_SCHEME,
                "reference_channels": list(channel_order),
                "segment_duration_sec": SEGMENT_DURATION_SEC,
                "segment_tail_policy": "drop incomplete tail",
            },
            ensure_ascii=False,
        )

        f.attrs["sample_rate"] = TARGET_SRATE
        f.attrs["trial_count"] = n_segments
        f.attrs["source_trial_count"] = n_source_trials
        f.attrs["segment_count"] = n_segments
        f.attrs["segment_duration_sec"] = SEGMENT_DURATION_SEC
        f.attrs["segment_samples"] = SEGMENT_SAMPLES
        f.attrs["presegmented_10s"] = True
        f.attrs["data_layout"] = "segments_channels_samples"
        f.attrs["incomplete_tail_policy"] = "drop"
        f.attrs["dropped_tail_samples_total"] = int(sum(dropped_tail_samples))
        f.attrs["has_labels"] = True
        f.attrs["all_trials_have_labels"] = True
        f.attrs["seizure_label_source"] = "TUSZ csv_bi"
        f.attrs["seizure_fraction_threshold"] = SEIZURE_FRACTION_THRESHOLD
        f.attrs["subject_id_original"] = subject_id
        f.attrs["channel_signature"] = channel_signature
        f.attrs["reference_scheme"] = REFERENCE_SCHEME
        f.attrs["reference_channel_count"] = n_channels
        f.attrs["reference_channel_signature"] = channel_signature
        f.attrs["preprocess_log"] = preprocess_log

    return {
        "status": "ok",
        "h5_path": output_path,
        "n_trials": n_segments,
        "n_source_trials": n_source_trials,
        "n_segments": n_segments,
        "n_channels": n_channels,
        "total_duration_sec": n_segments * SEGMENT_DURATION_SEC,
        "dropped_tail_samples": int(sum(dropped_tail_samples)),
        "errors": errors,
    }


def discover_subjects(edf_root: str) -> list[str]:
    """发现所有被试目录。结构: edf/{partition}/{subject_id}/"""
    subjects = []
    partitions = sorted(glob.glob(os.path.join(edf_root, "*")))
    for partition in partitions:
        if not os.path.isdir(partition):
            continue
        for subject_dir in sorted(glob.glob(os.path.join(partition, "*"))):
            if os.path.isdir(subject_dir):
                subjects.append(subject_dir)
    return subjects

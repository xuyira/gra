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
L_FREQ, H_FREQ, NOTCH_FREQ, TARGET_SRATE = 0.1, 75.0, 50.0, 200.0
CHUNK_TIME, GZIP_LEVEL = 2048, 4

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
    represents a recording with no seizure intervals. Missing annotations are
    retained as unknown so generic TUEG inputs remain supported.
    """
    annotation_path = str(Path(edf_path).with_suffix(".csv_bi"))
    if not os.path.isfile(annotation_path):
        return {
            "available": False,
            "intervals_sec": np.empty((0, 2), dtype=np.float32),
            "annotation_path": "",
        }

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
        "available": True,
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

    # MNE 内部单位是伏特，乘 1e6 得到 µV
    data = raw.get_data() * 1e6  # shape: (n_channels, n_samples)

    # 按 channel_order 重排通道
    raw_canonical = {canonicalize_channel_name(ch): i for i, ch in enumerate(raw.ch_names)}
    reordered = np.zeros((len(channel_order), data.shape[1]), dtype=np.float32)
    for target_idx, ch_name in enumerate(channel_order):
        source_idx = raw_canonical.get(ch_name)
        if source_idx is not None:
            reordered[target_idx] = data[source_idx]

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

def process_group(edf_files: list[str], output_path: str, subject_id: str, channel_order: list[str]) -> dict:
    """
    处理同一通道配置的所有 EDF，输出为一个 H5 文件。
    两遍处理：第一遍收集元信息和长度，第二遍逐 trial 写入 H5，避免内存爆炸。
    """
    sorted_files = sorted(edf_files)
    errors = []

    # === 第一遍：只收集元信息（不保留 data 在内存中） ===
    valid_files = []
    trial_lengths = []
    original_sfreqs = []
    source_paths = []
    trial_ids = []
    seizure_intervals = []
    seizure_annotation_available = []
    seizure_annotation_paths = []

    for edf_path in sorted_files:
        try:
            result = process_single_edf(edf_path, channel_order)
            if result["error"]:
                errors.append({"file": edf_path, "error": result["error"]})
                continue
            annotation = read_tusz_seizure_annotation(edf_path)
            valid_files.append(edf_path)
            trial_lengths.append(result["n_samples"])
            original_sfreqs.append(result["original_sfreq"])
            source_paths.append(result["source_path"])
            trial_ids.append(result["trial_id"])
            seizure_intervals.append(annotation["intervals_sec"])
            seizure_annotation_available.append(annotation["available"])
            seizure_annotation_paths.append(annotation["annotation_path"])
        except Exception as e:
            errors.append({"file": edf_path, "error": str(e), "traceback": traceback.format_exc()})

    if not valid_files:
        return {"status": "empty", "h5_path": None, "errors": errors}

    n_trials = len(valid_files)
    n_channels = len(channel_order)
    max_samples = max(trial_lengths)

    # 计算 LaBraM 映射
    in_labram, labram_indices, labram_names = compute_labram_mapping(channel_order)

    # === 写入 H5：逐 trial 写入，不构建完整 3D 数组 ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunk_shape = (1, n_channels, min(CHUNK_TIME, max_samples))

    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset("data", shape=(n_trials, n_channels, max_samples),
                                dtype="float32", chunks=chunk_shape,
                                compression="gzip", compression_opts=GZIP_LEVEL,
                                fillvalue=0.0)

        # 逐 trial 重新处理并写入（每次只有 1 个 EDF 在内存中）
        write_idx = 0
        for edf_path in valid_files:
            try:
                result = process_single_edf(edf_path, channel_order)
                if result["error"]:
                    raise RuntimeError(result["error"])
                trial_data = result["data"]  # (n_channels, n_samples)
                dset[write_idx, :, :trial_data.shape[1]] = trial_data
                del trial_data, result
                write_idx += 1
            except Exception as e:
                errors.append({"file": edf_path, "error": f"write_pass: {e}"})
                write_idx += 1

        f.create_dataset("trial_lengths_in_samples", data=np.array(trial_lengths, dtype=np.int32))
        f.create_dataset("channels", data=np.array(channel_order, dtype="S"))
        f.create_dataset("channels_in_labram", data=in_labram)
        f.create_dataset("labram_channel_indices", data=labram_indices)
        f.create_dataset("labram_channel_names", data=np.array(labram_names, dtype="S"))
        f.create_dataset("original_sample_rates_hz", data=np.array(original_sfreqs, dtype=np.float32))
        f.create_dataset("source_paths", data=np.array(source_paths, dtype="S"))
        f.create_dataset("trial_ids", data=np.array(trial_ids, dtype="S"))
        interval_counts = np.asarray(
            [len(intervals) for intervals in seizure_intervals], dtype=np.int32
        )
        max_intervals = int(interval_counts.max(initial=0))
        padded_intervals = np.full(
            (n_trials, max_intervals, 2), np.nan, dtype=np.float32
        )
        for trial_index, intervals in enumerate(seizure_intervals):
            padded_intervals[trial_index, : len(intervals)] = intervals
        f.create_dataset("seizure_interval_count", data=interval_counts)
        f.create_dataset("seizure_intervals_sec", data=padded_intervals)
        f.create_dataset(
            "seizure_annotation_available",
            data=np.asarray(seizure_annotation_available, dtype=np.bool_),
        )
        f.create_dataset(
            "seizure_annotation_paths",
            data=np.asarray(seizure_annotation_paths, dtype="S"),
        )

        channel_signature = "|".join(channel_order)
        preprocess_log = json.dumps({
            "bandpass_hz": [L_FREQ, H_FREQ],
            "notch_hz": [NOTCH_FREQ],
            "resample_to_hz": TARGET_SRATE,
            "source_dataset": "TUEG_v2.0.2",
        }, ensure_ascii=False)

        f.attrs["sample_rate"] = TARGET_SRATE
        f.attrs["trial_count"] = n_trials
        f.attrs["has_labels"] = bool(any(seizure_annotation_available))
        f.attrs["all_trials_have_labels"] = bool(all(seizure_annotation_available))
        f.attrs["seizure_label_source"] = "TUSZ csv_bi"
        f.attrs["subject_id_original"] = subject_id
        f.attrs["channel_signature"] = channel_signature
        f.attrs["preprocess_log"] = preprocess_log

    total_duration = sum(trial_lengths) / TARGET_SRATE
    return {
        "status": "ok",
        "h5_path": output_path,
        "n_trials": n_trials,
        "n_channels": n_channels,
        "total_duration_sec": total_duration,
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

"""Fit the four rule-QA thresholds once on TUSZ training features."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from eeg_captioner.io import write_json
from .calibration import default_calibration_path, load_rule_calibration
from .rules import ROI_CHANNELS


QUANTILE = 0.33
TEMPORAL_CONTEXTS = {
    "alpha": ("parietal",),
    "gamma": ("temporal", "frontal", "central", "parietal"),
}
LATERAL_CONTEXTS = {
    "delta": (
        ("last", "F3-F4"),
        ("middle", "F3-F4"),
        ("early", "F3-F4"),
        ("middle", "C3-C4"),
    ),
    "gamma": (("early", "F3-F4"),),
}


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_attr(handle: h5py.File, name: str):
    if name not in handle.attrs:
        raise ValueError(f"Missing H5 attribute {name}: {handle.filename}")
    value = handle.attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _assert_close(actual, expected, label: str) -> None:
    if not np.allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(f"Calibration input mismatch for {label}: {actual} != {expected}")


def _validate_processed_h5(path: Path, contract: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed H5 referenced by features: {path}")
    with h5py.File(path, "r") as handle:
        reference = _decode(handle.attrs.get("reference_scheme", ""))
        if reference != str(contract["reference_scheme"]):
            raise ValueError(f"Expected average-reference input, got {reference}: {path}")
        _assert_close(
            [float(handle.attrs.get("sample_rate", math.nan))],
            [float(contract["sample_rate_hz"])],
            "sample_rate_hz",
        )
        if not bool(handle.attrs.get("presegmented_10s", False)):
            raise ValueError(f"Expected presegmented_10s input: {path}")
        _assert_close(
            [float(handle.attrs.get("segment_duration_sec", math.nan))],
            [float(contract["segment_duration_sec"])],
            "segment_duration_sec",
        )
        log = json.loads(_decode(handle.attrs.get("preprocess_log", "{}")))
        _assert_close(log.get("bandpass_hz"), contract["bandpass_hz"], "bandpass_hz")
        _assert_close(log.get("notch_hz"), contract["notch_hz"], "notch_hz")
        return {
            "source_file": path.name,
            "subject_id": _decode(handle.attrs.get("subject_id_original", path.stem)),
            "segments": int(handle["data"].shape[0]),
            "channel_signature": _decode(handle.attrs.get("channel_signature", "")),
        }


def _eligible_trials(
    bad_channel_mask: np.ndarray,
    channels: list[str],
    required: Iterable[str],
) -> np.ndarray:
    channel_index = {name.upper(): index for index, name in enumerate(channels)}
    missing = [name for name in required if name.upper() not in channel_index]
    if missing:
        raise ValueError(f"Feature H5 is missing required channels: {missing}")
    indices = [channel_index[name.upper()] for name in required]
    return ~bad_channel_mask[:, indices].any(axis=1)


def _weighted_patient_quantile(patient_values: dict[str, np.ndarray], quantile: float) -> float:
    arrays = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in patient_values.values()
        if np.asarray(values).size
    ]
    arrays = [values[np.isfinite(values)] for values in arrays]
    arrays = [values for values in arrays if values.size]
    if not arrays:
        raise ValueError("No finite training values are available for threshold fitting")
    values = np.concatenate(arrays)
    weights = np.concatenate(
        [np.full(values_i.size, 1.0 / values_i.size, dtype=np.float64) for values_i in arrays]
    )
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    target = float(quantile) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    result = float(values[index])
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Fitted threshold must be positive and finite, got {result}")
    return result


def _collect_feature_file(path: Path, processed_root: Path, contract: dict) -> tuple[dict, dict]:
    patient_values: dict[str, list[np.ndarray]] = defaultdict(list)
    with h5py.File(path, "r") as handle:
        required_datasets = (
            "spatial/roi_rel_power",
            "spatial/lateralization_index",
            "meta/channels",
            "meta/bad_channel_mask",
            "meta/segment_valid_mask",
            "meta/bad_segment_mask",
        )
        missing = [name for name in required_datasets if name not in handle]
        if missing:
            raise ValueError(f"Missing calibration datasets in {path}: {missing}")

        source_file = _decode(handle.attrs.get("source_file", path.name))
        processed_record = _validate_processed_h5(processed_root / source_file, contract)
        _assert_close(
            _json_attr(handle, "welch_params_json")["analysis_range"],
            contract["feature_analysis_range_hz"],
            "feature_analysis_range_hz",
        )
        feature_bands = _json_attr(handle, "bands_json")
        if set(feature_bands) != set(contract["bands"]):
            raise ValueError(f"Feature band names do not match calibration contract: {path}")
        for name, expected in contract["bands"].items():
            _assert_close(feature_bands[name], expected, f"bands.{name}")
        _assert_close(
            [float(handle.attrs.get("sample_rate_hz", math.nan))],
            [float(contract["sample_rate_hz"])],
            "feature_sample_rate_hz",
        )

        channels = [_decode(value).upper() for value in handle["meta/channels"][:]]
        periods = [_decode(value).lower() for value in handle["meta/segment_part_names"][:]]
        rois = [str(value).lower() for value in _json_attr(handle, "spatial_roi_names_json")]
        pairs = [str(value).upper() for value in _json_attr(handle, "spatial_pair_names_json")]
        bands = [str(value).lower() for value in _json_attr(handle, "spatial_band_names_json")]
        if periods != ["early", "middle", "last"]:
            raise ValueError(f"Expected early/middle/last feature parts: {path}")

        rel_power = np.asarray(handle["spatial/roi_rel_power"][:], dtype=np.float64)
        lateralization = np.asarray(
            handle["spatial/lateralization_index"][:], dtype=np.float64
        )
        if rel_power.ndim != 5 or lateralization.ndim != 5:
            raise ValueError(f"Unexpected spatial feature shape in {path}")
        bad_channels = np.asarray(handle["meta/bad_channel_mask"][:], dtype=bool)
        valid_segments = np.asarray(handle["meta/segment_valid_mask"][:], dtype=bool)
        bad_segments = np.asarray(handle["meta/bad_segment_mask"][:], dtype=bool)
        valid_segments &= ~bad_segments
        if bad_channels.shape != (rel_power.shape[0], len(channels)):
            raise ValueError(f"Bad-channel mask shape does not match features: {path}")
        if valid_segments.shape != rel_power.shape[:2]:
            raise ValueError(f"Segment-valid mask shape does not match features: {path}")

        roi_index = {name: index for index, name in enumerate(rois)}
        pair_index = {name: index for index, name in enumerate(pairs)}
        band_index = {name: index for index, name in enumerate(bands)}
        period_index = {name: index for index, name in enumerate(periods)}

        for band, roi_names in TEMPORAL_CONTEXTS.items():
            for roi in roi_names:
                values = rel_power[:, :, :, roi_index[roi], band_index[band]]
                trial_ok = _eligible_trials(bad_channels, channels, ROI_CHANNELS[roi])
                mask = valid_segments & trial_ok[:, None] & np.isfinite(values).all(axis=2)
                changes = np.abs(np.diff(values, axis=2)[mask]).reshape(-1)
                if changes.size:
                    patient_values[f"temporal_pattern.{band}"].append(changes)

        for band, contexts in LATERAL_CONTEXTS.items():
            for period, pair in contexts:
                left, right = pair.split("-")
                values = lateralization[
                    :, :, period_index[period], pair_index[pair], band_index[band]
                ]
                trial_ok = _eligible_trials(bad_channels, channels, (left, right))
                mask = valid_segments & trial_ok[:, None] & np.isfinite(values)
                selected = np.abs(values[mask]).reshape(-1)
                if selected.size:
                    patient_values[f"lateral_dominance.{band}"].append(selected)

        patient_arrays = {
            key: np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
            for key, parts in patient_values.items()
        }
        record = {
            **processed_record,
            "feature_file": path.name,
            "valid_segments": int(valid_segments.sum()),
            "values": {key: int(value.size) for key, value in patient_arrays.items()},
        }
        return patient_arrays, record


def fit_rule_calibration(
    features_root: str | Path,
    processed_root: str | Path,
    output: str | Path,
    *,
    base_calibration: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    features_root = Path(features_root)
    processed_root = Path(processed_root)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite calibration: {output}")
    base_path = Path(base_calibration) if base_calibration else default_calibration_path()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_loaded = load_rule_calibration(base_path)
    contract = dict(base_loaded.data_contract)

    feature_files = sorted(features_root.glob("*.h5"))
    if not feature_files:
        raise ValueError(f"No feature H5 files found in {features_root}")

    values_by_key: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    input_records = []
    for feature_path in feature_files:
        patient_arrays, record = _collect_feature_file(
            feature_path, processed_root, contract
        )
        patient_id = str(record["subject_id"])
        for key, values in patient_arrays.items():
            existing = values_by_key[key].get(patient_id)
            values_by_key[key][patient_id] = (
                values if existing is None else np.concatenate((existing, values))
            )
        input_records.append(record)

    expected_keys = {
        "temporal_pattern.alpha",
        "temporal_pattern.gamma",
        "lateral_dominance.delta",
        "lateral_dominance.gamma",
    }
    if set(values_by_key) != expected_keys:
        missing = sorted(expected_keys.difference(values_by_key))
        extra = sorted(set(values_by_key).difference(expected_keys))
        raise ValueError(f"Calibration values incomplete; missing={missing}, extra={extra}")

    fitted = {
        key: _weighted_patient_quantile(patient_values, QUANTILE)
        for key, patient_values in sorted(values_by_key.items())
    }
    signature_text = json.dumps(input_records, sort_keys=True, separators=(",", ":"))
    input_signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()
    result = deepcopy(base)
    result["rules"]["temporal_pattern"] = {
        band: fitted[f"temporal_pattern.{band}"] for band in ("alpha", "gamma")
    }
    result["rules"]["lateral_dominance"] = {
        band: fitted[f"lateral_dominance.{band}"] for band in ("delta", "gamma")
    }
    value_signature = hashlib.sha256(
        json.dumps(result["rules"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    result["calibration_id"] = f"tusz_train_avgref_1p6_70_{value_signature}"
    result["fit"] = {
        "status": "fitted",
        "dataset": "TUSZ",
        "split": "train",
        "method": "absolute patient-equal weighted quantile",
        "weighting": "equal_patient",
        "quantile": QUANTILE,
        "feature_files": len(feature_files),
        "patients": len({str(record["subject_id"]) for record in input_records}),
        "segments": int(sum(record["segments"] for record in input_records)),
        "valid_segments": int(sum(record["valid_segments"] for record in input_records)),
        "valid_counts": {
            key: int(sum(values.size for values in patient_values.values()))
            for key, patient_values in sorted(values_by_key.items())
        },
        "patients_with_values": {
            key: int(sum(values.size > 0 for values in patient_values.values()))
            for key, patient_values in sorted(values_by_key.items())
        },
        "input_signature_sha256": input_signature,
        "base_calibration": str(base_path.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output, result)
    loaded = load_rule_calibration(output)
    return {
        "output": str(output.resolve()),
        "calibration_id": loaded.calibration_id,
        "calibration_sha256": loaded.file_sha256,
        "rules": result["rules"],
        "fit": result["fit"],
    }

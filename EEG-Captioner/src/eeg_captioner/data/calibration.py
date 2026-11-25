"""Load and validate the four data-dependent rule-QA thresholds."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from eeg_captioner.io import sha256


def default_calibration_path() -> Path:
    return Path(__file__).with_name("qa_calibration.json")


@dataclass(frozen=True)
class RuleCalibration:
    path: Path
    calibration_id: str
    min_margin: float
    temporal_pattern: Mapping[str, float]
    lateral_dominance: Mapping[str, float]
    data_contract: Mapping[str, Any]
    fit: Mapping[str, Any]
    file_sha256: str

    def threshold(self, question_type: str, band: str) -> float:
        if question_type == "temporal_pattern":
            values = self.temporal_pattern
        elif question_type == "lateral_dominance":
            values = self.lateral_dominance
        else:
            raise ValueError(f"Question type has no calibrated threshold: {question_type}")
        if band not in values:
            raise KeyError(f"Missing {question_type} threshold for band={band}")
        return float(values[band])


def _positive_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def load_rule_calibration(path: str | Path | None = None) -> RuleCalibration:
    calibration_path = Path(path) if path is not None else default_calibration_path()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported QA calibration schema: {calibration_path}")
    rules = payload.get("rules")
    if not isinstance(rules, dict):
        raise ValueError(f"Missing rules mapping in {calibration_path}")

    min_margin = float(rules.get("min_margin", math.nan))
    if not math.isfinite(min_margin) or not 0 <= min_margin < 1:
        raise ValueError("rules.min_margin must be finite and in [0, 1)")

    temporal = rules.get("temporal_pattern")
    lateral = rules.get("lateral_dominance")
    if not isinstance(temporal, dict) or set(temporal) != {"alpha", "gamma"}:
        raise ValueError("rules.temporal_pattern must contain exactly alpha and gamma")
    if not isinstance(lateral, dict) or set(lateral) != {"delta", "gamma"}:
        raise ValueError("rules.lateral_dominance must contain exactly delta and gamma")

    temporal_values = {
        name: _positive_finite(value, f"rules.temporal_pattern.{name}")
        for name, value in temporal.items()
    }
    lateral_values = {
        name: _positive_finite(value, f"rules.lateral_dominance.{name}")
        for name, value in lateral.items()
    }
    calibration_id = str(payload.get("calibration_id") or "").strip()
    if not calibration_id:
        raise ValueError(f"Missing calibration_id in {calibration_path}")

    return RuleCalibration(
        path=calibration_path.resolve(),
        calibration_id=calibration_id,
        min_margin=min_margin,
        temporal_pattern=temporal_values,
        lateral_dominance=lateral_values,
        data_contract=dict(payload.get("data_contract") or {}),
        fit=dict(payload.get("fit") or {}),
        file_sha256=sha256(calibration_path),
    )


def validate_processed_contract(path: str | Path, calibration: RuleCalibration) -> None:
    """Reject applying a calibration to a differently preprocessed H5."""
    contract = calibration.data_contract
    required = (
        "reference_scheme",
        "bandpass_hz",
        "notch_hz",
        "sample_rate_hz",
        "segment_duration_sec",
    )
    missing = [key for key in required if key not in contract]
    if missing:
        raise ValueError(f"Calibration data_contract is missing: {missing}")
    with h5py.File(Path(path), "r") as handle:
        reference = handle.attrs.get("reference_scheme", "")
        if isinstance(reference, bytes):
            reference = reference.decode("utf-8")
        if str(reference) != str(contract["reference_scheme"]):
            raise ValueError(
                f"Calibration expects reference={contract['reference_scheme']}, "
                f"got {reference}: {path}"
            )
        if not bool(handle.attrs.get("presegmented_10s", False)):
            raise ValueError(f"Calibration requires presegmented_10s input: {path}")
        scalar_checks = {
            "sample_rate_hz": float(handle.attrs.get("sample_rate", math.nan)),
            "segment_duration_sec": float(
                handle.attrs.get("segment_duration_sec", math.nan)
            ),
        }
        for name, actual in scalar_checks.items():
            if not math.isclose(
                actual, float(contract[name]), rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError(
                    f"Calibration expects {name}={contract[name]}, got {actual}: {path}"
                )
        raw_log = handle.attrs.get("preprocess_log", "{}")
        if isinstance(raw_log, bytes):
            raw_log = raw_log.decode("utf-8")
        preprocess_log = json.loads(str(raw_log))
        for name in ("bandpass_hz", "notch_hz"):
            if not np.allclose(
                np.asarray(preprocess_log.get(name), dtype=float),
                np.asarray(contract[name], dtype=float),
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"Calibration expects {name}={contract[name]}, "
                    f"got {preprocess_log.get(name)}: {path}"
                )

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

from eeg_captioner.channels import STANDARD_19_CHANNELS
from eeg_captioner.io import write_json
from . import tueg


def process_subject(task):
    subject, output = task
    subject = Path(subject)
    paths = []
    errors = []
    for path in sorted(subject.rglob("*.edf")):
        try:
            with tueg.mne.io.read_raw_edf(path, preload=False, verbose=False) as raw:
                names = [tueg.canonicalize_channel_name(c) for c in tueg.get_eeg_channel_names(raw)]
                if len(names) != len(set(names)):
                    raise ValueError("Duplicate canonical EEG channel names")
                if not names:
                    raise ValueError("No EEG channels")
                missing = [name for name in STANDARD_19_CHANNELS if name not in names]
                if missing:
                    raise ValueError(f"Missing required standard-19 channels: {missing}")
            paths.append(str(path))
        except Exception as e:
            errors.append({"file": str(path), "error": str(e)})
    results = []
    if paths:
        destination = Path(output) / (subject.name + ".h5")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        temporary = destination.with_suffix(".h5.partial")
        result = tueg.process_group(
            paths,
            str(temporary),
            subject.name,
            list(STANDARD_19_CHANNELS),
        )
        if result["status"] == "ok" and not result["errors"]:
            temporary.replace(destination)
            result["h5_path"] = str(destination)
        else:
            result["status"] = "failed"
        results.append(result)
    return {"subject": subject.name, "results": results, "header_errors": errors}


def run(edf_root, output, workers=1, max_subjects=None, partition=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if partition is None:
        subjects = tueg.discover_subjects(edf_root)
    else:
        partition_root = Path(edf_root) / str(partition)
        if not partition_root.is_dir():
            raise ValueError(f"Missing requested TUSZ partition: {partition_root}")
        subjects = [
            str(path)
            for path in sorted(partition_root.iterdir())
            if path.is_dir()
        ]
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    if not subjects:
        expected = (
            "EDF_ROOT/partition/subject/session/montage/*.edf"
            if partition is None
            else f"EDF_ROOT/{partition}/subject/session/montage/*.edf"
        )
        raise ValueError(f"Expected {expected}")
    if len({Path(s).name for s in subjects}) != len(subjects):
        raise ValueError("Duplicate subject IDs across partitions would overwrite H5 files")
    tasks = [(s, str(output)) for s in subjects]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reports = list(pool.map(process_subject, tasks))
    write_json(output / "preprocess_manifest.json", reports)
    if any(r["header_errors"] or any(x["status"] != "ok" for x in r["results"]) for r in reports):
        raise RuntimeError("Some recordings failed; inspect preprocess_manifest.json")


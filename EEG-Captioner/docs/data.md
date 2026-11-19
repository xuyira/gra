# Data and preprocessing

The TUEG-specific code is extracted from `build_tueg_h5.py`, not from the
multi-dataset inventory utility. It applies a 1.6–70 Hz bandpass, a 60 Hz
notch for TUSZ, MNE resampling to 200 Hz, fixed standard-19 channel selection,
and average re-referencing over exactly those 19 channels before converting
volts to microvolts. Each subject is exported in one fixed standard-19 order.

Preprocessing refuses to overwrite existing H5 files. Successful groups are
published from `.partial` files only after both passes succeed. Failures are
recorded in `preprocess_manifest.json` and cause a nonzero command exit. Keep
the original EDFs available when diagnosing a failed recording.

## Shared 10-second segment H5

- `data`: float32 `[segments, 19, 2000]`, microvolts at 200 Hz. Every row is
  the authoritative signal window used by both EEG-Captioner and CerebraGloss.
- `channels`: fixed standard-19 order `FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8
  T3 T4 T5 T6 FZ CZ PZ`. A recording missing any required channel is an error;
  preprocessing never silently fills a missing standard channel with zeros.
- `segment_ids`: stable source-recording IDs ending in `__segNNNN`;
  `parent_trial_ids`, `segment_indices`, `segment_start_samples`, and
  `segment_start_sec` retain the exact EDF correspondence.
- `trial_ids` is a compatibility alias of `segment_ids`, and every value in
  `trial_lengths_in_samples` is 2000.
- `sample_rate` attribute: 200.
- `reference_scheme` attribute: `average`; `reference_channel_signature`
  records the exact 19 channels included in the common average.
- `labram_channel_indices`: historical standard-channel selection metadata;
  this does not imply a dependency on the LaBraM repository.
- For TUSZ inputs, the sibling `.csv_bi` annotation is retained as
  `seizure_intervals_sec` (`[trial,max_events,2]`, seconds),
  `seizure_interval_count`, and `seizure_annotation_paths`. Overlapping
  seizure rows are merged and clipped to window-local time. Each row also
  stores `seizure_overlap_sec`, `seizure_fraction`, and binary `seizure_label`.
  A valid annotation containing only `bckg` rows has zero intervals. Every EDF
  must have a sibling `.csv_bi`; preprocessing fails immediately when the
  annotation is missing. Incomplete tails shorter than 10 seconds are recorded
  in metadata and dropped rather than padded.

Feature extraction retains the original quality control: 500 µV peak/1 µV
flat criteria, trial rejection thresholds, and
rejection of a 10-second segment if a remaining channel exceeds 100 µV.
It extracts 10-second non-overlapping segments and divides each into
`[0:666]`, `[666:1333]`, `[1333:2000]`. Feature plugins cover band powers,
time-domain statistics, spectral properties, spatial asymmetry and source networks.
The feature H5 also stores segment-aligned `meta/seizure_label`
(`1` when seizure overlap covers at least 15% of the segment and `0`
otherwise), `meta/seizure_overlap_sec`, and `meta/seizure_fraction`. For a
10-second segment the positive threshold is therefore 1.5 seconds. Labels are
strictly binary; missing annotations are errors rather than a third label.

The handcrafted feature route uses an effective 1.6–45 Hz analysis range and
defines delta as 1.6–4 Hz. The encoder's historical fixed Welch frontend keeps
its 0.5–45 Hz input shape for checkpoint compatibility; bins below 1.6 Hz see
the already high-pass-filtered signal. The original EEG-Captioner model still
reads 20 channels in order:

```text
FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 OZ F7 F8 T3 T4 T5 T6 FZ CZ PZ
```

The shared H5 deliberately omits OZ. The EEG-Captioner loader zero-fills only
OZ at its model boundary, while CerebraGloss consumes the stored `[19,2000]`
row directly. The model receives `[batch,20,10,200]` scaled by `/100`; feature
extraction and CerebraGloss use the same stored microvolt window directly.
Use `eeg_captioner.data.segment_store.load_shared_segment` to select a raw
window by row or stable `segment_id` for the CerebraGloss route.

Source-network artifacts are prepared explicitly with `source-artifacts` and
cached under the configured `artifacts_dir`. MNE may download fsaverage/Yeo
assets on first use. For an offline reduced smoke test, copy the YAML and set
`features.source_network.enabled: false`. This is not the paper annotation setup.

## Manifest

One JSON object per segment; paths are relative to the manifest location:

```json
{"id":"subject__trial__seg0000","eeg_path":"processed/subject.h5","summary_path":"summaries/0-1/template_v2_segments/subject__trial__seg0000.md","trial_index":0,"start_sample":0,"split":"train","caption":"...","labels":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
```

Labels follow the packaged 25-question ordering; `-100` means invalid and is
never sampled as a QA target. The preparation command defaults to the original
numeric validity plus bin-consistency filter, including quality-control channel
requirements. `--numeric-only` deliberately changes eligibility and is recorded
in manifest metadata. The shipped thresholds are frozen historical calibration
values; provenance should be considered before applying them to a new corpus.
The mainline does not need the older `train4` text-description training branch:
it applies its frozen bin thresholds directly to numerical features for QA
eligibility. Captions are generated from the full numerical segment summaries.

The annotation command selects full-segment stems ending in `__segNNNN` only;
per-third `__early`, `__middle`, and `__last` templates are diagnostic outputs,
not additional caption samples. Duplicate full-segment stems are rejected.

## Rule QA without captions

Deterministic signal-property QA can be exported directly from numerical
summaries without a teacher LLM or caption archive:

```bash
eeg-captioner rule-qa \
  --processed data/processed \
  --summaries data/summaries \
  --output data/rule_qa.jsonl
```

The JSONL is flattened to one record per eligible segment/question pair and
contains the exact H5 row, stable segment ID, question, options, answer index,
answer text, TUSZ seizure metadata, and a `bin_consistent` audit flag. By
default, all numerically decisive rule answers are retained. Add
`--require-bin-consistency` to reproduce the older preparation filter based on
the frozen population thresholds. Those thresholds were calibrated on the
historical multi-dataset corpus and should not be assumed optimal for TUSZ.

Caption text is embedded in the manifest, eliminating dependencies on external
caption files during training. EEG and summary paths are relocatable when moved
together with the manifest. Preparation records split counts, rejection counts,
question and threshold hashes. Duplicate sample IDs and cross-split file overlap
are rejected on loading.

Splitting uses the original seed-42 file-balancing algorithm. It is file-disjoint,
not guaranteed subject-disjoint: different H5 channel-signature groups of one
subject may land in different splits. Applications requiring subject-disjoint
evaluation should supply their own manifest with that grouping enforced.

The class-balanced training sampler requires at least one training example of
every class of every QA condition. A tiny subset can violate this requirement;
the trainer then reports the missing head/class rather than inventing targets.
Use the synthetic tests for a small complete-coverage smoke run.

## Retained caption archive

`prepare --captions annotations/captions` accepts the original checked JSONL gzip archive, as an alternative to directories of `.txt` files. See [annotation documentation](../annotations/README.md). The retained 204,460-caption collection is larger than the 160,781-sample historical training snapshot; exact snapshot membership is not yet recovered. Building a new manifest does not certify that it uses the paper's exact samples or partitions.

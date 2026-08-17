# FBG Impact Analysis

A signal-processing pipeline for analyzing Fiber Bragg Grating (FBG) interrogator data and automatically detecting impact events.

The project currently focuses on **Phase 3: Signal Processing Pipeline**, **Phase 4: Automatic Impact Detection**, **Phase 4.5: Ensemble Impact Detection with False-Positive Rejection** and **Phase 5: Impact Feature Extraction**. Later phases will extend the processed data into Machine Learning and Deep Learning models.

---

## Project Objective

The objective of this project is to convert raw FBG interrogator measurements into meaningful engineering signals and automatically identify impact events.

The overall pipeline is:

Raw Interrogator Data
        ↓
Data Import
        ↓
Data Cleaning
        ↓
Baseline Correction
        ↓
Wavelength Shift Calculation
        ↓
Signal Filtering
        ↓
Noise Analysis
        ↓
Impact Detection
        ↓
Ensemble Fusion (Phase 4.5)
        ↓
False-Positive Rejection (Phase 4.5)
        ↓
Impact Start / Peak / Recovery / End
        ↓
Impact Feature Extraction (Phase 5)
        ↓
Machine Learning / Deep Learning
        ↓
Future Phases

---

# Current Progress

## Phase 3 – Signal Processing Pipeline

Completed:

- Raw FBG data loading
- Data validation
- Missing-value checking
- Baseline estimation
- Baseline correction
- Wavelength-shift calculation
- Multiple filtering techniques
- Noise evaluation
- Filter comparison
- Saving processed/filtered signals
- Visualization of filtered signals

### Filters Implemented

Four filtering techniques are currently implemented:

1. Moving Average
2. Butterworth Filter
3. Savitzky-Golay Filter
4. Median Filter

The filters are evaluated using metrics such as:

- Raw noise standard deviation
- Filtered noise standard deviation
- Noise reduction
- Peak preservation
- Peak timing error

The best filter will be selected only after evaluating multiple datasets rather than relying on a single experiment.

## Phase 4.5 – Ensemble Impact Detection

Completed:

- Multi-method ensemble detector (peak, threshold, derivative and
  change-point detectors run in parallel)
- Temporal event matching across detectors
- Weighted evidence fusion (peak deviation, detector agreement,
  signal-to-noise)
- Impact boundary refinement (start / peak / end)
- False-positive rejection rules
- Per-channel baseline drift diagnostics
- Data-driven single-detector selection (selected primary analysis)
- Evaluation summary and diagnostic plots
- Unit and scenario tests

## Phase 5 – Impact Feature Extraction

Completed:

- Peak Shift, Residual Shift, Rise Time and Recovery Time extraction
  from the accepted Peak Detection events of the selected pipeline
- Reuse of the existing FBG2 Savitzky-Golay signal and boundaries
- Robust residual-shift estimation (median of a stable post-recovery
  window, excluding other detected impacts; NaN + reason when data
  is insufficient)
- Diagnostic feature plots (`results/phase5/plots/`)
- Unit tests for the feature calculations

---

# Selected Primary Analysis

In addition to the multi-method ensemble, a **single-detector** path
is provided for a controlled comparison. It operates only on the
FBG2 channel filtered with Savitzky-Golay (window 11, polyorder 3)
and runs exactly one detector. It does not use ensemble fusion.

The detector is **not hardcoded**: it is chosen from the actual
results. `src/analysis/method_selection.py` runs each of the four
detectors (peak, threshold, derivative, change_point) on FBG2 +
Savitzky-Golay across all datasets and scores them as:

```text
score = coverage x plausible_fraction x median_dev_std
```

where:

- `coverage` – fraction of datasets with at least one accepted event
  (recall proxy).
- `plausible_fraction` – fraction of events whose duration falls in
  [0.1 s, 3.0 s]. Events far shorter are glitch/edge artifacts;
  events far longer are usually drift or merged impacts.
- `median_dev_std` – median |peak - baseline| / baseline_std (impact
  strength proxy).

On the current `data/raw/` sets the selection is:

| method | events | coverage | median_dev_std | plausible_fraction | score |
|--------|-------:|---------:|---------------:|-------------------:|------:|
| **peak** | 110 | 0.83 | 64.9 | 0.955 | **51.7** |
| derivative | 12 | 0.83 | 29.3 | 0.583 | 14.2 |
| threshold | 8 | 0.67 | 19.0 | 0.500 | 6.4 |
| change_point | 15 | 0.83 | 64.1 | 0.000 | 0.0 |

Peak is selected: it finds the strongest, most physically plausible
events (durations ~0.4-1.3 s), while derivative/change-point detect
only glitch/edge-scale events (<= 0.23 s) and threshold produces
very long merged excursions. Run `python run_selected.py` to
recompute this table on the current data; the comparison is printed
before processing and an auditable `method_comparison.csv` can be
produced from it.

Selected-path events go through the same amplitude (>= 5x baseline
std), minimum-duration and confirmed-recovery gates as the ensemble,
with only the multi-detector agreement/evidence rules relaxed (a
single method can never satisfy ">= 2 detectors agreed"). Accepted
events therefore carry a meaningful `evidence_score` (the selected
detector's weight, e.g. 0.30 for peak) and `accepted` flag.

---

# Phase 5 – Impact Feature Extraction

Phase 5 consumes the **accepted** Peak Detection events produced by
the selected primary analysis (FBG2 + Savitzky-Golay + Peak
Detection) and extracts four scalar features per event:

**Peak Shift**

The difference between the peak wavelength shift and the
pre-impact baseline:

```text
peak_shift = peak_value - pre_impact_baseline
```

**Residual Shift**

The remaining offset between the post-impact stable level and the
pre-impact baseline:

```text
residual_shift = post_impact_level - pre_impact_baseline
```

The post-impact stable level is the **median** of a window after the
recovery/end point (the window starts a small gap after the event
end, contains enough samples, and excludes any region belonging to
another detected event). When there is not enough valid post-impact
data the residual features are NaN with a recorded reason. Residual
shift is only a signal offset relative to the baseline; it is not
claimed to represent physical damage.

**Rise Time**

```text
rise_time = peak_time - start_time
```

**Recovery Time**

```text
recovery_time = end_time - peak_time
```

Rise and recovery times reuse the existing start / peak /
recovery-end boundaries produced by the selected pipeline's boundary
refinement (the end is the first sample after the peak where the
filtered signal stays within the recovery tolerance for
`confirmation_samples` consecutive samples).

Phase 5 introduces no new detector and no ML/DL. It uses the same
FBG2 Savitzky-Golay signal and the same pre-impact baseline as the
selected pipeline.

Run:

```bash
python run_phase5.py --data data/raw
```

Outputs are written to `results/phase5/`:

- `phase5_features_all_datasets.csv` – one row per accepted event
  with `dataset`, `fbg`, `impact_id`, `start_time`, `peak_time`,
  `end_time`, `pre_impact_baseline`, `peak_value`, `peak_shift`,
  `absolute_peak_shift`, `post_impact_level`, `residual_shift`,
  `rise_time`, `recovery_time`
- `plots/` – a small set of diagnostic plots per dataset showing the
  signal, baseline, start/peak/end boundaries and the four features

---

# Phase 4 – Automatic Impact Detection

Four independent impact-detection approaches are being investigated.

### 1. Peak Detection

Detects significant positive and negative local peaks using:

- Peak prominence
- Minimum peak distance
- Baseline statistics

### 2. Threshold Detection

Detects regions where the wavelength shift deviates sufficiently from the baseline.

Conceptually:

|signal - baseline| > threshold

The threshold is currently derived from baseline noise statistics.

### 3. Derivative Detection

Uses the first derivative of the wavelength-shift signal:

d(signal) / dt

Large changes in the derivative can indicate rapid impact events.

Persistence is used to reduce isolated noise detections.

### 4. Change-Point Detection

Attempts to identify points where the statistical behavior of the signal changes.

### Ensemble fusion and false-positive rejection (Phase 4.5)

Candidates from all four detectors are matched into groups, fused into
single events and scored by weighted evidence (peak deviation, detector
agreement and signal-to-noise). A set of rejection rules removes likely
false positives before an event is accepted:

- **Extremely short events** – below a minimum duration.
- **No confirmed recovery** – events whose refined end reaches the end
  of the recording never returned to baseline, so they cannot be
  confirmed as impacts.
- **Noise-like events** – a single detector agreed and the peak
  deviation is too small relative to baseline noise.
- **Low amplitude** – peak deviation below a multiple of the baseline
  standard deviation.
- **Insufficient detector agreement / evidence score** – fewer
  detectors agreed or the weighted evidence is below threshold.

The ensemble summary (`results/ensemble/ensemble_summary.csv`) also
reports per-channel baseline diagnostics. Channels that exhibit large
baseline **drift** (reported as `drift_std` and `excursion_std` in
baseline-std units) can produce events that are drift artifacts rather
than physical impacts; such channels should be interpreted with
caution. No ground-truth labels exist for the expert datasets, so
accepted events can only be audited through the diagnostic plots in
`results/ensemble/plots/` and internal consistency metrics.

---

# Running the Pipeline

## Phase 4.5 – Ensemble detector

Run the ensemble detector on all datasets in `data/raw/`:

```bash
python run_ensemble.py --data data/raw --filter butterworth
```

Outputs are written to `results/ensemble/`:

- `ensemble_summary.csv` – per dataset/channel counts and baseline
  diagnostics (including `drift_std` and `excursion_std`)
- `accepted_events_all_datasets.csv` – accepted impacts across datasets
- `events_all_datasets.csv` – all candidates (accepted and rejected)
- `events_<dataset>.json` – per-dataset structured event records
- `plots/` – diagnostic plot per event for auditing accept/reject
  decisions

## Selected primary analysis (single detector)

Run the data-driven single-detector analysis on FBG2 + Savitzky-Golay:

```bash
python run_selected.py --data data/raw
```

The best detector is chosen from the results and printed before
processing. Override with `--method peak` (or threshold/derivative/
change_point). Outputs are written to `results/selected/`:

- `selected_summary.csv` – per-dataset channel/method counts and
  baseline diagnostics
- `selected_accepted_all_datasets.csv` – accepted events
- `selected_events_all_datasets.csv` – all candidates (accepted and
  rejected)
- `selected_events_<dataset>.json` – per-dataset structured records
- `plots/` – diagnostic plot per event

## Evaluation

```bash
python evaluate_ensemble.py --data data/raw --filter butterworth
```

## Tests

```bash
python -m pytest tests/
```

---

# Impact Event Information

For each detected impact region, the pipeline attempts to determine:

- Impact Start
- Peak
- Recovery
- Impact End
- Impact Duration
- Peak Value
- Peak Time

Example:

```text
Impact Start
     ↓
    Peak
     ↓
  Recovery
     ↓
  Impact End
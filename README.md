# FBG Impact Analysis

A signal-processing pipeline for analyzing Fiber Bragg Grating (FBG) interrogator data and automatically detecting impact events.

The project currently focuses on **Phase 3: Signal Processing Pipeline** and **Phase 4: Automatic Impact Detection**. Later phases will extend the processed data into Machine Learning and Deep Learning models.

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
Impact Start / Peak / Recovery / End
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
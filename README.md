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
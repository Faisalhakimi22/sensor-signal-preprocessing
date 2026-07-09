# Sensor Signal Preprocessing Pipeline

An 8-stage preprocessing pipeline for Human Activity Recognition (HAR) using smartphone accelerometer/gyroscope data — the unglamorous but high-leverage part of an ML project: turning noisy raw sensor signals into a clean, leakage-free feature set.

## Overview

This pipeline implements preprocessing techniques for the UCI HAR dataset end-to-end: outlier detection (Isolation Forest + IQR), Savitzky-Golay signal filtering, robust feature scaling, PCA-based dimensionality reduction, and statistical-moment feature engineering — each step chosen and ordered to avoid train/test leakage.

## Pipeline Steps

1. **Data Loading** - Load and parse UCI HAR dataset with unique feature naming
2. **Missing Value Handling** - Hybrid forward-fill + mean imputation
3. **Outlier Detection** - Isolation Forest + IQR capping
4. **Signal Filtering** - Savitzky-Golay filter for noise reduction
5. **Feature Scaling** - RobustScaler (median + IQR based)
6. **Dimensionality Reduction** - PCA with 95% variance retention
7. **Train-Test Split** - Stratified 80-20 split
8. **Feature Engineering** - Statistical moment features (skewness, kurtosis, range)

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
scipy
```

## Usage

```bash
python preprocessing_pipeline.py
```

## Dataset

UCI Human Activity Recognition Using Smartphones  
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Raw and generated data files are intentionally excluded from Git. See [DATA.md](DATA.md) for the expected local dataset layout and reproduction steps.

## Output

- `X_train_preprocessed.csv` - Preprocessed training features
- `X_test_preprocessed.csv` - Preprocessed test features
- `y_train.csv` / `y_test.csv` - Labels
- Visualization plots (`.png` files)

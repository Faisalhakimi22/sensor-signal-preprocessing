# Sensor Signal Preprocessing Pipeline

A comprehensive preprocessing pipeline for Human Activity Recognition (HAR) using smartphone sensor data.

## Overview

This pipeline implements advanced preprocessing techniques for the UCI HAR dataset, including outlier detection, signal filtering, feature scaling, dimensionality reduction, and feature engineering.

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

## Output

- `X_train_preprocessed.csv` - Preprocessed training features
- `X_test_preprocessed.csv` - Preprocessed test features
- `y_train.csv` / `y_test.csv` - Labels
- Visualization plots (`.png` files)

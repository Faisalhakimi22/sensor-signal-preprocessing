"""
Advanced Preprocessing Pipeline for Human Activity Recognition Dataset
Author: Student Name
Date: January 2026

Reference Paper: Sassi Hidri et al. (2025) - Enhancing Sensor-Based Human Physical 
Activity Recognition Using Deep Neural Networks. Journal of Sensor and Actuator Networks.
DOI: https://doi.org/10.3390/jsan14020042

Dataset: UCI Human Activity Recognition Using Smartphones
URL: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================
print("=" * 70)
print("PREPROCESSING PIPELINE FOR HUMAN ACTIVITY RECOGNITION")
print("=" * 70)

dataset_path = 'UCI HAR Dataset'
train_path = os.path.join(dataset_path, 'train', 'X_train.txt')
test_path = os.path.join(dataset_path, 'test', 'X_test.txt')
train_labels_path = os.path.join(dataset_path, 'train', 'y_train.txt')
test_labels_path = os.path.join(dataset_path, 'test', 'y_test.txt')
features_path = os.path.join(dataset_path, 'features.txt')

print("\n[STEP 1] Loading Dataset...")

# Load feature names and make unique
feature_names = pd.read_csv(features_path, sep=r'\s+', header=None, names=['index', 'feature'])
feature_list = feature_names['feature'].tolist()

seen = {}
unique_feature_list = []
for name in feature_list:
    if name in seen:
        seen[name] += 1
        unique_feature_list.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 0
        unique_feature_list.append(name)

# Load data
X_train_raw = pd.read_csv(train_path, sep=r'\s+', header=None, names=unique_feature_list)
y_train_raw = pd.read_csv(train_labels_path, sep=r'\s+', header=None, names=['activity'])['activity']
X_test_raw = pd.read_csv(test_path, sep=r'\s+', header=None, names=unique_feature_list)
y_test_raw = pd.read_csv(test_labels_path, sep=r'\s+', header=None, names=['activity'])['activity']

# Combine for preprocessing
X = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
y = pd.concat([y_train_raw, y_test_raw], axis=0, ignore_index=True)

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# =============================================================================
# STEP 2: MISSING VALUE HANDLING
# =============================================================================
print("\n[STEP 2] Handling Missing Values...")
print("Method: Hybrid forward-fill + mean imputation")

X_imputed = X.copy()
X_imputed = X_imputed.fillna(method='ffill').fillna(method='bfill')
for col in X_imputed.columns:
    if X_imputed[col].isnull().any():
        X_imputed[col].fillna(X_imputed[col].mean(), inplace=True)

print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")

# =============================================================================
# STEP 3: OUTLIER DETECTION AND REMOVAL
# =============================================================================
print("\n[STEP 3] Outlier Detection...")
print("Method: Isolation Forest + IQR capping")

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100, n_jobs=-1)
outlier_predictions = iso_forest.fit_predict(X_imputed)
outlier_mask = outlier_predictions == -1
n_outliers = outlier_mask.sum()

X_cleaned = X_imputed[~outlier_mask].copy()
y_cleaned = y[~outlier_mask].copy()

print(f"Outliers removed: {n_outliers} ({n_outliers/len(X_imputed)*100:.2f}%)")

# IQR capping
def cap_outliers_iqr(df, multiplier=1.5):
    """Cap outliers using IQR method"""
    df_capped = df.copy()
    for col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df_capped[col] = df_capped[col].clip(Q1 - multiplier*IQR, Q3 + multiplier*IQR)
    return df_capped

X_cleaned = cap_outliers_iqr(X_cleaned)
print(f"Dataset after cleaning: {X_cleaned.shape}")

# Visualization: Outlier Detection
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Outlier Detection (Isolation Forest)', fontsize=14, fontweight='bold')
for idx, feature in enumerate(X_imputed.columns[:3]):
    # Before
    axes[0, idx].scatter(range((~outlier_mask).sum()), X_imputed[~outlier_mask][feature].values, 
                         c='steelblue', alpha=0.5, s=5, label='Normal')
    axes[0, idx].scatter(range((~outlier_mask).sum(), len(X_imputed)), X_imputed[outlier_mask][feature].values,
                         c='red', alpha=0.7, s=10, label='Outliers')
    axes[0, idx].set_title(f'BEFORE: {feature[:20]}', fontsize=9)
    axes[0, idx].legend(fontsize=7)
    axes[0, idx].grid(True, alpha=0.3)
    # After
    axes[1, idx].scatter(range(len(X_cleaned)), X_cleaned[feature].values, c='green', alpha=0.5, s=5)
    axes[1, idx].set_title(f'AFTER: {feature[:20]}', fontsize=9)
    axes[1, idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('00_outlier_detection.png', dpi=300, bbox_inches='tight')
plt.close()

# Box plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Outlier Removal: Box Plot Comparison', fontsize=14, fontweight='bold')
X_imputed[X_imputed.columns[:6]].boxplot(ax=axes[0])
axes[0].set_title(f'BEFORE (n={len(X_imputed)})')
axes[0].tick_params(axis='x', rotation=45)
X_cleaned[X_cleaned.columns[:6]].boxplot(ax=axes[1])
axes[1].set_title(f'AFTER (n={len(X_cleaned)})')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('00b_outlier_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 00_outlier_detection.png, 00b_outlier_boxplot.png")

# =============================================================================
# STEP 4: SIGNAL FILTERING (NOVEL)
# =============================================================================
print("\n[STEP 4] Signal Filtering (Novel Addition)...")
print("Method: Savitzky-Golay filter (window=11, poly=3)")

X_filtered = X_cleaned.copy()
for i in range(min(100, X_filtered.shape[1])):
    try:
        X_filtered.iloc[:, i] = signal.savgol_filter(X_filtered.iloc[:, i], window_length=11, polyorder=3)
    except:
        pass

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Signal Filtering (Savitzky-Golay)', fontsize=14, fontweight='bold')
for idx, feat_idx in enumerate(np.random.choice(50, 4, replace=False)):
    ax = axes[idx//2, idx%2]
    ax.plot(X_cleaned.iloc[:200, feat_idx], alpha=0.5, label='Before', linewidth=1)
    ax.plot(X_filtered.iloc[:200, feat_idx], label='After', linewidth=2)
    ax.set_title(f'{X_cleaned.columns[feat_idx][:25]}', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_signal_filtering_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 01_signal_filtering_comparison.png")

# =============================================================================
# STEP 5: FEATURE SCALING
# =============================================================================
print("\n[STEP 5] Feature Scaling...")
print("Method: RobustScaler (median + IQR)")

scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Feature Scaling (RobustScaler)', fontsize=14, fontweight='bold')
X_filtered[X_filtered.columns[:5]].boxplot(ax=axes[0])
axes[0].set_title('Before Scaling')
axes[0].tick_params(axis='x', rotation=45)
X_scaled[X_scaled.columns[:5]].boxplot(ax=axes[1])
axes[1].set_title('After Scaling')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('02_feature_scaling_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 02_feature_scaling_comparison.png")

# =============================================================================
# STEP 6: DIMENSIONALITY REDUCTION (PCA)
# =============================================================================
print("\n[STEP 6] Dimensionality Reduction...")
print("Method: PCA with 95% variance retention")

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

print(f"Features reduced: {X_scaled.shape[1]} -> {X_pca.shape[1]} ({(1-X_pca.shape[1]/X_scaled.shape[1])*100:.1f}% reduction)")
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('PCA Analysis', fontsize=14, fontweight='bold')
# Variance ratio
axes[0,0].bar(range(1, min(21, len(pca.explained_variance_ratio_)+1)), 
              pca.explained_variance_ratio_[:20], color='steelblue', alpha=0.7)
axes[0,0].set_xlabel('Component')
axes[0,0].set_ylabel('Variance Ratio')
axes[0,0].set_title('Explained Variance by Component')
# Cumulative
cumsum = np.cumsum(pca.explained_variance_ratio_)
axes[0,1].plot(range(1, len(cumsum)+1), cumsum, 'g-', linewidth=2)
axes[0,1].axhline(0.95, color='r', linestyle='--', label='95% threshold')
axes[0,1].set_xlabel('Components')
axes[0,1].set_ylabel('Cumulative Variance')
axes[0,1].set_title('Cumulative Explained Variance')
axes[0,1].legend()
# PC1 vs PC2
y_reset = y_cleaned.reset_index(drop=True)
scatter = axes[1,0].scatter(X_pca_df['PC1'], X_pca_df['PC2'], c=y_reset, cmap='viridis', alpha=0.5, s=5)
axes[1,0].set_xlabel('PC1')
axes[1,0].set_ylabel('PC2')
axes[1,0].set_title('PC1 vs PC2 (by activity)')
plt.colorbar(scatter, ax=axes[1,0])
# Reduction bar
axes[1,1].bar(['Original', 'After PCA'], [X_scaled.shape[1], X_pca.shape[1]], color=['coral', 'lightgreen'])
axes[1,1].set_ylabel('Features')
axes[1,1].set_title('Dimensionality Reduction')
plt.tight_layout()
plt.savefig('03_pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 03_pca_analysis.png")

# =============================================================================
# STEP 7: TRAIN-TEST SPLIT
# =============================================================================
print("\n[STEP 7] Train-Test Split...")
print("Method: Stratified 80-20 split")

y_reset = y_cleaned.reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca_df, y_reset, test_size=0.2, random_state=42, stratify=y_reset
)

print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Train-Test Split Distribution', fontsize=14, fontweight='bold')
y_train.value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
axes[0].set_title(f'Training Set (n={len(y_train)})')
axes[0].set_xlabel('Activity')
y_test.value_counts().sort_index().plot(kind='bar', ax=axes[1], color='coral', alpha=0.7)
axes[1].set_title(f'Test Set (n={len(y_test)})')
axes[1].set_xlabel('Activity')
plt.tight_layout()
plt.savefig('04_train_test_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 04_train_test_distribution.png")

# =============================================================================
# STEP 8: FEATURE ENGINEERING (NOVEL)
# =============================================================================
print("\n[STEP 8] Feature Engineering (Novel Addition)...")
print("Method: Statistical moment features")

def add_statistical_moments(df):
    """Add statistical moment features"""
    df_enhanced = df.copy()
    df_enhanced['mean_skewness'] = skew(df.values, axis=1)
    df_enhanced['mean_kurtosis'] = kurtosis(df.values, axis=1)
    df_enhanced['max_value'] = df.max(axis=1)
    df_enhanced['min_value'] = df.min(axis=1)
    df_enhanced['range'] = df_enhanced['max_value'] - df_enhanced['min_value']
    return df_enhanced

X_train_final = add_statistical_moments(X_train)
X_test_final = add_statistical_moments(X_test)

print(f"Features after engineering: {X_train_final.shape[1]} (+5 new features)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Feature Engineering: Statistical Moments', fontsize=14, fontweight='bold')
new_features = ['mean_skewness', 'mean_kurtosis', 'max_value', 'min_value', 'range']
for idx, feat in enumerate(new_features):
    ax = axes[idx//3, idx%3]
    data = [X_train_final[y_train==act][feat].values for act in sorted(y_train.unique())]
    ax.boxplot(data, labels=sorted(y_train.unique()))
    ax.set_title(feat)
    ax.set_xlabel('Activity')
axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('05_feature_engineering.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 05_feature_engineering.png")

# =============================================================================
# SAVE PREPROCESSED DATA
# =============================================================================
print("\n[EXPORT] Saving preprocessed data...")

X_train_final.to_csv('X_train_preprocessed.csv', index=False)
X_test_final.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Saved: X_train_preprocessed.csv, X_test_preprocessed.csv, y_train.csv, y_test.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)
print(f"""
Original Dataset:     {X.shape[0]} samples, {X.shape[1]} features
After Outlier Removal: {X_cleaned.shape[0]} samples
After PCA:            {X_pca.shape[1]} components (95% variance)
After Feature Eng:    {X_train_final.shape[1]} features
Final Training Set:   {X_train_final.shape[0]} samples
Final Test Set:       {X_test_final.shape[0]} samples

Key Improvements over Reference Paper:
1. Isolation Forest outlier detection (vs manual removal)
2. Savitzky-Golay signal filtering (novel)
3. RobustScaler normalization (vs basic scaling)
4. Adaptive PCA (95% variance threshold)
5. Statistical moment features (novel)
""")
print("=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

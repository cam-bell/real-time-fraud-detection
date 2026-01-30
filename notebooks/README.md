# Notebook Experiments - mlg-ulb dataset

## Overview

This folder contains modeling work on the canonical credit card fraud dataset (`creditcard.csv`, a.k.a. the ULB/MLG dataset). It is a widely accepted benchmark for imbalanced learning, precision/recall tradeoffs, and PR-AUC evaluation.

**Primary notebook**: `full_analysis.ipynb`
**Legacy notebooks**: `full_analysis_original.ipynb`, `draft_analysis.ipynb`

These notebooks serve **technical reviewers and ML evaluators** who want to inspect model development decisions. The streaming notebooks (e.g., `stream_simulation_*`) serve a different audience and focus on real-time behavior, drift detection, and system design.

## Project Structure (high level)

- `data/creditcard.csv` - canonical ULB credit card fraud dataset used by the notebooks here.
- `notebooks/full_analysis.ipynb` - end-to-end batch modeling, EDA, model comparison, and tuning.
- `notebooks/stream_simulation_*.ipynb` - streaming feature engineering, drift, and deployment-oriented workflows.
- `src/` - production pipeline code (feature engineering, streaming, scoring).
- `models/` - trained artifacts saved by notebooks or pipeline runs.
- `outputs/` - evaluation outputs, plots, and reports.
- `tests/` - unit/integration tests.

## Dataset Description

**File**: `data/creditcard.csv`
**Rows**: 284,807 transactions
**Fraud cases**: 492 (~0.17% fraud rate)
**Columns**: `Time`, `Amount`, `V1`-`V28` (PCA components), `Class` (target)

Notes:
- `Time` is seconds elapsed between each transaction and the first transaction in the dataset.
- `Amount` is the raw transaction value.
- `V1`-`V28` are anonymized PCA features.
- The extreme class imbalance makes PR-AUC and precision/recall the most meaningful metrics.

## Exploratory Data Analysis (full_analysis.ipynb)

The notebook walks through structured EDA with an emphasis on imbalance and distributional differences:

- **Class distribution** and imbalance ratio visualization
- **Duplicate analysis** and a decision to remove duplicates to reduce leakage risk
- **Correlation analysis** to identify features most associated with fraud
- **Distribution comparisons** between fraud and non-fraud classes
- **Statistical significance tests** for feature differences
- **Time and amount analysis** to surface temporal and value-based patterns

## Baseline Modeling & Model Comparison

The modeling workflow in `full_analysis.ipynb` is organized to compare algorithms and imbalance strategies:

- **Models**: Logistic Regression, Random Forest, XGBoost
- **Resampling**: None, Random Under Sampling (RUS), Random Over Sampling (ROS), SMOTE
- **Metric**: PR-AUC (with ROC-AUC, F1, precision, recall reported)

Key results:

- **Best baseline**: XGBoost with `scale_pos_weight` and **no resampling**
  (PR-AUC ~ 0.823 at baseline; outperforms RF and LogReg)
- **SMOTE helps Random Forest**, but **hurts XGBoost** relative to class weighting
- **RUS degrades precision** and is not favored for this dataset

## Preprocessing & Scaling Decisions

- **Train/test split**: 80/20, stratified
- **Scaling**: StandardScaler vs RobustScaler comparison (applied to `Time` and `Amount` for LogReg)
- **Decision**: StandardScaler chosen (marginally better and consistent with common practice)

## Hyperparameter Tuning & Final Model

- **Tuning**: RandomizedSearchCV on XGBoost with PR-AUC scoring
- **Outcome**: tuned XGBoost improves PR-AUC to **~0.83**
- **Final selection**: XGBoost with class weighting (no resampling)

## Ensemble Methods

The notebook briefly tests an ensemble (XGBoost + Random Forest). It does **not** outperform the tuned XGBoost model and is not selected.

## Conclusions (full_analysis.ipynb)

Key takeaways from the notebook:

- **PR-AUC > ROC-AUC** for imbalanced datasets
- **Class weighting** in XGBoost is more effective than SMOTE for this dataset
- **Duplicate removal** improves data integrity with minimal downside
- **Standard scaling** is sufficient for linear models; tree-based models are scale-invariant

## How to run

1. Ensure `data/creditcard.csv` exists locally.
2. Install requirements: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open and run `notebooks/full_analysis.ipynb`

For streaming and drift-focused work, see `notebooks/stream_simulation_detailed.ipynb`.

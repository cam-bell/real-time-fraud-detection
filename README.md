# Fraud Real-Time Pipeline

An end-to-end fraud detection and risk scoring project that combines model training, streaming simulation, monitoring, drift detection, and a Streamlit dashboard for interactive playback.

The repository supports two complementary workflows:

- notebook-based experimentation and model development
- source-based replay of a fraud transaction stream with saved model artifacts

## Overview

This project implements a real-time fraud detection pipeline designed to:

- ingest financial transactions in simulated real time
- score each transaction using a trained machine learning model
- track key performance metrics continuously
- detect concept drift in feature distributions and model behavior
- provide real-time visualizations and alerting
- support future expansion into AI-based fraud, AML, and broader risk scoring workflows

The system is aimed at financial institutions, banks, fintechs, and compliance teams that need scalable and adaptive fraud monitoring on structured transaction data.

## System Architecture

The pipeline is organized into modular components:

- Ingestion layer: receives transaction events from a stream emulator, with a path to future queue-backed ingestion
- Feature engineering service: derives temporal, behavioral, geographic, amount-based, velocity, and EDA-guided features
- Scoring service: applies the trained fraud model to incoming transactions in real time
- Metrics tracker: computes sliding-window precision, recall, F1, PR-AUC, ROC-AUC, and fraud-rate trends
- Drift detector: monitors feature and score distributions for statistical drift
- Dashboard and alerts: surfaces risk scores, metrics, confusion matrices, and alert conditions in Streamlit

This separation makes it possible to evolve feature engineering or model logic without rewriting the rest of the pipeline.

## Repository Layout

- `src/` application code for ingestion, features, models, monitoring, and dashboard logic
- `src/dashboards/` Streamlit app and dashboard helpers
- `src/features/` online and batch feature engineering
- `src/ingestion/` transaction stream simulator
- `src/models/` training and scoring components
- `src/monitoring/` metrics, drift detection, and alerting
- `notebooks/` exploratory analysis, model development, and streaming-oriented experiments
- `data/` local datasets used by notebooks and dashboard replay
- `models/` trained pipelines, thresholds, and metadata used by the dashboard
- `outputs/` generated charts, histories, snapshots, and other local run artifacts
- `tests/` lightweight regression tests for core source modules

## Python And Environment Setup

Use Python `3.12`. The repository includes [`.python-version`](/Users/cameronbell/Projects/fraud_real_time_pipeline/.python-version) for local tool alignment.

Recommended setup with `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra notebook --extra dev
```

If you prefer `pip`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependency management notes:

- [pyproject.toml](/Users/cameronbell/Projects/fraud_real_time_pipeline/pyproject.toml) is the source of truth for dependencies
- [requirements.txt](/Users/cameronbell/Projects/fraud_real_time_pipeline/requirements.txt) is kept as a compatibility shim for `pip`
- runtime dependencies are separated from notebook and dev extras

## Data And Model Artifacts

The dashboard expects local CSV and model artifacts rather than downloading them at runtime.

### Dataset Setup

Download the datasets from their canonical Kaggle sources and place the files under `data/` with these exact names:

- `data/creditcard.csv` from [Credit Card Fraud Detection | Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- `data/fraudTrain.csv` from [Credit Card Transactions Fraud Detection Dataset | Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- `data/fraudTest.csv` from [Credit Card Transactions Fraud Detection Dataset | Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

After downloading, your `data/` directory should contain:

```text
data/
├── creditcard.csv
├── fraudTrain.csv
└── fraudTest.csv
```

Required dashboard inputs:

- `data/fraudTest.csv`
- `models/fraud_detection_pipeline.pkl`
- `models/model_metadata.json`

Optional v3 support is enabled automatically when these artifacts are present:

- `models/fraud_detection_pipeline_v3.pkl`
- `models/model_metadata_v3.json`
- `models/target_encodings_v3.json`
- `models/threshold_cost_v3.pkl`
- `models/threshold_f1_v3.pkl`

The application also supports these environment overrides:

- `FRAUD_DATA_DIR`
- `FRAUD_MODELS_DIR`

## Running The Project

Launch the dashboard from the project root:

```bash
streamlit run src/dashboards/streamlit_app.py
```

Run the test suite:

```bash
pytest
```

The Streamlit dashboard loads the v2 model by default and offers v3 when the corresponding artifacts exist in `models/`.

## Notebooks

The notebooks support both batch fraud analysis and streaming-oriented fraud simulation.

Primary streaming notebooks:

| Notebook | Role | Model artifacts |
| --- | --- | --- |
| `stream_quick.ipynb` | Main pipeline covering v1 to v3, including velocity features, target encoding, and dual-threshold logic | v2 and v3 |
| `stream_quick_v2_only.ipynb` | v2-only variant without the v3 extensions | v2 |
| `stream_simulation_detailed.ipynb` | Full run with more detailed EDA and explanatory notes | v2 |
| `stream_simulation_alternate.ipynb` | Same phases with a different presentation of feature engineering and targeted EDA | v2 |

Primary batch-analysis notebook:

- `notebooks/full_analysis.ipynb` for work on the canonical `creditcard.csv` fraud dataset

To start a notebook environment:

```bash
jupyter lab
```

For more notebook-specific context, see [notebooks/README.md](/Users/cameronbell/Projects/fraud_real_time_pipeline/notebooks/README.md).

## Machine Learning Approach

### Model Choice

The core classifier is XGBoost, chosen for:

- strong performance on tabular fraud data
- built-in handling of class imbalance through `scale_pos_weight`
- fast enough inference for real-time scoring scenarios

### Feature Engineering

The project uses a mix of basic and higher-signal engineered features:

- temporal features such as hour of day, day of week, and weekend flags
- amount-based transforms and risk bins
- geographic features such as merchant distance
- behavioral and velocity features at the card level
- interaction features such as high amount plus late-night activity
- categorical encodings, including target encoding in the v3 workflow

These choices are intended to improve fraud signal capture without creating an unnecessarily large online feature surface.

### Imbalance Handling

The project treats fraud as an inherently imbalanced classification problem:

- `scale_pos_weight` is used instead of relying on production-time resampling
- thresholding is tuned separately from model fitting
- notebook experiments compare resampling approaches, but the production-oriented path favors class weighting

This aligns with common practice in fraud systems, where the cost of missed fraud is usually much higher than the cost of extra reviews.

## Real-Time Simulation, Metrics, And Thresholds

Transactions are replayed chronologically to simulate a live stream. Metrics are computed over sliding windows rather than a single static evaluation set.

Recommended metric window sizes are generally in the `2,000` to `5,000` transaction range so that sparse fraud events still produce stable recall, precision, and PR-AUC estimates.

The project supports two operational threshold views:

- cost-optimized threshold, typically favoring higher recall
- F1-optimized threshold, balancing precision and recall

The v3 workflow also supports dual-threshold decisioning, which is useful when separating manual review from stronger automated actions.

## Drift Detection And Alerting

The monitoring layer uses sliding-window statistics to detect drift in:

- score distributions
- selected feature distributions
- model performance relative to a baseline

Drift and risk events can trigger alerts that are visualized in the dashboard and can be used to drive downstream retraining, recalibration, or investigation workflows.

This makes the project more than a static fraud classifier: it is a prototype of an operational fraud monitoring system.

## Vision And Future Enhancements

The current pipeline establishes a base for a broader AI-based fraud and AML platform.

Likely next steps include:

- continuous risk scoring that combines transaction, behavioral, and entity signals
- graph-based or network-based fraud features
- integration of unstructured sources such as KYC documents, adverse media, or investigation notes
- explainability layers using SHAP, LIME, or similar approaches
- automated retraining or threshold recalibration on drift events
- deployment into cloud-native serving and monitoring stacks

## Security, Compliance, And Governance

Any real-world deployment in a financial setting should account for:

- data privacy obligations such as GDPR and PCI DSS
- model governance, auditability, and reproducibility
- fairness and bias controls in risk scoring
- explainable outputs for reviewers, compliance teams, and regulators

The repository is a prototype and research workflow, but those constraints should shape any productionization path.

## References And Further Reading

- [Financial fraud detection using machine learning | Alloy](https://www.alloy.com/blog/data-and-machine-learning-in-financial-fraud-prevention?utm_source=chatgpt.com)
- [AI Transaction Monitoring and how it works](https://www.ir.com/guides/ai-transaction-monitoring-and-how-it-works-complete-guide-2025?utm_source=chatgpt.com)
- [AI in AML: Top Use Cases You Need To Know](https://smartdev.com/ai-use-cases-in-aml/?utm_source=chatgpt.com)
- [Strategic Defense Against Financial Crime: A 3-Phase AI Approach](https://ankura.com/insights/strategic-defense-against-financial-crime-a-3-phase-ai-approach/?utm_source=chatgpt.com)
- [Quantifind](https://en.wikipedia.org/wiki/Quantifind?utm_source=chatgpt.com)
- [Deep Learning in Financial Fraud Detection](https://www.sciencedirect.com/science/article/pii/S2666764925000372?utm_source=chatgpt.com)
- [AML AI overview | Anti Money Laundering AI](https://docs.cloud.google.com/financial-services/anti-money-laundering/docs/concepts/overview?utm_source=chatgpt.com)

# **Real-Time Fraud Detection & Risk Scoring System**

Streaming fraud detection: train models, simulate a transaction stream, track metrics, detect drift, and visualise in a dashboard.

**Technical Documentation**

---

## **1. Overview**

This project implements an end-to-end **real-time fraud detection pipeline** designed to:

- Ingest financial transactions in simulated real time
- Score each transaction using an ML model (XGBoost with engineered features)
- Track key performance metrics continuously
- Detect concept drift in feature distributions and model performance
- Provide real-time visualisations and alerting
- Support future expansion into **AI-based fraud/AML detection** and **risk scoring with structured and unstructured data**

The system is aimed at **financial institutions, banks, fintechs, and compliance teams** that require scalable, adaptive fraud detection and risk scoring workflows.

---

## **2. System Architecture**

The pipeline comprises the following modular components:

### **Ingestion Layer**

Receives transaction events in real time from a streaming emulator or real message queue (e.g., Kafka/Redis).

### **Feature Engineering Service**

Applies both basic and advanced features per transaction, including temporal, behavioural, amount, velocity, geographic and EDA-guided features.

### **Scoring Service**

Serves the trained XGBoost classifier for real-time scoring of incoming transactions.

### **Metrics Tracker**

Computes performance metrics in sliding windows — precision, recall, F1, PR-AUC, and fraud rate — and tracks them over time.

### **Drift Detector**

Monitors feature and score distributions to detect statistical drift in incoming data streams.

### **Dashboard & Alerts**

A visual interface (Streamlit/Grafana) that shows:

- Transaction score over time
- Metrics history
- Alerts for drift or anomalies
- Confusion matrices for current windows
- Performance dashboards

This modular design enables both **end-to-end pipeline functionality** as well as **model-centric improvements without disrupting core services**.

## Notebooks

| Notebook                            | Role                                                                                          | Model artifacts |
| ----------------------------------- | --------------------------------------------------------------------------------------------- | --------------- |
| `stream_quick.ipynb`                | Main pipeline: v1→v2→v3 (velocity features, target encoding, dual thresholds).                | v2 + v3         |
| `stream_quick_v2_only.ipynb`        | Same flow as main but v2 only (no v3).                                                        | v2              |
| `stream_simulation_detailed.ipynb`  | Full run with detailed EDA and long docstrings.                                               | v2              |
| `stream_simulation_alternate.ipynb` | Same phases; Phase 1 structured as “Production-Ready Feature Engineering” and “Targeted EDA”. | v2              |

- **Main pipeline**: `stream_quick.ipynb`
- **v2-only variant**: `stream_quick_v2_only.ipynb` (predecessor to stream_quick, no v3)
- **Detailed run**: `stream_simulation_detailed.ipynb` (more EDA breadth/depth)
- **Alternate Phase 1 structure**: `stream_simulation_alternate.ipynb`

### Dashboard and artifacts

The Streamlit dashboard (`src/dashboards/streamlit_app.py`) loads **v2** by default (`fraud_detection_pipeline.pkl`, `model_metadata.json`) and **v3** when present (`fraud_detection_pipeline_v3.pkl`, `model_metadata_v3.json`, `target_encodings_v3.json`). Run any notebook that produces v2 to use the dashboard; run `stream_quick.ipynb` to also produce v3 and enable the v3 model option in the dashboard.

**Artifact paths (under `models/`):**

- **v2**: `fraud_detection_pipeline.pkl`, `model_metadata.json`, `threshold.pkl` or `optimal_threshold.pkl`
- **v3**: `fraud_detection_pipeline_v3.pkl`, `model_metadata_v3.json`, `target_encodings_v3.json`, optional `threshold_cost_v3.pkl`, `threshold_f1_v3.pkl`

## Running the dashboard

From project root:

```bash
streamlit run src/dashboards/streamlit_app.py
```

## Project structure

- `notebooks/` – Streaming fraud notebooks (see table above)
- `src/dashboards/` – Streamlit app and helpers
- `src/features/` – Feature engineering
- `src/ingestion/` – Stream simulator
- `src/models/` – Training and scoring
- `src/monitoring/` – Metrics, drift, alerts
- `models/` – Saved pipelines and metadata (v2 and v3)
- `config/` – Logging and settings

---

## **3. Machine Learning Model**

### **Model Choice**

The core classifier is **XGBoost**, chosen for:

- Strong performance on tabular data
- Ability to handle imbalance via `scale_pos_weight`
- Fast inference suitable for real-time scoring

### **Features**

Key engineered feature groups include:

**Temporal & Behaviour Features**

- Hour of day, day of week, weekend flag
- Rolling spend/velocity metrics per card

**Amount & Risk Indicators**

- Log-transformed amount, high/medium/low risk bins
- Interaction features (e.g., high amount + late night)

**Categorical Encoding**

- Target/mean encoding for mediums such as merchant, category, state
- Frequency encoding for rare categories

Using target/mean encoding avoids explosion of dimensionality compared to One-Hot Encoding and improves interpretability and performance in real time.

### **Imbalance Handling**

- `scale_pos_weight` set based on class ratio
- Avoids costly resampling in production
- Optionally combined with cost-sensitive threshold optimisation

This reflects _industry practice_ where class imbalance is inherent and managed within model loss rather than artificial oversampling. ([Alloy][1])

---

## **4. Real-Time Simulation & Windowing**

### **Streaming Simulation**

Transactions are read chronologically to simulate a real stream.
Metrics are updated in sliding windows sized to reflect _fraud event density_ rather than arbitrary counts.

**Recommended Window Sizes**

- 2,000–5,000 transactions per window (to capture meaningful fraud events given ~0.3–0.6% fraud rate).
- Dynamic window controls allow observing metrics over different timescales.

This ensures performance metrics like precision/recall/PR-AUC are informative rather than undefined — a common challenge with sparse events in early windows.

---

## **5. Performance & Threshold Optimisation**

Two thresholds are often considered:

- **Cost-optimised threshold** — prioritises high recall to catch fraud early.
- **F1-optimised threshold** — balances precision and recall.

In practice, **cost optimisation is usually preferred** because the cost of missed fraud (false negatives) typically outweighs the cost of extra reviews (false positives). This matches how production fraud pipelines are tuned in real institutions. ([ir.com][2])

Additionally, systems can apply **dual thresholds**:

- Lower threshold → automated alerts/blocks
- Higher threshold → manual review queue

This _tiered decision logic_ improves operational decisioning.

---

## **6. Drift Detection & Alerting**

The system implements statistical monitoring of feature distributions using sliding windows to detect concept drift (e.g., shifts in amount, velocity or category usage) that could degrade model performance.

Drift triggers alerts that can be visualised in real time and used to:

- Retrain models
- Re-calibrate thresholds
- Initiate deeper investigation

Integrating drift detection ensures the model stays robust to evolving fraud tactics — a key requirement in deployed systems.

---

## **7. Vision: AI-Based Fraud & AML Detection Pipeline**

Your current work sets a solid foundation for broader scope — combining structured and unstructured data within an AI-based fraud/AML detection system.

### **Why This Matters**

Traditional rule-based systems are rigid and generate many false positives. AI-based systems leverage statistical learning to detect complex, evolving fraud patterns that rules alone miss. ([ir.com][2])

**Anti-Money Laundering (AML)** systems extend fraud detection into compliance — spotting suspicious activity that could indicate money laundering, terrorist financing, or regulatory violations. Modern AML AI systems can detect _2–4× more suspicious activity_ and reduce false positives significantly compared to rules-based methods. ([SmartDev][3])

### **Extending to Structured + Unstructured Data**

Your architecture can be expanded in two ways:

#### **Structured Data**

Already part of your pipeline — transactions, amounts, temporal features, cards, etc. Additional structured data sources can include:

- Customer account metadata
- Device and geolocation telemetry
- Velocity and network features (graph-based)

#### **Unstructured Data**

In real scenarios, compliance and risk decisions often depend on unstructured sources such as:

- Customer documents (KYC forms)
- Adverse media / news feeds
- Emails or chat logs related to investigations
  AI techniques (NLP, graph embeddings) can extract signals from unstructured text and feed them into risk models. ([Ankura.com][4])

Applications:

- Entity resolution and enrichment
- Serious adverse media linkage
- Network behaviour analysis

Systems like Quantifind ingest structured and unstructured public data (news, filings, sanctions lists) to score entity risk. ([Wikipedia][5])

---

## **8. Future Enhancements (Roadmap)**

Here’s how your project can evolve toward an industry-grade risk scoring platform:

**A. AI-Powered Risk Scoring Module**

- Integrate behavioural, transaction, and entity signals into continuous risk scores.
- Use graph analytics for relational risk (merchant–customer networks).

**B. Unstructured Data Integration**

- Incorporate NLP to derive risk signals from text sources (news, KYC documents).
- Implement Named Entity Recognition (NER) in risk features.

**C. Model Explainability**

- Incorporate SHAP, LIME or attention explanations for transparency and auditability — critical for compliance.

**D. Online Learning and Adaptive Retraining**

- Set triggers for automated retraining when drift or performance deterioration occurs.

**E. Production Deployment**

- Migrate scoring service to Kubernetes or cloud functions.
- Store metrics in time-series DBs (Prometheus/InfluxDB) and use Grafana for operational monitoring.

---

## **9. Security, Compliance & Ethical Considerations**

Real financial systems must consider:

- Data privacy (GDPR, PCI DSS)
- Model governance and auditability
- Fairness and bias in risk scoring
- Explainable outputs for regulatory reporting

Ensuring explainability and compliance aligns with industry expectations for ML in regulated domains.

---

## **10. References & Further Reading**

For deeper context on how AI is reshaping fraud and AML detection:

- Deep learning frameworks for financial fraud detection research overview. ([ScienceDirect][6])
- AI-based anti-money laundering concepts and risk scoring. ([Google Cloud Documentation][7])
- Strategic AI in financial crime defence. ([Ankura.com][4])
- AI transaction monitoring advantages over traditional rule systems. ([ir.com][2])

---

[1]: https://www.alloy.com/blog/data-and-machine-learning-in-financial-fraud-prevention?utm_source=chatgpt.com "Financial fraud detection using machine learning | Alloy"
[2]: https://www.ir.com/guides/ai-transaction-monitoring-and-how-it-works-complete-guide-2025?utm_source=chatgpt.com "AI Transaction Monitoring and how it works"
[3]: https://smartdev.com/ai-use-cases-in-aml/?utm_source=chatgpt.com "AI in AML: Top Use Cases You Need To Know"
[4]: https://ankura.com/insights/strategic-defense-against-financial-crime-a-3-phase-ai-approach/?utm_source=chatgpt.com "Strategic Defense Against Financial Crime: A 3-Phase AI ..."
[5]: https://en.wikipedia.org/wiki/Quantifind?utm_source=chatgpt.com "Quantifind"
[6]: https://www.sciencedirect.com/science/article/pii/S2666764925000372?utm_source=chatgpt.com "Deep Learning in Financial Fraud Detection"
[7]: https://docs.cloud.google.com/financial-services/anti-money-laundering/docs/concepts/overview?utm_source=chatgpt.com "AML AI overview | Anti Money Laundering AI"

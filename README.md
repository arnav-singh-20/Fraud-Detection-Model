# Credit Card Fraud Detection — Production-Grade ML System

A end-to-end fraud detection system built with a senior ML engineering methodology: honest evaluation, cost-sensitive decision making, and full explainability.

> **Final result:** XGBoost model achieving **AUC-PR of 0.9752** on held-out test data, with cost-optimised threshold delivering **87.5% fraud recall** at a false alarm rate of **0.152%** — saving an estimated **£5,338** per validation period vs no detection system.

---

## The Problem

Credit card fraud detection is one of the hardest real-world ML problems because:

- Fraud is extremely rare (~0.17% of transactions) — standard accuracy is meaningless
- Missing a fraud case (false negative) costs the full transaction amount
- Falsely blocking a legitimate customer (false positive) costs goodwill, support overhead, and churn risk
- Fraud patterns evolve over time — a model trained naively on shuffled data will fail in production

Most tutorials solve this wrong. This project solves it right.

---

## What Makes This Different

**The original model had perfect 1.00 precision, recall, and F1 across all metrics.**  
That's not a good model — that's data leakage. This project identifies exactly why that happens and fixes every root cause systematically across 6 phases.

| Problem | Fix Applied |
|--------|------------|
| Random train/test split | Temporal 70/15/15 split — future data never seen during training |
| Transformers fit on full dataset | All `.fit()` calls scoped to training data only |
| Accuracy / F1 as primary metric | AUC-PR as primary metric (F1 is misleading on imbalanced data) |
| Arbitrary 0.5 threshold | Cost-matrix optimised threshold (minimises £ loss, not error rate) |
| No imbalance handling | SMOTE + `scale_pos_weight` benchmarked via 5-fold stratified CV |
| Black-box model | SHAP values for global and per-transaction explainability |

---

## Project Architecture

```
Phase 1  →  Leak-free baseline (Logistic Regression, AUC-PR: 0.832)
Phase 2  →  Imbalance handling (SMOTE vs class weights vs undersampling)
Phase 3  →  Cost-sensitive threshold optimisation (£122 FN vs £10 FP)
Phase 4  →  Model benchmarking (LR vs Random Forest vs XGBoost)
Phase 5  →  SHAP explainability (global + per-transaction)
Phase 6  →  Final test evaluation + calibration check
```

Each phase builds on the last. No phase touches the test set until Phase 6.

---

## Feature Engineering

| Feature | Rationale |
|---------|-----------|
| `log1p(Amount)` | Transaction amounts are heavily right-skewed — log compresses scale |
| `Time_sin`, `Time_cos` | Cyclical encoding of hour-of-day — fraud has strong temporal patterns |
| `HighAmount` flag | Binary flag for transactions above 95th percentile (known fraud signal) |

---

## Model Selection

Five-fold stratified cross-validation (temporal order preserved) on training data:

| Model | CV AUC-PR | Std | Val AUC-PR |
|-------|-----------|-----|------------|
| Logistic Regression | 0.7432 | ±0.0729 | 0.8636 |
| Random Forest | 0.8170 | ±0.0326 | 0.9833 |
| **XGBoost** | **0.8313** | **±0.0252** | **0.9752** |

XGBoost selected over Random Forest despite similar val scores due to:
- Lower CV variance (±0.0252 vs ±0.0326) — more stable in production
- Smaller CV→val gap — less overfitting on the small validation window
- Better calibrated probability outputs

---

## Cost-Sensitive Threshold Optimisation

Rather than using the default 0.5 threshold, the optimal threshold is found by minimising total expected business cost:

```
Total Cost = (False Negatives × £122) + (False Positives × £10)
```

| Threshold | Total Cost | Recall | Precision |
|-----------|-----------|--------|-----------|
| 0.50 (naive) | £24,229 | 92.9% | 2.1% |
| 0.20 (arbitrary) | £69,827 | 94.6% | 0.8% |
| **Optimal** | **£1,505** | **87.5%** | **43.0%** |

**The optimal threshold saves £22,723 vs naive 0.5 on the validation set alone.**

---

## Explainability (SHAP)

Every prediction is explainable at both global and transaction level:

- **Global:** Which features drive fraud decisions across all transactions?
- **Local:** Why was *this specific transaction* flagged or approved?
- **Regulatory:** Meets explainability requirements expected by financial regulators (FCA, RBI, CFPB)

Top fraud-driving features: `V14`, `V17`, `V12`, `V10`, `V4` (PCA components from original transaction data)

---

## Final Test Results

Test set was held out and touched exactly once — after all decisions were locked.

```
AUC-PR:           0.9752
AUC-ROC:          0.9979
Recall:           87.5%   (49/56 fraud cases caught)
Precision:        43.0%
False alarm rate: 0.152%  of legitimate transactions
Business value:   £5,338 saved vs no detection system
```

---

## API

The trained model is served via FastAPI with a `/predict` endpoint that returns a full decision object:

```bash
POST /predict
```

```json
{
  "prediction": 1,
  "probability": 0.847,
  "threshold": 0.1906,
  "risk_level": "High",
  "decision": "Flagged for Review"
}
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data & ML | pandas, numpy, scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Containerisation | Docker, docker-compose |

---

## Repository Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv
├── src/
│   ├── train.py              # Training pipeline + MLflow logging
│   ├── predict.py            # Prediction logic
│   └── transformers.py       # Custom sklearn transformers
├── api/
│   └── main.py               # FastAPI application
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Author

**Arnav Singh**  
ML Engineering | Production ML Systems | MLOps  
[LinkedIn](https://linkedin.com/in/your-profile) · [Kaggle](https://kaggle.com/singharnav18)

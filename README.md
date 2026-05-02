# 💳 Credit Card Fraud Detection — End-to-End ML System

## 🚀 Overview
This project builds a **production-ready fraud detection system** that goes beyond simple classification and focuses on **business decision-making**.

The system:
- Minimizes financial loss using **cost-sensitive threshold optimization**
- Handles extreme class imbalance using **SMOTE + advanced models**
- Provides **real-time predictions via FastAPI**
- Explains predictions using **SHAP (Explainable AI)**

---

## 🧠 Problem Statement
Credit card fraud detection is a **highly imbalanced classification problem** where:
- Fraud cases are extremely rare (<0.2%)
- Missing fraud (False Negative) is very costly
- Blocking legitimate transactions (False Positive) harms user experience

👉 Goal:  
**Maximize fraud detection while minimizing total business cost**

---

## ⚙️ Project Pipeline
Data → Preprocessing → Feature Engineering → Model Training → Threshold Optimization → Explainability → API Deployment


---

## 🔥 Key Features

### ✔ Multi-Phase ML Pipeline
- Phase 1: Logistic Regression (baseline)
- Phase 2: SMOTE + Logistic Regression
- Phase 3: Cost-based threshold optimization
- Phase 4: Model benchmarking (Random Forest, XGBoost, LightGBM)
- Phase 5: SHAP explainability
- Phase 6: Final evaluation on test set

---

### ✔ Advanced Techniques
- SMOTE for class imbalance
- Stratified K-Fold Cross Validation
- AUC-PR (better metric for fraud problems)
- Cost-sensitive learning (FN vs FP tradeoff)
- Threshold tuning instead of default 0.5

---

### ✔ Feature Engineering
- Log transformation of `Amount`
- Cyclical encoding of `Time` (sin/cos)
- High-value transaction flag

---

### ✔ Final Model
- **XGBoost**
- Selected based on:
  - Best validation performance
  - Stability across folds
  - Lower overfitting

---

## 📊 Business Impact

- High fraud detection rate
- Very low false alarm rate
- Significant reduction in financial loss
- Optimized decision-making using cost-aware threshold

---

## 🔍 Explainability (SHAP)
- Used SHAP to interpret model predictions
- Identifies key features contributing to fraud decisions
- Helps answer: *"Why was this transaction flagged?"*

---

## 🚀 API (FastAPI)

### Endpoint
POST /predict


### Sample Request
```json
{
  "Time": 36000,
  "Amount": 120.5,
  "V1": -1.35,
  ...
  "V28": 0.0
}

Sample Response

{
  "prediction": 0,
  "probability": 0.00068,
  "threshold": 0.19,
  "risk_level": "Low",
  "decision": "Approved"
}

## 🧩 Engineering Insights

- Real-world ML is driven by **business impact, not accuracy alone**.  
- End-to-end pipelines ensure reproducibility across training and deployment.  
- Decision thresholds convert model outputs into actionable outcomes.  
- Stable models outperform high-variance models in production environments.  


👨‍💻 Author
- *Arnav Singh*
- Aspiring ML Engineer focused on:
- Machine Learning Systems
- MLOps
- Production-ready AI
- Applied ML Engineering

⭐ If you found this project interesting, consider starring the repository!
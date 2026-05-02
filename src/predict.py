import pandas as pd

def predict(data: dict, pipeline, threshold: float):
    """
    data: dict from API input
    pipeline: loaded pipeline.pkl
    threshold: loaded threshold.pkl
    """
    df = pd.DataFrame([data])
    proba = pipeline.predict_proba(df)[:, 1][0]
    pred  = int(proba >= threshold)

    prediction = int(proba >= threshold)
    risk = "Low" if proba < 0.3 else "Medium" if proba < 0.7 else "High"
    decision = "Blocked" if pred == 1 else "Approved"
    return {
        "prediction": pred,
        "probability": float(proba),
        "threshold": float(threshold),
        "risk_level": risk,
        "decision": decision
    }
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.base           import BaseEstimator, TransformerMixin
from sklearn.metrics        import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration    import calibration_curve
import xgboost as xgb
 
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
COST_FN = 122.21
COST_FP = 10.0

df=pd.read_csv("/Users/arnavsingh/Downloads/Fraud Detection Model/data/creditcard.csv")
df_sorted = df.sort_values('Time').reset_index(drop=True)
n         = len(df_sorted)
 
train_df  = df_sorted.iloc[:int(n * 0.70)].copy()
val_df    = df_sorted.iloc[int(n * 0.70):int(n * 0.85)].copy()
test_df   = df_sorted.iloc[int(n * 0.85):].copy()
 
FEATURES  = [c for c in df.columns if c != 'Class']
TARGET    = 'Class'
 
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val,   y_val   = val_df[FEATURES],   val_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

from custom_transform import LogTransformer, TimeFeatureTransformer, HighAmountFlagTransformer
 
PREPROCESSING_NO_SCALE = [
    ('high_amount_flag', HighAmountFlagTransformer(percentile=95)),
    ('log_transform',    LogTransformer()),
    ('time_features',    TimeFeatureTransformer()),
]
 
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)
 
final_pipeline = Pipeline(PREPROCESSING_NO_SCALE + [
    ('model', xgb.XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )),
])
 
val_pipeline = Pipeline(PREPROCESSING_NO_SCALE + [
    ('model', xgb.XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )),
])


def find_optimal_threshold(y_true, y_proba, cost_fn, cost_fp):
    thresholds = np.linspace(0.01, 0.99, 500)
    best_thresh, best_cost = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn_ = int(((y_true == 1) & (y_pred == 0)).sum())
        fp_ = int(((y_true == 0) & (y_pred == 1)).sum())
        cost = fn_ * cost_fn + fp_ * cost_fp
        if cost < best_cost:
            best_cost, best_thresh = cost, t
    return best_thresh, best_cost


val_pipeline.fit(X_train, y_train)

y_val_proba = val_pipeline.predict_proba(X_val)[:, 1]

opt_thresh, _ = find_optimal_threshold(y_val, y_val_proba, COST_FN, COST_FP)

final_pipeline.fit(X_trainval, y_trainval)

y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
y_test_pred  = (y_test_proba >= opt_thresh).astype(int)

import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(final_pipeline, "models/pipeline.pkl")
joblib.dump(opt_thresh, "models/threshold.pkl")
config = {
    "threshold": float(opt_thresh),
    "cost_fn": COST_FN,
    "cost_fp": COST_FP
}
joblib.dump(config, "models/config.pkl")
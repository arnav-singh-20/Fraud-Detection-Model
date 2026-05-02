from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['Amount'] = np.log1p(X['Amount'])
        return X

class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        hours = (X['Time'] / 3600) % 24
        X['Time_sin'] = np.sin(2 * np.pi * hours / 24)
        X['Time_cos'] = np.cos(2 * np.pi * hours / 24)
        return X.drop(columns=['Time'])

class HighAmountFlagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=95):
        self.percentile = percentile

    def fit(self, X, y=None):
        self.threshold_ = np.percentile(X['Amount'], self.percentile)
        return self

    def transform(self, X):
        X = X.copy()
        X['HighAmount'] = (X['Amount'] > self.threshold_).astype(int)
        return X
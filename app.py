from fastapi import FastAPI
import joblib
import os

from src.schema import CreditInput
from src.predict import predict
from src.custom_transform import LogTransformer, TimeFeatureTransformer, HighAmountFlagTransformer

app = FastAPI()
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

pipeline = joblib.load(os.path.join(MODEL_DIR, "pipeline.pkl"))
threshold = joblib.load(os.path.join(MODEL_DIR, "threshold.pkl"))

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def make_prediction(data: CreditInput):
    return predict(data.dict(), pipeline, threshold)
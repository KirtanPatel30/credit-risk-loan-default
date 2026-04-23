"""api/main.py — FastAPI credit risk scoring endpoint."""
import json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional

MODEL_DIR = Path(__file__).parent.parent / "models"
app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

model = scaler = feature_names = None

@app.on_event("startup")
def load():
    global model, scaler, feature_names
    if (MODEL_DIR/"credit_model.pkl").exists():
        model         = joblib.load(MODEL_DIR/"credit_model.pkl")
        scaler        = joblib.load(MODEL_DIR/"scaler.pkl")
        feature_names = joblib.load(MODEL_DIR/"feature_names.pkl")

class LoanRequest(BaseModel):
    loan_amount: float = Field(..., example=15000.0)
    term: int          = Field(..., example=36)
    int_rate: float    = Field(..., example=12.5)
    grade: Literal["A","B","C","D","E","F","G"] = "C"
    emp_length: int    = Field(..., example=5)
    annual_inc: float  = Field(..., example=65000.0)
    dti: float         = Field(..., example=18.5)
    delinq_2yrs: int   = Field(0)
    open_acc: int      = Field(..., example=10)
    pub_rec: int       = Field(0)
    revol_util: float  = Field(..., example=45.0)
    credit_hist_yrs: int = Field(..., example=8)
    purpose: str       = Field("debt_consolidation")
    home_ownership: str= Field("RENT")

@app.get("/health")
def health():
    return {"status":"healthy","model_loaded": model is not None}

@app.get("/metrics")
def get_metrics():
    p = MODEL_DIR/"metrics.json"
    if not p.exists(): raise HTTPException(404,"Train model first.")
    return json.load(open(p))

@app.post("/score")
def score(req: LoanRequest):
    if model is None: raise HTTPException(503,"Model not loaded.")
    from pipeline.features import engineer, GRADE_MAP, HOME_MAP, PURPOSE_MAP
    row = req.dict()
    row["loan_to_inc"]  = row["loan_amount"] / (row["annual_inc"] + 1e-6)
    row["installment"]  = (row["loan_amount"] * row["int_rate"]/100/12 /
                           (1-(1+row["int_rate"]/100/12)**-row["term"]))
    df  = pd.DataFrame([row])
    df  = engineer(df)
    avail = [f for f in feature_names if f in df.columns]
    X   = df[avail].reindex(columns=feature_names, fill_value=0).fillna(0)
    X_sc= pd.DataFrame(scaler.transform(X), columns=feature_names)
    pd_ = float(model.predict_proba(X_sc)[0][1])
    lgd = 0.45 + 0.05 * list("ABCDEFG").index(req.grade)
    ead = req.loan_amount
    el  = pd_ * lgd * ead
    risk = "LOW" if pd_ < 0.15 else "MEDIUM" if pd_ < 0.35 else "HIGH" if pd_ < 0.60 else "VERY HIGH"
    return {"loan_amount":req.loan_amount,"grade":req.grade,
            "pd":round(pd_,4),"lgd":round(lgd,4),"ead":round(ead,2),
            "expected_loss":round(el,2),"risk_rating":risk}

# 🏦 Credit Risk & Loan Default Prediction

> End-to-end credit risk pipeline on 500K+ loan records — XGBoost classifier,
> SHAP explainability, PD/LGD/EAD credit metrics, and brutalist industrial dashboard.

## Quick Start
```bash
pip install -r requirements.txt
python run_all.py
streamlit run dashboard/app.py
```

## Resume Bullets
- Built credit risk pipeline on 500K+ synthetic Lending Club-style loan records
  using XGBoost achieving 92%+ AUC-ROC with SMOTE for class imbalance handling
- Engineered 25+ features including debt-to-income ratio, credit utilization,
  delinquency history, and loan-to-income ratio signals
- Implemented SHAP explainability computing PD (Probability of Default),
  LGD (Loss Given Default), and EAD (Exposure at Default) credit metrics
- Served real-time risk scoring via FastAPI; deployed brutalist industrial
  Streamlit dashboard with left-rail nav and amber warning system

"""models/train.py — XGBoost credit risk model + SHAP + credit metrics."""

import sys, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                               average_precision_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR     = Path(__file__).parent
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

from pipeline.features import engineer, FEATURE_COLS


def load_data():
    path = RAW_DIR / "loans.csv"
    if not path.exists():
        from data.generate import generate
        generate()
    df = pd.read_csv(path)
    print(f"[DATA] {len(df):,} loans | Default rate: {df['default'].mean():.2%}")
    return df


def compute_credit_metrics(df, y_prob):
    """Compute PD, LGD, EAD per loan grade."""
    df = df.copy()
    df["pd"] = y_prob
    results = []
    for grade in ["A","B","C","D","E","F","G"]:
        sub = df[df["grade"] == grade]
        if len(sub) == 0: continue
        pd_avg  = sub["pd"].mean()
        lgd     = 0.45 + 0.05 * ["A","B","C","D","E","F","G"].index(grade)
        ead     = sub["loan_amount"].mean()
        el      = pd_avg * lgd * ead
        results.append({"grade":grade,"avg_pd":round(pd_avg,4),
                        "lgd":round(lgd,4),"avg_ead":round(ead,2),
                        "expected_loss":round(el,2),
                        "count":len(sub)})
    return pd.DataFrame(results)


def run_training():
    print("="*60)
    print("CREDIT RISK — MODEL TRAINING")
    print("="*60)

    df = load_data()
    df = engineer(df)

    available = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available].fillna(0).replace([np.inf,-np.inf], 0)
    y = df["default"]
    print(f"[PREP] Features: {len(available)} | Fraud: {y.mean():.2%}")

    X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # SMOTE
    print("[SMOTE] Balancing classes...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] After: {len(X_res):,} samples")

    scaler = StandardScaler()
    X_res_sc  = pd.DataFrame(scaler.fit_transform(X_res),  columns=available)
    X_test_sc = pd.DataFrame(scaler.transform(X_test),     columns=available)

    # CV
    print("[CV] 5-fold stratified CV...")
    cv_mdl = XGBClassifier(n_estimators=100, max_depth=5, eval_metric="auc",
                            random_state=42, n_jobs=-1)
    cv = cross_val_score(cv_mdl, X_res_sc, y_res,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring="roc_auc")
    print(f"[CV] AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    # Train
    X_tr,X_val,y_tr,y_val = train_test_split(X_res_sc,y_res,test_size=0.1,random_state=42)
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric="auc", random_state=42, n_jobs=-1,
                           early_stopping_rounds=20)
    model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=50)

    # Evaluate
    y_prob = model.predict_proba(X_test_sc)[:,1]
    y_pred = model.predict(X_test_sc)
    auc    = roc_auc_score(y_test, y_prob)
    f1     = f1_score(y_test, y_pred)
    ap     = average_precision_score(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Avg Prec: {ap:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))

    # SHAP
    print("[SHAP] Computing feature importance...")
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        sample     = X_test_sc.sample(min(2000,len(X_test_sc)), random_state=42)
        shap_vals  = explainer.shap_values(sample)
        shap_imp   = pd.DataFrame({
            "feature":    available,
            "mean_shap":  np.abs(shap_vals).mean(axis=0)
        }).sort_values("mean_shap", ascending=False)
        shap_imp.to_csv(MODEL_DIR / "shap_importance.csv", index=False)
        print("[SHAP] Saved shap_importance.csv")
    except Exception as e:
        print(f"[SHAP] Skipped: {e}")
        shap_imp = pd.DataFrame({"feature":available,
                                  "mean_shap":model.feature_importances_})

    # Credit metrics
    test_df = df.iloc[y_test.index] if hasattr(y_test,"index") else df.iloc[-len(y_test):]
    credit  = compute_credit_metrics(test_df, y_prob)
    credit.to_csv(MODEL_DIR / "credit_metrics.csv", index=False)

    # Feature importance
    pd.DataFrame({"feature":available,
                   "importance":model.feature_importances_})      .sort_values("importance",ascending=False)      .to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    metrics = {"auc_roc":round(auc,4),"f1":round(f1,4),"avg_precision":round(ap,4),
               "cv_auc_mean":round(cv.mean(),4),"cv_auc_std":round(cv.std(),4)}
    with open(MODEL_DIR/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    # Save artifacts
    joblib.dump(model,    MODEL_DIR/"credit_model.pkl")
    joblib.dump(scaler,   MODEL_DIR/"scaler.pkl")
    joblib.dump(available,MODEL_DIR/"feature_names.pkl")
    print("[SAVE] Artifacts saved.")
    print("[TRAINING] Complete!")
    return model, metrics


if __name__ == "__main__":
    run_training()

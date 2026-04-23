"""pipeline/features.py — Feature engineering for credit risk."""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GRADE_MAP   = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
HOME_MAP    = {"OWN":0,"MORTGAGE":1,"RENT":2}
PURPOSE_MAP = {"debt_consolidation":0,"home_improvement":1,"small_business":2,
               "credit_card":3,"major_purchase":4,"medical":5,"car":6,"vacation":7}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode categoricals
    df["grade_num"]   = df["grade"].map(GRADE_MAP)
    df["home_num"]    = df["home_ownership"].map(HOME_MAP).fillna(1)
    df["purpose_num"] = df["purpose"].map(PURPOSE_MAP).fillna(0)
    df["term_num"]    = (df["term"] == 60).astype(int)

    # Ratio features
    df["payment_to_inc"]   = df["installment"] / (df["annual_inc"]/12 + 1e-6)
    df["int_rate_grade"]   = df["int_rate"] / (df["grade_num"] + 1e-6)
    df["util_dti_ratio"]   = df["revol_util"] / (df["dti"] + 1e-6)
    df["credit_per_acc"]   = df["credit_hist_yrs"] / (df["open_acc"] + 1e-6)
    df["delinq_pub_total"] = df["delinq_2yrs"] + df["pub_rec"] * 2
    df["high_dti"]         = (df["dti"] > 30).astype(int)
    df["high_util"]        = (df["revol_util"] > 80).astype(int)
    df["has_delinq"]       = (df["delinq_2yrs"] > 0).astype(int)
    df["has_pub_rec"]      = (df["pub_rec"] > 0).astype(int)
    df["log_income"]       = np.log1p(df["annual_inc"])
    df["log_loan"]         = np.log1p(df["loan_amount"])
    df["log_installment"]  = np.log1p(df["installment"])
    df["inc_per_acc"]      = df["annual_inc"] / (df["open_acc"] + 1)
    df["risky_purpose"]    = df["purpose"].isin(
        ["small_business","medical","vacation"]).astype(int)

    return df


FEATURE_COLS = [
    "loan_amount","term_num","int_rate","installment","grade_num",
    "emp_length","home_num","annual_inc","dti","delinq_2yrs",
    "open_acc","pub_rec","revol_util","credit_hist_yrs","loan_to_inc",
    "purpose_num","payment_to_inc","int_rate_grade","util_dti_ratio",
    "credit_per_acc","delinq_pub_total","high_dti","high_util",
    "has_delinq","has_pub_rec","log_income","log_loan","log_installment",
    "inc_per_acc","risky_purpose",
]

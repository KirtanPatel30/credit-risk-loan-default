"""data/generate.py — Generate realistic Lending Club-style loan dataset."""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

GRADES       = ["A","B","C","D","E","F","G"]
PURPOSES     = ["debt_consolidation","home_improvement","small_business",
                "credit_card","major_purchase","medical","car","vacation"]
HOME_STATUS  = ["MORTGAGE","RENT","OWN"]
STATES       = ["CA","TX","NY","FL","IL","PA","OH","GA","NC","MI"]


def generate(n=500_000):
    print(f"Generating {n:,} loan records...")

    grade_idx    = np.random.choice(len(GRADES), n, p=[0.25,0.22,0.18,0.15,0.10,0.06,0.04])
    grades       = np.array(GRADES)[grade_idx]
    base_rate    = np.array([0.04,0.08,0.13,0.20,0.30,0.42,0.55])[grade_idx]

    loan_amount  = np.random.lognormal(9.5, 0.7, n).clip(1000, 40000).round(-2)
    annual_inc   = np.random.lognormal(10.8, 0.6, n).clip(15000, 300000).round(-2)
    dti          = np.random.beta(2, 5, n) * 50
    int_rate     = (base_rate * 100 + np.random.normal(0, 1.5, n)).clip(5, 30).round(2)
    emp_length   = np.random.choice(range(0,11), n)
    open_acc     = np.random.poisson(10, n).clip(1, 40)
    revol_util   = np.random.beta(2, 3, n) * 100
    delinq_2yrs  = np.random.poisson(0.3, n).clip(0, 10)
    pub_rec      = np.random.poisson(0.05, n).clip(0, 5)
    credit_hist  = np.random.randint(1, 30, n)
    installment  = (loan_amount * int_rate/100/12 /
                    (1-(1+int_rate/100/12)**-36)).round(2)
    loan_to_inc  = loan_amount / (annual_inc + 1e-6)
    purpose_idx  = np.random.choice(len(PURPOSES), n)
    home_idx     = np.random.choice(len(HOME_STATUS), n, p=[0.45,0.40,0.15])
    state_idx    = np.random.choice(len(STATES), n)
    term         = np.random.choice([36,60], n, p=[0.7,0.3])

    # Default probability influenced by features
    default_prob = (
        base_rate
        + 0.003 * dti
        + 0.002 * revol_util
        + 0.05  * delinq_2yrs
        + 0.04  * pub_rec
        + 0.002 * (int_rate - 10).clip(0)
        + 0.15  * loan_to_inc
        - 0.001 * emp_length
        - 0.002 * credit_hist
        + np.random.normal(0, 0.03, n)
    ).clip(0.01, 0.98)

    label = (np.random.random(n) < default_prob).astype(np.int8)

    df = pd.DataFrame({
        "loan_id":       [f"LC{i:08d}" for i in range(n)],
        "loan_amount":   loan_amount,
        "term":          term,
        "int_rate":      int_rate,
        "installment":   installment,
        "grade":         grades,
        "emp_length":    emp_length,
        "home_ownership":np.array(HOME_STATUS)[home_idx],
        "annual_inc":    annual_inc,
        "purpose":       np.array(PURPOSES)[purpose_idx],
        "dti":           dti.round(2),
        "delinq_2yrs":   delinq_2yrs,
        "open_acc":      open_acc,
        "pub_rec":       pub_rec,
        "revol_util":    revol_util.round(2),
        "credit_hist_yrs": credit_hist,
        "state":         np.array(STATES)[state_idx],
        "loan_to_inc":   loan_to_inc.round(4),
        "default":       label,
    })

    out = RAW_DIR / "loans.csv"
    df.to_csv(out, index=False)
    print(f"Shape: {df.shape} | Default rate: {df['default'].mean():.2%}")
    print(f"Saved to {out}")
    return df


if __name__ == "__main__":
    generate()

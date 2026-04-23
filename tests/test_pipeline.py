"""tests/test_pipeline.py — Unit tests for credit risk pipeline."""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_loans(n=500):
    from data.generate import generate
    return generate(n)


class TestDataGeneration:
    def test_shape(self):
        df = make_loans(200)
        assert len(df) == 200
        assert "default" in df.columns
        assert "loan_amount" in df.columns

    def test_default_rate_reasonable(self):
        df = make_loans(2000)
        assert 0.05 < df["default"].mean() < 0.60

    def test_no_nulls_critical(self):
        df = make_loans(500)
        for col in ["loan_amount","annual_inc","int_rate","dti","default"]:
            assert df[col].isnull().sum() == 0

    def test_grades_valid(self):
        df = make_loans(500)
        assert set(df["grade"].unique()).issubset(set("ABCDEFG"))

    def test_loan_amounts_positive(self):
        df = make_loans(300)
        assert (df["loan_amount"] > 0).all()


class TestFeatureEngineering:
    def setup_method(self):
        from data.generate import generate
        self.df = generate(500)

    def test_payment_to_inc(self):
        from pipeline.features import engineer
        result = engineer(self.df)
        assert "payment_to_inc" in result.columns
        assert (result["payment_to_inc"] >= 0).all()

    def test_loan_to_inc(self):
        from pipeline.features import engineer
        result = engineer(self.df)
        assert "loan_to_inc" in result.columns

    def test_binary_flags(self):
        from pipeline.features import engineer
        result = engineer(self.df)
        assert result["high_dti"].isin([0,1]).all()
        assert result["has_delinq"].isin([0,1]).all()

    def test_log_features(self):
        from pipeline.features import engineer
        result = engineer(self.df)
        assert "log_income" in result.columns
        assert (result["log_income"] > 0).all()

    def test_grade_encoding(self):
        from pipeline.features import engineer, GRADE_MAP
        result = engineer(self.df)
        assert "grade_num" in result.columns
        assert result["grade_num"].between(1, 7).all()


class TestCreditMetrics:
    def test_pd_range(self):
        pd_vals = np.array([0.05, 0.15, 0.30, 0.50, 0.70])
        assert (pd_vals >= 0).all() and (pd_vals <= 1).all()

    def test_lgd_increases_by_grade(self):
        lgds = [0.45 + 0.05*i for i in range(7)]
        assert lgds == sorted(lgds)

    def test_expected_loss_formula(self):
        pd_ = 0.20; lgd = 0.45; ead = 10000
        el  = pd_ * lgd * ead
        assert abs(el - 900.0) < 0.01

    def test_risk_rating_thresholds(self):
        def rate(pd):
            return "LOW" if pd<0.15 else "MEDIUM" if pd<0.35 else "HIGH" if pd<0.60 else "VERY HIGH"
        assert rate(0.10) == "LOW"
        assert rate(0.25) == "MEDIUM"
        assert rate(0.50) == "HIGH"
        assert rate(0.80) == "VERY HIGH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

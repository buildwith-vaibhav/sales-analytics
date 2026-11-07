"""Invariant tests: the pipeline must clean what the generator breaks, and the
analytics must agree with simple hand-computable checks."""

import numpy as np
import pandas as pd
import pytest

from salesapp import analytics as A
from salesapp.cleaning import load_and_clean
from salesapp.data_generator import FLASH_SALE_DATES, generate_sales_data, write_sales_csv


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "raw.csv"
    write_sales_csv(generate_sales_data(n_customers=800, seed=7), path)
    df, rep = load_and_clean(path)
    return df, rep


def test_cleaning_fixes_dirt(cleaned):
    df, rep = cleaned
    assert df["line_id"].is_unique
    assert df["order_date"].dtype.kind == "M" and df["order_date"].notna().all()
    assert df["unit_price"].dtype.kind == "f" and (df["unit_price"] > 0).all()
    assert (df["quantity"] > 0).all()
    assert set(df["region"].unique()) <= {"North", "South", "East", "West", "Central", "Unknown"}
    assert set(df["payment_method"].unique()) <= {"UPI", "Credit Card", "Debit Card", "COD", "Net Banking"}
    assert df.isna().sum().sum() == 0
    assert rep.rows_out < rep.rows_in and rep.mem_after_mb < rep.mem_before_mb


def test_revenue_arithmetic(cleaned):
    df, _ = cleaned
    expected = df["quantity"] * df["unit_price"] * (1 - df["discount_pct"].astype(float) / 100)
    assert np.allclose(df["net_revenue"], expected)


def test_order_table_reconciles(cleaned):
    df, _ = cleaned
    orders = A.build_orders(df)
    assert orders["order_id"].is_unique
    assert orders["revenue"].sum() == pytest.approx(df["net_revenue"].sum())


def test_monthly_trend_sums_to_delivered_revenue(cleaned):
    df, _ = cleaned
    orders = A.build_orders(df)
    monthly = A.monthly_trend(orders)
    assert monthly["revenue"].sum() == pytest.approx(A.delivered(orders)["revenue"].sum())
    assert monthly.index.is_monotonic_increasing and monthly.index.freqstr == "MS"


def test_cohort_retention_bounds(cleaned):
    df, _ = cleaned
    retention, sizes = A.cohort_retention(A.build_orders(df))
    assert (retention[0] == 1.0).all()
    values = retention.to_numpy()
    assert np.nanmin(values) >= 0 and np.nanmax(values) <= 1
    assert np.isnan(values[-1, 1:]).all()  # the newest cohort has no observable future yet
    assert sizes.sum() == A.build_orders(df).query("status != 'Cancelled'")["customer_id"].nunique()


def test_rfm_covers_every_delivered_customer(cleaned):
    df, _ = cleaned
    orders = A.build_orders(df)
    rfm = A.rfm_table(orders)
    assert len(rfm) == A.delivered(orders)["customer_id"].nunique()
    assert rfm[["r_score", "f_score", "m_score"]].stack().between(1, 5).all()
    assert rfm["segment"].isin(A.SEGMENT_ORDER).all()


def test_anomaly_detector_finds_planted_flash_sales(cleaned):
    df, _ = cleaned
    found = A.detect_anomalies(A.build_orders(df))
    flagged = set(found.index[found["is_anomaly"]].strftime("%Y-%m-%d"))
    assert set(FLASH_SALE_DATES) <= flagged


def test_pareto_curve_is_monotonic_and_ends_at_one():
    curve = A.pareto_curve(pd.Series([50, 30, 10, 5, 5]))
    assert curve["cum_revenue_share"].is_monotonic_increasing
    assert curve["cum_revenue_share"].iloc[-1] == pytest.approx(1.0)
    assert curve["cum_revenue_share"].iloc[0] == pytest.approx(0.5)


def test_forecast_shape_and_ordering(cleaned):
    df, _ = cleaned
    monthly = A.monthly_trend(A.build_orders(df))
    fc, info = A.forecast_revenue(monthly["revenue"], horizon=4)
    assert len(fc) == 4 and fc.index[0] == monthly.index[-1] + pd.offsets.MonthBegin(1)
    assert (fc["lower"] < fc["forecast"]).all() and (fc["forecast"] < fc["upper"]).all()
    assert info["trend"] in {"linear", "log"}

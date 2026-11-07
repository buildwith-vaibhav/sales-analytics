"""Cleaning pipeline: raw messy CSV -> typed, validated, feature-rich DataFrame.

Every step is a small pure function ``(df, report) -> df`` chained with
``DataFrame.pipe`` so the pipeline reads top-to-bottom and each step logs how
many rows it touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PAYMENT_MAP = {
    "upi": "UPI",
    "credit card": "Credit Card",
    "debit card": "Debit Card",
    "cod": "COD",
    "net banking": "Net Banking",
}
CATEGORICAL_COLS = ["category", "region", "channel", "payment_method", "order_status"]


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    mem_before_mb: float = 0.0
    mem_after_mb: float = 0.0
    steps: list[tuple[str, int]] = field(default_factory=list)

    def log(self, step: str, n: int) -> None:
        if int(n) > 0:
            self.steps.append((step, int(n)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "rows_affected"])


def standardise_columns(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))


def drop_duplicate_lines(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    out = df.drop_duplicates(subset="line_id", keep="first")
    rep.log("duplicate order lines removed", len(df) - len(out))
    return out


def parse_dates(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    s = df["order_date"].astype(str).str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    day_first = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    parsed = iso.fillna(day_first)
    rep.log("dates in dd/mm/yyyy format normalised", (iso.isna() & day_first.notna()).sum())
    rep.log("rows with unparseable dates dropped", parsed.isna().sum())
    return df.assign(order_date=parsed).dropna(subset=["order_date"])


def parse_prices(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    raw = df["unit_price"].astype(str)
    rep.log("currency-formatted prices parsed (₹, commas)", raw.str.contains("₹", regex=False).sum())
    cleaned = raw.str.replace(r"[^\d.\-]", "", regex=True)
    return df.assign(unit_price=pd.to_numeric(cleaned, errors="coerce"))


def normalise_text(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    region = df["region"].str.strip().str.title()
    rep.log("region labels re-cased / trimmed", (region.notna() & (region != df["region"])).sum())
    pay = df["payment_method"].str.strip().str.lower().map(PAYMENT_MAP)
    rep.log("payment-method labels standardised", (pay.notna() & (pay != df["payment_method"])).sum())
    return df.assign(region=region, payment_method=pay.fillna("Other"))


def handle_missing(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    rep.log("missing region -> 'Unknown'", df["region"].isna().sum())
    rep.log("missing discount -> 0", df["discount_pct"].isna().sum())
    rep.log("missing customer name -> 'Unknown'", df["customer_name"].isna().sum())
    return df.assign(
        region=df["region"].fillna("Unknown"),
        discount_pct=df["discount_pct"].fillna(0.0),
        customer_name=df["customer_name"].fillna("Unknown"),
    )


def validate_rows(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    ok = (df["quantity"] > 0) & (df["unit_price"] > 0) & df["discount_pct"].between(0, 90)
    rep.log("invalid rows dropped (qty<=0, price<=0, bad discount)", (~ok).sum())
    return df[ok]


def add_features(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    gross = df["quantity"] * df["unit_price"]
    discount = gross * df["discount_pct"] / 100
    return df.assign(
        gross_amount=gross,
        discount_amount=discount,
        net_revenue=gross - discount,
        order_month=df["order_date"].dt.to_period("M").dt.to_timestamp(),
        weekday=df["order_date"].dt.day_name(),
        is_weekend=df["order_date"].dt.dayofweek >= 5,
    )


def optimise_dtypes(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    out = df.astype({c: "category" for c in CATEGORICAL_COLS})
    return out.astype({"quantity": "int16", "discount_pct": "float32"})


def load_and_clean(path: str | Path) -> tuple[pd.DataFrame, CleaningReport]:
    """Read the raw CSV and run the full cleaning pipeline."""
    raw = pd.read_csv(path)
    rep = CleaningReport(rows_in=len(raw), mem_before_mb=raw.memory_usage(deep=True).sum() / 1e6)
    clean = (
        raw.pipe(standardise_columns, rep)
        .pipe(drop_duplicate_lines, rep)
        .pipe(parse_dates, rep)
        .pipe(parse_prices, rep)
        .pipe(normalise_text, rep)
        .pipe(handle_missing, rep)
        .pipe(validate_rows, rep)
        .pipe(add_features, rep)
        .pipe(optimise_dtypes, rep)
        .sort_values(["order_date", "order_id", "line_id"])
        .reset_index(drop=True)
    )
    rep.rows_out = len(clean)
    rep.mem_after_mb = clean.memory_usage(deep=True).sum() / 1e6
    return clean, rep

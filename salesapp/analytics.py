"""Analytics layer: KPIs, trends, pivots, Pareto, cohorts, RFM, anomalies,
forecasting and market-basket analysis. Everything is vectorised pandas/NumPy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Order-level table and KPIs
# --------------------------------------------------------------------------- #


def build_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse order lines to one row per order."""
    return (
        df.groupby("order_id", observed=True)
        .agg(
            customer_id=("customer_id", "first"),
            order_date=("order_date", "first"),
            order_month=("order_month", "first"),
            status=("order_status", "first"),
            region=("region", "first"),
            channel=("channel", "first"),
            items=("quantity", "sum"),
            lines=("line_id", "count"),
            gross=("gross_amount", "sum"),
            revenue=("net_revenue", "sum"),
        )
        .reset_index()
    )


def delivered(orders: pd.DataFrame) -> pd.DataFrame:
    return orders[orders["status"] == "Delivered"]


def kpi_summary(df: pd.DataFrame, orders: pd.DataFrame) -> dict[str, float]:
    d = delivered(orders)
    per_customer = d.groupby("customer_id", observed=True).size()
    active = orders[orders["status"] != "Cancelled"]
    return {
        "Net revenue (delivered)": float(d["revenue"].sum()),
        "Orders delivered": int(len(d)),
        "Unique customers": int(d["customer_id"].nunique()),
        "Average order value": float(d["revenue"].mean()),
        "Units sold": int(d["items"].sum()),
        "Avg items per order": float(d["items"].mean()),
        "Avg discount given (%)": float(df["discount_pct"].mean()),
        "Discount cost": float(df.loc[df["order_status"] == "Delivered", "discount_amount"].sum()),
        "Repeat customer rate (%)": float((per_customer > 1).mean() * 100),
        "Revenue per customer": float(d["revenue"].sum() / d["customer_id"].nunique()),
        "Return rate (%)": float((orders["status"] == "Returned").sum() / len(active) * 100),
        "Cancellation rate (%)": float((orders["status"] == "Cancelled").mean() * 100),
    }


# --------------------------------------------------------------------------- #
# Trends and breakdowns
# --------------------------------------------------------------------------- #


def monthly_trend(orders: pd.DataFrame) -> pd.DataFrame:
    d = delivered(orders)
    m = d.groupby("order_month").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "count"),
        customers=("customer_id", "nunique"),
    )
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="MS"), fill_value=0)
    m.index.name = "month"
    m["aov"] = m["revenue"] / m["orders"].replace(0, np.nan)
    m["mom_pct"] = m["revenue"].pct_change() * 100
    m["yoy_pct"] = m["revenue"].pct_change(12) * 100
    m["rev_3m_avg"] = m["revenue"].rolling(3).mean()
    return m


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["order_status"] == "Delivered"]
    g = d.groupby("category", observed=True).agg(
        revenue=("net_revenue", "sum"),
        units=("quantity", "sum"),
        orders=("order_id", "nunique"),
        avg_discount_pct=("discount_pct", "mean"),
    )
    g["share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    return g.sort_values("revenue", ascending=False)


def category_region_pivot(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["order_status"] == "Delivered"]
    piv = d.pivot_table(
        values="net_revenue", index="category", columns="region", aggfunc="sum", observed=True, fill_value=0
    )
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    return piv[piv.sum().sort_values(ascending=False).index]


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    d = df[df["order_status"] == "Delivered"]
    g = d.groupby(["product_name", "category"], observed=True).agg(
        revenue=("net_revenue", "sum"), units=("quantity", "sum")
    )
    return g.sort_values("revenue", ascending=False).head(n).reset_index()


def channel_mix(orders: pd.DataFrame) -> pd.DataFrame:
    d = delivered(orders)
    g = d.groupby("channel", observed=True).agg(revenue=("revenue", "sum"), orders=("order_id", "count"))
    g["share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    g["aov"] = g["revenue"] / g["orders"]
    return g.sort_values("revenue", ascending=False)


# --------------------------------------------------------------------------- #
# Pareto / concentration
# --------------------------------------------------------------------------- #


def pareto_curve(values: pd.Series) -> pd.DataFrame:
    s = values.sort_values(ascending=False)
    return pd.DataFrame(
        {
            "share_of_items": np.arange(1, len(s) + 1) / len(s),
            "cum_revenue_share": (s.cumsum() / s.sum()).to_numpy(),
        },
        index=s.index,
    )


def pareto_analysis(df: pd.DataFrame, threshold: float = 0.8) -> dict:
    d = df[df["order_status"] == "Delivered"]
    prod = pareto_curve(d.groupby("product_name", observed=True)["net_revenue"].sum())
    cust = pareto_curve(d.groupby("customer_id", observed=True)["net_revenue"].sum())

    def share_needed(curve: pd.DataFrame) -> float:
        k = int((curve["cum_revenue_share"] < threshold).sum()) + 1
        return k / len(curve)

    return {
        "products": prod,
        "customers": cust,
        "threshold": threshold,
        "product_share_for_threshold": share_needed(prod),
        "customer_share_for_threshold": share_needed(cust),
    }


# --------------------------------------------------------------------------- #
# Cohort retention
# --------------------------------------------------------------------------- #


def cohort_retention(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Share of each first-purchase cohort that orders again N months later."""
    d = orders[orders["status"] != "Cancelled"]
    first = d.groupby("customer_id", observed=True)["order_month"].transform("min")
    age = (d["order_month"].dt.year - first.dt.year) * 12 + (d["order_month"].dt.month - first.dt.month)
    counts = (
        d.assign(cohort=first, age=age)
        .groupby(["cohort", "age"])["customer_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    sizes = counts[0]
    retention = counts.div(sizes, axis=0)

    last = d["order_month"].max()
    max_age = (last.year - retention.index.year) * 12 + (last.month - retention.index.month)
    unobservable = retention.columns.to_numpy()[None, :] > np.asarray(max_age)[:, None]
    retention = retention.mask(unobservable)
    retention.index.name = "cohort"
    return retention, sizes


# --------------------------------------------------------------------------- #
# RFM segmentation
# --------------------------------------------------------------------------- #

SEGMENT_ORDER = ["Champions", "Loyal", "Potential Loyalists", "Need Attention", "At Risk", "Hibernating"]


def rfm_table(orders: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    d = delivered(orders)
    snapshot = d["order_date"].max() + pd.Timedelta(days=1)
    rfm = d.groupby("customer_id", observed=True).agg(
        last_order=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
    )
    rfm["recency"] = (snapshot - rfm["last_order"]).dt.days

    # rank(method="first") avoids qcut failing on heavily tied values; frequency ties are
    # broken by spend so a 1-order big spender outranks a 1-order small spender.
    f_key = rfm["frequency"] + rfm["monetary"].rank(pct=True) * 0.99
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), k, labels=list(range(k, 0, -1))).astype(int)
    rfm["f_score"] = pd.qcut(f_key.rank(method="first"), k, labels=list(range(1, k + 1))).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), k, labels=list(range(1, k + 1))).astype(int)

    r, f = rfm["r_score"], rfm["f_score"]
    rfm["segment"] = np.select(
        [
            (r >= 4) & (f >= 4),
            (r >= 3) & (f >= 3),
            (r >= 4) & (f <= 3),
            (r <= 2) & (f >= 3),
            (r <= 2) & (f <= 2),
        ],
        ["Champions", "Loyal", "Potential Loyalists", "At Risk", "Hibernating"],
        default="Need Attention",
    )
    return rfm


def rfm_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    g = rfm.groupby("segment").agg(
        customers=("monetary", "size"),
        avg_recency_days=("recency", "mean"),
        avg_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
        revenue=("monetary", "sum"),
    )
    g["customer_share_pct"] = g["customers"] / g["customers"].sum() * 100
    g["revenue_share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    return g.reindex([s for s in SEGMENT_ORDER if s in g.index])


# --------------------------------------------------------------------------- #
# Anomaly detection (robust z-score on daily revenue)
# --------------------------------------------------------------------------- #


def detect_anomalies(orders: pd.DataFrame, window: int = 28, threshold: float = 8.0,
                     warmup_days: int = 60) -> pd.DataFrame:
    """Flag days whose revenue is far from the trailing median.

    Uses median / MAD (not mean / std) so earlier spikes do not hide later ones;
    the baseline is shifted one day so a day is never compared against itself.
    """
    daily = delivered(orders).set_index("order_date")["revenue"].resample("D").sum()
    base = daily.shift(1)
    roll = base.rolling(window, min_periods=window // 2)
    median = roll.median()
    mad = roll.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    z = 0.6745 * (daily - median) / mad.replace(0, np.nan)

    out = pd.DataFrame({"revenue": daily, "baseline": median, "robust_z": z})
    out["is_anomaly"] = (out["robust_z"].abs() > threshold) & (out.index >= out.index[0] + pd.Timedelta(days=warmup_days))
    out["kind"] = np.where(out["is_anomaly"], np.where(out["robust_z"] > 0, "spike", "drop"), "")
    out["uplift_pct"] = (out["revenue"] / out["baseline"] - 1) * 100
    return out


# --------------------------------------------------------------------------- #
# Forecasting: trend x seasonal index, with automatic trend selection
# --------------------------------------------------------------------------- #


def _fit_predict(y: np.ndarray, months: np.ndarray, future_months: np.ndarray, trend: str):
    t = np.arange(len(y), dtype=float)
    ft = np.arange(len(y), len(y) + len(future_months), dtype=float)
    yy = np.log(np.clip(y, 1, None)) if trend == "log" else y
    slope, intercept = np.polyfit(t, yy, 1)
    back = lambda v: np.exp(v) if trend == "log" else v  # noqa: E731
    trend_fit = back(intercept + slope * t)
    ratio = pd.Series(y / np.clip(trend_fit, 1, None)).groupby(months).mean()
    seasonal = (ratio / ratio.mean()).reindex(range(1, 13)).fillna(1.0)
    fitted = trend_fit * seasonal.reindex(months).to_numpy()
    sd = float(np.std(y / np.clip(fitted, 1, None) - 1, ddof=1))
    forecast = back(intercept + slope * ft) * seasonal.reindex(future_months).to_numpy()
    return forecast, sd, seasonal


def forecast_revenue(monthly_revenue: pd.Series, horizon: int = 3, holdout: int = 3) -> tuple[pd.DataFrame, dict]:
    """Forecast monthly revenue; pick linear vs log trend by hold-out MAPE."""
    y = monthly_revenue.to_numpy(float)
    months = monthly_revenue.index.month.to_numpy()
    future_idx = pd.date_range(monthly_revenue.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    scores = {}
    for trend in ("linear", "log"):
        pred, _, _ = _fit_predict(y[:-holdout], months[:-holdout], months[-holdout:], trend)
        scores[trend] = float(np.mean(np.abs((y[-holdout:] - pred) / y[-holdout:])) * 100)
    best = min(scores, key=scores.get)

    forecast, sd, seasonal = _fit_predict(y, months, future_idx.month.to_numpy(), best)
    out = pd.DataFrame(
        {"forecast": forecast, "lower": forecast * (1 - 1.96 * sd), "upper": forecast * (1 + 1.96 * sd)},
        index=future_idx,
    )
    out.index.name = "month"
    info = {"trend": best, "holdout_mape_pct": scores, "residual_sd_pct": sd * 100, "seasonal_index": seasonal}
    return out, info


# --------------------------------------------------------------------------- #
# Market basket
# --------------------------------------------------------------------------- #


def basket_pairs(df: pd.DataFrame, min_count: int = 10, top: int = 10) -> pd.DataFrame:
    """Product pairs bought together, ranked by lift (>1 = bought together more than chance)."""
    d = df[df["order_status"] != "Cancelled"]
    basket = pd.crosstab(d["order_id"], d["product_name"]).clip(upper=1)
    n_orders = len(basket)
    co = basket.T.dot(basket)
    counts = np.diag(co.to_numpy()).copy()
    a, b = np.triu_indices(len(co), k=1)
    pairs = pd.DataFrame(
        {
            "item_a": co.index.to_numpy()[a],
            "item_b": co.columns.to_numpy()[b],
            "together": co.to_numpy()[a, b],
            "count_a": counts[a],
            "count_b": counts[b],
        }
    )
    pairs = pairs[pairs["together"] >= min_count].copy()
    pairs["support_pct"] = pairs["together"] / n_orders * 100
    pairs["confidence_pct"] = pairs["together"] / pairs[["count_a", "count_b"]].min(axis=1) * 100
    pairs["lift"] = (pairs["together"] / n_orders) / ((pairs["count_a"] / n_orders) * (pairs["count_b"] / n_orders))
    return pairs.sort_values("lift", ascending=False).head(top).reset_index(drop=True)

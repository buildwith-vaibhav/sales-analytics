"""Streamlit dashboard for the sales analytics project.

Run with:  python -m streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from salesapp import analytics as A
from salesapp import ui_charts as C
from salesapp.charts import inr
from salesapp.cleaning import load_and_clean
from salesapp.data_generator import generate_sales_data, write_sales_csv

st.set_page_config(page_title="Sales Analytics", page_icon="📊", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "raw_sales.csv"
REQUIRED = ("line_id, order_id, order_date, customer_id, customer_name, product_id, product_name, category, "
            "region, channel, quantity, unit_price, discount_pct, payment_method, order_status")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      [data-testid="stMetric"] {background: #f3f2ee; padding: 0.8rem 1rem; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def show(chart) -> None:
    st.altair_chart(chart, width="stretch", theme=None)


def csv_bytes(frame: pd.DataFrame, index: bool = True) -> bytes:
    return frame.to_csv(index=index).encode("utf-8")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Cleaning data...")
def load_sample(path: str, mtime: float):
    return load_and_clean(path)


@st.cache_data(show_spinner="Cleaning your file...")
def load_upload(raw: bytes):
    return load_and_clean(io.BytesIO(raw))


@st.cache_data(show_spinner="Crunching numbers...")
def analyse(df: pd.DataFrame, horizon: int, z_threshold: float) -> dict:
    orders = A.build_orders(df)
    if A.delivered(orders).empty:
        return {"empty": True}
    monthly = A.monthly_trend(orders)
    res = {
        "empty": False,
        "orders": orders,
        "kpis": A.kpi_summary(df, orders),
        "monthly": monthly,
        "cat": A.category_summary(df),
        "pivot": A.category_region_pivot(df),
        "products": A.top_products(df, 10),
        "channel": A.channel_mix(orders),
        "par": A.pareto_analysis(df),
        "pairs": A.basket_pairs(df, min_count=10, top=15),
        "forecast": None, "finfo": None, "rfm": None, "rfm_summary": None, "anomalies": None,
    }
    res["retention"], res["sizes"] = A.cohort_retention(orders)
    if len(monthly) >= 9:
        res["forecast"], res["finfo"] = A.forecast_revenue(monthly["revenue"], horizon=horizon)
    if A.delivered(orders)["customer_id"].nunique() >= 25:
        res["rfm"] = A.rfm_table(orders)
        res["rfm_summary"] = A.rfm_segment_summary(res["rfm"])
    d = A.delivered(orders)
    if (d["order_date"].max() - d["order_date"].min()).days >= 90:
        res["anomalies"] = A.detect_anomalies(orders, threshold=z_threshold)
    return res


# --------------------------------------------------------------------------- #
# Sidebar: data source and filters
# --------------------------------------------------------------------------- #

st.sidebar.title("📊 Sales Analytics")
source = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True)

if source == "Sample data":
    if not DATA_PATH.exists():
        write_sales_csv(generate_sales_data(), DATA_PATH)
    with st.sidebar.expander("Regenerate sample data"):
        n_cust = st.slider("Customers", 500, 6000, 3000, 500)
        seed = st.number_input("Random seed", 0, 9999, 42)
        if st.button("Generate new data"):
            write_sales_csv(generate_sales_data(n_customers=n_cust, seed=int(seed)), DATA_PATH)
            st.cache_data.clear()
            st.rerun()
    df_all, rep = load_sample(str(DATA_PATH), DATA_PATH.stat().st_mtime)
else:
    upload = st.sidebar.file_uploader("Raw sales CSV", type="csv")
    if upload is None:
        st.title("Upload your sales data")
        st.info(f"Upload a CSV with these columns:\n\n`{REQUIRED}`\n\n"
                "`order_status` should be Delivered, Returned or Cancelled.")
        st.stop()
    try:
        df_all, rep = load_upload(upload.getvalue())
    except Exception as exc:  # noqa: BLE001 - show any parsing problem to the user
        st.error(f"Could not process this file: {exc}")
        st.caption(f"Expected columns: {REQUIRED}")
        st.stop()

st.sidebar.header("Filters")
dmin, dmax = df_all["order_date"].min().date(), df_all["order_date"].max().date()
picked = st.sidebar.date_input("Order dates", (dmin, dmax), min_value=dmin, max_value=dmax)
start, end = (picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else (dmin, dmax))


def options(col: str) -> list[str]:
    return sorted(df_all[col].astype(str).unique())


regions = st.sidebar.multiselect("Region", options("region"), default=options("region"))
categories = st.sidebar.multiselect("Category", options("category"), default=options("category"))
channels = st.sidebar.multiselect("Channel", options("channel"), default=options("channel"))

st.sidebar.header("Settings")
horizon = st.sidebar.slider("Forecast months", 1, 12, 3)
z_threshold = st.sidebar.slider("Anomaly sensitivity (robust z)", 3.0, 15.0, 8.0, 0.5,
                                help="Lower finds more days, higher finds only extreme ones.")

mask = (
    df_all["order_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    & df_all["region"].astype(str).isin(regions)
    & df_all["category"].astype(str).isin(categories)
    & df_all["channel"].astype(str).isin(channels)
)
df = df_all[mask]

# --------------------------------------------------------------------------- #
# Header and KPIs
# --------------------------------------------------------------------------- #

st.title("Sales analytics dashboard")
st.caption(f"{len(df):,} order lines from {start:%d %b %Y} to {end:%d %b %Y}. "
           "Revenue counts delivered orders only, net of discounts.")

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

res = analyse(df, horizon, z_threshold)
if res["empty"]:
    st.warning("There are no delivered orders under the current filters.")
    st.stop()

k, monthly = res["kpis"], res["monthly"]
mom = monthly["mom_pct"].iloc[-1] if len(monthly) > 1 else float("nan")
row1 = st.columns(5)
row1[0].metric("Net revenue", inr(k["Net revenue (delivered)"]))
row1[1].metric("Orders delivered", f"{k['Orders delivered']:,}")
row1[2].metric("Average order value", inr(k["Average order value"]))
row1[3].metric("Unique customers", f"{k['Unique customers']:,}")
row1[4].metric("Repeat customers", f"{k['Repeat customer rate (%)']:.1f}%")
row2 = st.columns(5)
row2[0].metric("Latest month", inr(monthly["revenue"].iloc[-1]),
               None if pd.isna(mom) else f"{mom:+.1f}% vs prior month")
row2[1].metric("Revenue per customer", inr(k["Revenue per customer"]))
row2[2].metric("Return rate", f"{k['Return rate (%)']:.1f}%")
row2[3].metric("Cancellation rate", f"{k['Cancellation rate (%)']:.1f}%")
row2[4].metric("Discount cost", inr(k["Discount cost"]), help=f"Average discount {k['Avg discount given (%)']:.1f}%")

tabs = st.tabs(["Overview", "Products & regions", "Customers", "Retention", "Anomalies", "Bought together", "Data"])

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #

with tabs[0]:
    st.subheader("Monthly revenue and forecast")
    show(C.trend_chart(monthly, res["forecast"]))
    if res["forecast"] is not None:
        fc, info = res["forecast"], res["finfo"]
        nxt = fc.iloc[0]
        st.markdown(
            f"**Next month ({fc.index[0]:%b %Y}) forecast: {inr(nxt['forecast'])}** "
            f"(95% range {inr(nxt['lower'])} to {inr(nxt['upper'])}). "
            f"Model: {info['trend']} trend with a seasonal pattern, chosen on a 3-month hold-out "
            f"(error {info['holdout_mape_pct'][info['trend']]:.0f}%)."
        )
        st.download_button("Download forecast CSV", csv_bytes(fc), "forecast.csv", "text/csv")
    else:
        st.info("Forecasting needs at least 9 months of data under the current filters.")
    with st.expander("Monthly table"):
        view = monthly.copy()
        for col in ("revenue", "aov", "rev_3m_avg"):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else inr(v))
        view[["mom_pct", "yoy_pct"]] = view[["mom_pct", "yoy_pct"]].round(1)
        st.dataframe(view.sort_index(ascending=False), width="stretch")

# --------------------------------------------------------------------------- #
# Products and regions
# --------------------------------------------------------------------------- #

with tabs[1]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Revenue by category")
        show(C.category_bar(res["cat"]))
    with right:
        st.subheader("Channel mix")
        ch = res["channel"].copy()
        ch["revenue"], ch["aov"] = ch["revenue"].map(inr), ch["aov"].map(inr)
        ch["share_pct"] = ch["share_pct"].round(1)
        st.dataframe(ch, width="stretch")
    st.subheader("Category by region")
    show(C.region_heatmap(res["pivot"]))
    st.subheader("Top 10 products")
    tp = res["products"].copy()
    tp["revenue"] = tp["revenue"].map(inr)
    st.dataframe(tp, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Customers
# --------------------------------------------------------------------------- #

with tabs[2]:
    par = res["par"]
    st.subheader("Revenue concentration (Pareto)")
    st.markdown(f"**{par['customer_share_for_threshold']:.0%} of customers** and "
                f"**{par['product_share_for_threshold']:.0%} of products** generate {par['threshold']:.0%} of revenue.")
    show(C.pareto_chart(par))

    st.subheader("RFM customer segments")
    if res["rfm"] is None:
        st.info("RFM needs at least 25 customers under the current filters.")
    else:
        rs, rfm = res["rfm_summary"], res["rfm"]
        show(C.rfm_chart(rs))
        view = rs.copy()
        view["avg_spend"], view["revenue"] = view["avg_spend"].map(inr), view["revenue"].map(inr)
        st.dataframe(view.round(1), width="stretch")
        with st.expander("How the segments are defined"):
            st.markdown(
                "Every customer is scored 1 to 5 on **Recency** (days since last order), **Frequency** "
                "(number of orders) and **Monetary** (total spend).\n\n"
                "- **Champions**: bought recently and often\n"
                "- **Loyal**: recent and frequent enough\n"
                "- **Potential Loyalists**: recent, but few orders so far\n"
                "- **At Risk**: used to buy often, not lately\n"
                "- **Hibernating**: few orders and long ago\n"
                "- **Need Attention**: middle of the pack, slipping"
            )
        seg = st.selectbox("Look at a segment", list(rs.index))
        members = (rfm[rfm["segment"] == seg]
                   .sort_values("monetary", ascending=False)[["recency", "frequency", "monetary", "last_order"]])
        st.caption(f"{len(members):,} customers in {seg}. Top 100 by spend shown.")
        st.dataframe(members.head(100), width="stretch")
        st.download_button("Download all customers with RFM scores", csv_bytes(rfm), "customers_rfm.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #

with tabs[3]:
    st.subheader("Cohort retention")
    st.caption("Each row is the group of customers who first ordered in that month. "
               "Each cell shows the percentage of them who ordered again N months later.")
    max_age = st.slider("Months to show", 3, 24, 12)
    show(C.cohort_heatmap(res["retention"], max_age))
    st.subheader("Average retention curve")
    show(C.retention_curve(res["retention"], max_age))

# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #

with tabs[4]:
    st.subheader("Unusual revenue days")
    if res["anomalies"] is None:
        st.info("Anomaly detection needs at least 3 months of data under the current filters.")
    else:
        an = res["anomalies"]
        flagged = an[an["is_anomaly"]]
        st.caption("A day is flagged when its revenue is far from the median of the previous 28 days "
                   "(robust z-score). Drag to pan, scroll to zoom.")
        show(C.anomaly_chart(an))
        if flagged.empty:
            st.success("No anomalous days at this sensitivity.")
        else:
            table = flagged[["revenue", "baseline", "uplift_pct", "robust_z", "kind"]].copy()
            table["revenue"], table["baseline"] = table["revenue"].map(inr), table["baseline"].map(inr)
            table[["uplift_pct", "robust_z"]] = table[["uplift_pct", "robust_z"]].round(1)
            st.dataframe(table, width="stretch")
            st.download_button("Download anomalies CSV", csv_bytes(flagged), "anomalies.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Bought together
# --------------------------------------------------------------------------- #

with tabs[5]:
    st.subheader("Products bought together")
    st.caption("Lift above 1 means the two products appear in the same order more often than chance. "
               "Confidence is the share of the rarer product's orders that also contained the other one.")
    pairs = res["pairs"]
    if pairs.empty:
        st.info("Not enough multi-item orders to find pairs.")
    else:
        view = pairs.rename(columns={"item_a": "Product A", "item_b": "Product B", "together": "Orders together"})
        st.dataframe(view[["Product A", "Product B", "Orders together", "confidence_pct", "lift"]].round(2),
                     width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

with tabs[6]:
    st.subheader("Data quality")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{rep.rows_in:,}")
    c2.metric("Rows after cleaning", f"{rep.rows_out:,}")
    c3.metric("Memory", f"{rep.mem_after_mb:.1f} MB", f"{rep.mem_after_mb - rep.mem_before_mb:.1f} MB vs raw",
              delta_color="inverse")
    st.caption("Everything the cleaning pipeline fixed in the file (before any filters):")
    st.dataframe(rep.to_frame(), width="stretch", hide_index=True)
    st.subheader("Cleaned data (filtered)")
    st.dataframe(df.head(500), width="stretch", hide_index=True)
    st.download_button("Download filtered clean data", csv_bytes(df, index=False), "clean_sales_filtered.csv", "text/csv")

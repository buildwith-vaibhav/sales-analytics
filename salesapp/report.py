"""Markdown report writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .charts import inr


def md_table(df: pd.DataFrame, formats: dict | None = None, index: bool = True) -> str:
    formats = formats or {}
    frame = df.reset_index() if index else df.copy()
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = row[c]
            if c in formats:
                v = formats[c](v)
            elif isinstance(v, float):
                v = f"{v:,.2f}"
            elif isinstance(v, pd.Timestamp):
                v = f"{v:%Y-%m-%d}"
            cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def _kpi_value(name: str, v: float) -> str:
    if "%" in name:
        return f"{v:.1f}%"
    if any(w in name.lower() for w in ("revenue", "value", "cost")):
        return inr(v)
    return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"


def write_report(path: Path, *, kpis, rep, monthly, forecast, finfo, cat, products, par, retention,
                 rfm_summary, anomalies, pairs, channel) -> Path:
    best = finfo["trend"]
    flagged = anomalies[anomalies["is_anomaly"]].copy()
    avg_ret = retention.mean()
    kpi_df = pd.DataFrame({"Metric": list(kpis), "Value": [_kpi_value(k, v) for k, v in kpis.items()]})
    lines = [
        "# Sales analytics report",
        "",
        f"Period: {monthly.index.min():%b %Y} to {monthly.index.max():%b %Y}. "
        f"Source rows: {rep.rows_in:,} raw, {rep.rows_out:,} after cleaning.",
        "",
        "## Headline KPIs",
        "",
        md_table(kpi_df, index=False),
        "",
        "## Revenue trend and forecast",
        "",
        "![Revenue trend](charts/01_revenue_trend.png)",
        "",
        f"Model: {best} trend x seasonal index, chosen on a 3-month hold-out "
        f"(MAPE linear {finfo['holdout_mape_pct']['linear']:.1f}%, log {finfo['holdout_mape_pct']['log']:.1f}%). "
        f"Bands are +/- 1.96 residual standard deviations ({finfo['residual_sd_pct']:.1f}%).",
        "",
        md_table(forecast, {c: inr for c in forecast.columns}),
        "",
        "## Categories, regions and products",
        "",
        "![Category by region](charts/02_category_region_heatmap.png)",
        "",
        md_table(cat, {"revenue": inr, "units": lambda v: f"{v:,.0f}", "orders": lambda v: f"{v:,.0f}",
                       "avg_discount_pct": lambda v: f"{v:.1f}", "share_pct": lambda v: f"{v:.1f}"}),
        "",
        "Top products:",
        "",
        md_table(products, {"revenue": inr, "units": lambda v: f"{v:,.0f}"}, index=False),
        "",
        "Channel mix:",
        "",
        md_table(channel, {"revenue": inr, "orders": lambda v: f"{v:,.0f}", "share_pct": lambda v: f"{v:.1f}",
                           "aov": inr}),
        "",
        "## Revenue concentration",
        "",
        "![Pareto](charts/03_pareto_concentration.png)",
        "",
        f"{par['customer_share_for_threshold']:.1%} of customers and {par['product_share_for_threshold']:.1%} "
        f"of products account for {par['threshold']:.0%} of delivered revenue.",
        "",
        "## Cohort retention",
        "",
        "![Cohorts](charts/04_cohort_retention.png)",
        "",
        f"Average retention: month 1 {avg_ret.get(1, float('nan')):.1%}, month 3 {avg_ret.get(3, float('nan')):.1%}, "
        f"month 6 {avg_ret.get(6, float('nan')):.1%}, month 12 {avg_ret.get(12, float('nan')):.1%}.",
        "",
        "## RFM segments",
        "",
        "![RFM](charts/05_rfm_segments.png)",
        "",
        md_table(rfm_summary, {"customers": lambda v: f"{v:,.0f}", "avg_recency_days": lambda v: f"{v:.0f}",
                               "avg_orders": lambda v: f"{v:.1f}", "avg_spend": inr, "revenue": inr,
                               "customer_share_pct": lambda v: f"{v:.1f}", "revenue_share_pct": lambda v: f"{v:.1f}"}),
        "",
        "## Anomalies",
        "",
        "![Anomalies](charts/06_anomalies.png)",
        "",
        md_table(flagged[["revenue", "baseline", "robust_z", "uplift_pct", "kind"]],
                 {"revenue": inr, "baseline": inr, "robust_z": lambda v: f"{v:.1f}",
                  "uplift_pct": lambda v: f"{v:+.0f}%"}) if len(flagged) else "No anomalous days found.",
        "",
        "## Products bought together",
        "",
        md_table(pairs, {"together": lambda v: f"{v:,.0f}", "support_pct": lambda v: f"{v:.2f}",
                         "confidence_pct": lambda v: f"{v:.1f}", "lift": lambda v: f"{v:.2f}"}, index=False),
        "",
        "## Data quality log",
        "",
        md_table(rep.to_frame(), {"rows_affected": lambda v: f"{v:,}"}, index=False),
        "",
        f"Memory: {rep.mem_before_mb:.1f} MB raw to {rep.mem_after_mb:.1f} MB after dtype optimisation.",
        "",
    ]
    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
